import argparse
import json
import logging
import os
from pathlib import Path
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from news_classifier.data_ingestion.ingester import ingest_data
from news_classifier.data_preprocessing.preprocessor import Preprocessor
from news_classifier.feature_engineering.featurizer import Featurizer
from news_classifier.model_development.models import RandomForestModel
from news_classifier.utils.reader import load_claims_from_csv, load_labels_from_csv

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    return parser.parse_args()

def set_random_seed(seed_val: int = 42) -> None:
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_config(config_path: str) -> dict:
    LOGGER.info(f"Loading config from {config_path}")
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def main():
    args = read_args()
    config = load_config(args.config_file)
    set_random_seed(42)

    mlflow.set_tracking_uri("cache/mlruns")
    mlflow.set_experiment(config.get("mlflow_experiment", "default_experiment"))

    with mlflow.start_run() as run:
        LOGGER.info(f"MLflow run_id: {run.info.run_id}")

        # Data Ingestion
        LOGGER.info("Data ingestion...")
        raw_data_paths = config["raw_data_paths"]
        raw_train_df = ingest_data(raw_data_paths["train"])
        raw_test_df = ingest_data(raw_data_paths["test"])
        raw_val_df = ingest_data(raw_data_paths["val"])
        raw_df = pd.concat([raw_train_df, raw_val_df, raw_test_df]).reset_index(drop=True)
        combined_path = raw_data_paths["combined"]
        raw_df.to_csv(combined_path, index=False)
        LOGGER.info(f"Combined data saved to {combined_path}")

        # Data Preprocessing
        LOGGER.info("Data Preprocessing...")
        preprocessor = Preprocessor()
        cleaned_df = preprocessor.clean_data(combined_path)
        X, y = cleaned_df.drop(columns=["label"]), cleaned_df["label"]
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
        )
        clean_data_paths = config["clean_data_paths"]["data"]
        clean_labels_paths = config["clean_data_paths"]["labels"]
        LOGGER.info("Saving cleaned & split datasets.")
        preprocessor.save(X_train, clean_data_paths["train"])
        LOGGER.info(f"Train data saved to {clean_data_paths['train']}")
        preprocessor.save(X_val, clean_data_paths["val"])
        LOGGER.info(f"Validation data saved to {clean_data_paths['val']}")
        preprocessor.save(X_test, clean_data_paths["test"])
        LOGGER.info(f"Test data saved to {clean_data_paths['test']}")
        preprocessor.save(y_train, clean_labels_paths["train"])
        LOGGER.info(f"Train labels saved to {clean_labels_paths['train']}")
        preprocessor.save(y_val, clean_labels_paths["val"])
        LOGGER.info(f"Validation labels saved to {clean_labels_paths['val']}")
        preprocessor.save(y_test, clean_labels_paths["test"])
        LOGGER.info(f"Test labels saved to {clean_labels_paths['test']}")

        # Feature Engineering
        LOGGER.info("Feature Engineering...")
        X_train_df = load_claims_from_csv(clean_data_paths["train"])
        featurizer_path = config["featurizer_path"]
        featurizer = Featurizer()
        featurizer.fit(X_train_df)
        featurizer.save(featurizer_path)
        LOGGER.info(f"Featurizer saved to {featurizer_path}")

        # Model Development
        LOGGER.info("Model Development...")
        X_train = load_claims_from_csv(clean_data_paths["train"])
        X_val = load_claims_from_csv(clean_data_paths["val"])
        X_test = load_claims_from_csv(clean_data_paths["test"])
        y_train = load_labels_from_csv(clean_labels_paths["train"])
        y_val = load_labels_from_csv(clean_labels_paths["val"])
        y_test = load_labels_from_csv(clean_labels_paths["test"])
        model = RandomForestModel(config)
        model.fit(X_train, y_train)
        LOGGER.info("Model fitted.")
        test_results = model.evaluate(X_test, y_test)
        val_results = model.evaluate(X_val, y_val)
        LOGGER.info(f"Test results: {test_results}")
        LOGGER.info(f"Validation results: {val_results}")
        model.save(config["model_path"])
        LOGGER.info(f"Model saved to {config['model_path']}")

        mlflow.log_metrics({f"test_{k}": v for k, v in test_results.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_results.items()})

        rf_params = config.get("params", {})
        mlflow.log_params(rf_params)

        LOGGER.info("Pipeline complete.")

if __name__ == "__main__":
    main()