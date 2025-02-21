from abc import ABC, abstractmethod
import pickle
from typing import List

import logging

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline

from news_classifier.utils.models import Claim
from news_classifier.utils.reader import load_claims_from_csv

LOGGER = logging.getLogger(__name__)

class BaseFeaturizer(ABC):
    @abstractmethod
    def fit(self, claims: List[Claim]) -> None:
        pass

    @abstractmethod
    def transform(self, claims: List[Claim]) -> np.array:
        pass

    @abstractmethod
    def save(self, output_path: str) -> None:
        pass

class Featurizer(BaseFeaturizer):
    def __init__(self):
        self.combined_featurizer = None
        self.optimal_credit_bins = {}

    def __compute_optimal_credit_bins__(self, claims: List[Claim]) -> dict:
        optimal_credit_bins = {}
        for credit_name in ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]:
            values = np.array([getattr(claim, credit_name) for claim in claims], dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                optimal_credit_bins[credit_name] = [0, 1]
            else:
                optimal_credit_bins[credit_name] = list(np.histogram_bin_edges(finite_values, bins=10))
        return optimal_credit_bins
    
    def __compute_bin_index__(self, val: float, bins: np.ndarray) -> int:
        for idx, bin_val in enumerate(bins):
            if val <= bin_val:
                return idx
        return len(bins) - 1

    def __extract_manual_features__(self, claims: List[Claim]) -> List[dict]:
        manual_features = []
        for claim in claims:
            features = {}
            features["party_affiliation"] = claim.party_affiliation
            features["speaker"] = claim.speaker
            features["state_info"] = claim.state_info

            claim = dict(claim)
            for credit_name in ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]:
                features[credit_name] = str(self.__compute_bin_index__(claim[credit_name], self.optimal_credit_bins[credit_name]))

            manual_features.append(features)
        return manual_features
    
    def __extract_text_features__(self, claims: List[Claim]) -> List[str]:
        return [claim.statement for claim in claims]

    def fit(self, claims: List[Claim]) -> None:
        self.optimal_credit_bins = self.__compute_optimal_credit_bins__(claims)

        dict_featurizer = DictVectorizer()
        tfidf_featurizer = TfidfVectorizer()

        manual_features_transformer = FunctionTransformer(self.__extract_manual_features__)
        textual_features_transformer = FunctionTransformer(self.__extract_text_features__)

        manual_feature_pipeline = Pipeline([
            ("manual_features", manual_features_transformer),
            ("manual_featurizer", dict_featurizer)
        ])

        text_feature_pipeline = Pipeline([
            ("textual_features", textual_features_transformer),
            ("textual_featurizer", tfidf_featurizer)
        ])

        self.combined_featurizer = FeatureUnion([
            ("manual_feature_pipeline", manual_feature_pipeline),
            ("textual_feature_pipeline", text_feature_pipeline)
        ])

        self.combined_featurizer.fit(claims)
        LOGGER.info("Featurizer has been fitted.")

    def transform(self, claims: List[Claim]) -> np.array:
        if self.combined_featurizer is None:
            raise ValueError("Featurizer has not been fitted yet.")
        return self.combined_featurizer.transform(claims)

    def save(self, output_path: str) -> None:
        if self.combined_featurizer is None:
            raise ValueError("Featurizer has not been fitted yet, cannot save.")
        LOGGER.info(f"Saving featurizer to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(self.combined_featurizer, f)
            LOGGER.info("Featurizer fitted and saved successfully.")

if __name__ == "__main__":
    X_train = load_claims_from_csv("../../../datasets/clean/data/train_data.csv")
    
    featurizer = Featurizer()
    featurizer.fit(X_train)
    featurizer.save("../../../cache/featurizer.pkl")