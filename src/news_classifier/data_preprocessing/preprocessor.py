from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import JOBS_MAPPINGS, LABELS_MAPPINGS, STATES_MAPPINGS

class BasePreprocessor(ABC):
    @abstractmethod
    def clean_data(self, input_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, data: pd.DataFrame, output_path: str) -> None:
        pass

class Preprocessor(BasePreprocessor):
    def __canonicalize_column__(self, label: str, mappings: dict[str, str]) -> str:
        if isinstance(label, str):
            cleaned = label.lower().strip()
            return mappings.get(cleaned, cleaned)
        return label
    
    def clean_data(self, input_path: str) -> pd.DataFrame:
        raw_df = pd.read_csv(input_path)
        raw_df.dropna(subset=["label"], inplace=True)
        raw_df["label"] = raw_df["label"].apply(self.__canonicalize_column__, args=(LABELS_MAPPINGS,))
        raw_df["state_info"] = raw_df["state_info"].apply(self.__canonicalize_column__, args=(STATES_MAPPINGS,))
        raw_df["speaker_job_title"] = raw_df["speaker_job_title"].apply(self.__canonicalize_column__, args=(JOBS_MAPPINGS,))
        raw_df["party_affiliation"] = raw_df["party_affiliation"].str.lower()
        cleaned_df = raw_df
        return cleaned_df

    def save(self, data: pd.DataFrame, output_path: str) -> None:
        data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    raw_input_path = "../../../datasets/raw/combined.csv"
    cleaned_df = preprocessor.clean_data(raw_input_path)

    X, y = cleaned_df.drop(columns=["label"]), cleaned_df["label"]

    # 60% train, 20% test, 20% validation
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )

    preprocessor.save(X_train, "../../../datasets/clean/data/train_data.csv")
    preprocessor.save(X_val, "../../../datasets/clean/data/val_data.csv")
    preprocessor.save(X_test, "../../../datasets/clean/data/test_data.csv")

    preprocessor.save(y_train, "../../../datasets/clean/labels/train_labels.csv")
    preprocessor.save(y_val, "../../../datasets/clean/labels/val_labels.csv")
    preprocessor.save(y_test, "../../../datasets/clean/labels/test_labels.csv")
