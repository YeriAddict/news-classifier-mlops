from abc import ABC, abstractmethod
import pandas as pd
from news_classifier.utils.constants import JOBS_MAPPINGS, LABELS_MAPPINGS, STATES_MAPPINGS

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