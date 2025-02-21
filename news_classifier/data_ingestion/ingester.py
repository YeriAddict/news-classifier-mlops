import pandas as pd

def ingest_data(path: str) -> pd.DataFrame:
    columns = [
        "id", 
        "label", 
        "statement", 
        "subjects", 
        "speaker",
        "speaker_job_title",
        "state_info", 
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context"
    ]
    return pd.read_csv(path, sep="\t", names=columns)