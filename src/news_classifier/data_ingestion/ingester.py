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

if __name__ == "__main__":
    raw_train_path = "../../../datasets/raw/train.tsv"
    raw_test_path = "../../../datasets/raw/test.tsv"
    raw_valid_path = "../../../datasets/raw/valid.tsv"
    raw_train_df = ingest_data(raw_train_path)
    raw_test_df = ingest_data(raw_test_path)
    raw_valid_df = ingest_data(raw_valid_path)
    raw_df = pd.concat([raw_train_df, raw_valid_df, raw_test_df])
    raw_df.reset_index(drop=True, inplace=True)
    raw_output_path = "../../../datasets/raw/combined.csv"
    raw_df.to_csv(raw_output_path, index=False)