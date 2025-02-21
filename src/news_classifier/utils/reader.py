from typing import List
import pandas as pd

from news_classifier.utils.models import Claim

def load_claims_from_csv(input_path: str) -> List[Claim]:
    df = pd.read_csv(input_path)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient='records')
    claims = [Claim(**record) for record in records]
    return claims

def load_labels_from_csv(input_path: str) -> List[bool]:
    return pd.read_csv(input_path)["label"].tolist()