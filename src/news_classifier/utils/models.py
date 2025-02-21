from typing import Optional
from pydantic import BaseModel

class Claim(BaseModel):
    id: Optional[str]
    statement: str
    subjects: Optional[str]
    speaker: Optional[str]
    speaker_job_title: Optional[str]
    state_info: Optional[str]
    party_affiliation: Optional[str]
    barely_true_counts: float
    false_counts: float
    half_true_counts: float
    mostly_true_counts: float
    pants_on_fire_counts: float
    context: Optional[str]