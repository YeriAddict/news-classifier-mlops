import logging
import json
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from news_classifier.model_development.models import RandomForestModel
from news_classifier.utils.models import Claim

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

class Settings(BaseSettings):
    config_path: str = "config/random_forest.json"

settings = Settings()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(settings.config_path, "r") as f:
    config = json.load(f)

model = RandomForestModel(config)

with open(config["featurizer_path"], "rb") as f:
    model.featurizer = pickle.load(f)

with open(config["model_path"], "rb") as f:
    model.model = pickle.load(f)

class Statement(BaseModel):
    text: str

class Prediction(BaseModel):
    label: int

def construct_claim(text: str) -> Claim:
    return Claim(**{
        "statement": text,
        "id": None,
        "subjects": None,
        "speaker": None,
        "speaker_job_title": None,
        "state_info": None,
        "party_affiliation": None,
        "barely_true_counts": float("nan"),
        "false_counts": float("nan"),
        "half_true_counts": float("nan"),
        "mostly_true_counts": float("nan"),
        "pants_on_fire_counts": float("nan"),
        "context": None,
    })

@app.post("/api/predict", response_model=Prediction)
def predict(statement: Statement):
    try:
        claim = construct_claim(statement.text)
        label = model.predict([claim])
        prediction = Prediction(label=label)
        LOGGER.info(f"Prediction: {prediction}")
        return prediction
    except Exception as e:
        LOGGER.error("Prediction error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
