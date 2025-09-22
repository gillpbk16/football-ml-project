from fastapi import FastAPI, HTTPException
from datetime import datetime
from pydantic import BaseModel

from pathlib import Path

import json
import joblib
import numpy as np
import time
import uuid
import random
import os
import uvicorn 

from fastapi.middleware.cors import CORSMiddleware

TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich",
    "Leeds",
    "Leicester",
    "Liverpool",
    "Luton",
    "Man City",
    "Man United",
    "Newcastle",
    "Norwich",
    "Nott'm Forest",
    "Sheffield United",
    "Southampton",
    "Tottenham",
    "Watford",
    "West Brom",
    "West Ham",
    "Wolves",
]


def create_odds_features(
    home_team: str,
    away_team: str,
    prob_home: float = 0.4,
    prob_draw: float = 0.3,
    prob_away: float = 0.3,
):

    features = [prob_home, prob_draw, prob_away]

    for team in sorted(TEAMS):
        features.append(1.0 if team == home_team else 0.0)

    for team in sorted(TEAMS):
        features.append(1.0 if team == away_team else 0.0)

    return np.array(features).reshape(1, -1)


def get_ab_model(default_model: str) -> str:
    ab_enabled = os.getenv("AB_TESTING_ENABLED", "false").lower() == "true"

    if not ab_enabled:
        return default_model

    if random.random() < 0.5:
        return default_model.replace("_rf_", "_logreg_")
    else:
        return default_model


app = FastAPI(title="Football ML Project", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Football ML API", "time": datetime.now()}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now()}


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.metadata = {}
        self.load_models()

    def load_models(self):
        models_dir = Path(__file__).parent.parent / "models"
        if not models_dir.exists():
            return

        for pkl_file in models_dir.glob("**/*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    model = joblib.load(pkl_file)

                relative_path = pkl_file.relative_to(models_dir)
                model_key = str(relative_path.parent).replace("/", "_")

                self.models[model_key] = model

                metadata_file = pkl_file.parent / "metadata.json"

                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        self.metadata[model_key] = json.load(f)

            except Exception as e:
                raise Exception(f"Warning: Could not load model {pkl_file}: {e}")

    def list_models(self):
        return list(self.models.keys())

    def predict(self, model_key: str, features):
        if model_key not in self.models:
            raise Exception(f"Model {model_key} not found")

        model = self.models[model_key]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        result_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        predicted_result = result_mapping[prediction]

        prob_dict = {
            "Home Win": float(probability[0]),
            "Draw": float(probability[1]),
            "Away Win": float(probability[2]),
        }

        return predicted_result, prob_dict


model_registry = ModelRegistry()


@app.get("/models")
async def get_models():
    return {"models": model_registry.list_models(), "count": len(model_registry.models)}


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    prob_home: float = 0.4
    prob_away: float = 0.3
    prob_draw: float = 0.3
    model_key: str = "odds_teams_rf_1.0.0"


class PredictionResponse(BaseModel):
    request_id: str
    home_team: str
    away_team: str
    predicted_result: str
    probabilities: dict
    confidence: float
    model_used: str
    model_version: str
    latency_ms: float
    timestamp: datetime


@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    try:

        home_team_normalised = request.home_team.strip().title()
        away_team_normalised = request.away_team.strip().title()

        if home_team_normalised == away_team_normalised:
            raise HTTPException(
                status_code=400, detail="Home team and away team cannot be the same"
            )

        total_prob = request.prob_home + request.prob_draw + request.prob_away
        if abs(total_prob - 1.0) > 0.1:
            raise HTTPException(
                status_code=400, detail="Probabilities should sum to approximately 1."
            )

        if home_team_normalised not in TEAMS or away_team_normalised not in TEAMS:
            raise HTTPException(status_code=400, detail="Invalid team.")

        ab_model_key = get_ab_model(request.model_key)

        features = create_odds_features(
            home_team_normalised,
            away_team_normalised,
            request.prob_home,
            request.prob_draw,
            request.prob_away,
        )

        predicted_result, probabilities = model_registry.predict(ab_model_key, features)

        latency_ms = (time.time() - start_time) * 1000
        confidence = max(probabilities.values())

        return PredictionResponse(
            request_id=request_id,
            home_team=home_team_normalised,
            away_team=away_team_normalised,
            predicted_result=predicted_result,
            probabilities=probabilities,
            confidence=confidence,
            model_used=ab_model_key,
            model_version="1.0.0",
            latency_ms=latency_ms,
            timestamp=datetime.now(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
