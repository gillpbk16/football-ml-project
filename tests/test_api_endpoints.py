import pytest 
import os 
import numpy as np 
import sys 
from fastapi import FastAPI
from fastapi.testclient import TestClient 
from unittest.mock import patch 
import warnings
from sklearn.exceptions import DataConversionWarning

sys.path.append('../api')
from api.app import app 

warnings.filterwarnings("ignore")

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    
    data = r.json()
    assert data["status"] == "ok"
    assert "time" in data

def test_models_endpoint():
    r = client.get("/models")
    assert r.status_code == 200

    data = r.json()
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)
    assert data["count"] == len(data["models"])
    assert data["count"] > 0

def test_predict_valid_request():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Chelsea", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }

    r = client.post("/predict", json=request_data)
    assert r.status_code == 200

    data = r.json()

    required_fields = [
        'request_id', 'home_team', 'away_team',
        'predicted_result', 'probabilities', 'confidence',
        'model_used','model_version','latency_ms', 'timestamp'
    ]

    for field in required_fields:
        assert field in data

    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"
    assert data["predicted_result"] in ["Home Win", "Draw", "Away Win"]
    assert 0 <= data["confidence"] <= 1

def test_predict_invalid_team():
    request_data = {
        "home_team" : "UNKNOWN", 
        "away_team" : "Chelsea", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }

    r = client.post("/predict", json=request_data)
    assert r.status_code == 400

def test_predict_invalid_probabilities():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Chelsea", 
        "prob_home" : 0.8, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }

    r = client.post("/predict", json=request_data)
    assert r.status_code == 400

def test_predict_missing_data():
    request_data = {
        "home_team" : "Arsenal", 
    }

    r = client.post("/predict", json=request_data)
    assert r.status_code == 422
    
@patch.dict(os.environ, {"AB_TESTING_ENABLED": "true"})
def test_predict_ab_testing():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Chelsea", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }

    models_used = set()

    for i in range(50):
        r = client.post("/predict", json=request_data)
        assert r.status_code == 200

        data = r.json()
        models_used.add(data["model_used"])
    assert len(models_used) >= 2


        
