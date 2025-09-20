import pytest 
from fastapi import FastAPI
from fastapi.testclient import TestClient 
import sys
import os
import warnings
from sklearn.exceptions import DataConversionWarning

sys.path.append('../api')
from api.app import app 

warnings.filterwarnings("ignore")
client = TestClient(app)

def test_predict_same_teams():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Arsenal", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }

    r = client.post("/predict", json=request_data)
    assert r.status_code == 400
    
    data = r.json()
    assert "same" in data["detail"].lower()



def test_predict_zero_probabilities():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Chelsea", 
        "prob_home" : 1.0, 
        "prob_draw" : 0.0, 
        "prob_away" : 0.0
    }
    r = client.post("/predict", json=request_data)
    assert r.status_code == 200

    data = r.json()
    assert data["predicted_result"] in ["Home Win", "Draw", "Away Win"]
    print(f"Zero draw/away probs: {data['predicted_result']} with {data['confidence']:.3f} confidence")
   


def test_predict_extreme_probabilities():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "Chelsea", 
        "prob_home" : 0.98, 
        "prob_draw" : 0.01, 
        "prob_away" : 0.01
    }
    r = client.post("/predict", json=request_data)
    assert r.status_code == 200

    data = r.json()

    assert data["predicted_result"] in ["Home Win", "Draw", "Away Win"]
    assert 0 <= data["confidence"] <= 1
    
    print(f"With 98% home odds, model predicted: {data['predicted_result']} with {data['confidence']:.3f} confidence")


def test_predict_case_insensitive_teams():
    request_data = {
        "home_team" : "Arsenal", 
        "away_team" : "CHELSEA", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }
    r = client.post("/predict", json=request_data)
    assert r.status_code == 200

    data = r.json()
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"

def test_predict_team_names_with_spaces():
    request_data = {
        "home_team" : " Arsenal", 
        "away_team" : "CHELSEA ", 
        "prob_home" : 0.4, 
        "prob_draw" : 0.3, 
        "prob_away" : 0.3
    }
    r = client.post("/predict", json=request_data)
    assert r.status_code == 200

    data = r.json()
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"



