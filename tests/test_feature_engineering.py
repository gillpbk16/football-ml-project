import pytest
import numpy as np 
import os
from unittest.mock import patch 

import sys 
sys.path.append('../api')
from api.app import create_odds_features, get_ab_model, TEAMS

def test_create_odds_features():

    features = create_odds_features("Arsenal", "Chelsea", 0.5, 0.3, 0.2)

    assert features.shape == (1, 57), f"Features shape {features.shape}"

    assert features[0][0] == 0.5, f"Expected prob_home 0.5, instead {features[0][0]}"
    assert features[0][1] == 0.3, f"Expected prob_draw 0.3, instead {features[0][1]}"
    assert features[0][2] == 0.2, f"Expected prob_away 0.2, instead {features[0][2]}"

    assert features[0][3] == 1.0, f"Expected Arsenal=1.0 at position 3, instead {features[0][3]}"

    for i, value in enumerate(features[0]):
        assert isinstance(value, (int, float, np.number)), f"Non-numeric value at position {i}: {value}"

    home_encoding = features[0][3:30]
    assert np.sum(home_encoding) == 1, "Expected exactly one 1.0 in home encoding"

    away_encoding = features[0][30:56]
    assert np.sum(away_encoding) == 1, "Expected exactly one 1.0 in away encoding"


def test_create_odds_features_team_encoding():

    arsenal_features = create_odds_features("Arsenal", "Chelsea", 0.4, 0.3, 0.3)
    liverpool_features = create_odds_features("Liverpool", "Chelsea", 0.4, 0.3, 0.3)

    assert not np.array_equal(arsenal_features[0][3:30], liverpool_features[0][3:30])

    chelsea_away = create_odds_features("Arsenal", "Chelsea", 0.4, 0.3, 0.3)
    city_away = create_odds_features("Arsenal", "Man City", 0.4, 0.3, 0.3)

    assert not np.array_equal(chelsea_away[0][30:56], city_away[0][30:56])

    features1 = create_odds_features("Arsenal", "Chelsea", 0.5, 0.3, 0.2)
    features2 = create_odds_features("Arsenal", "Chelsea", 0.5, 0.3, 0.2)
    
    assert np.array_equal(features1, features2)


@patch.dict(os.environ, {"AB_TESTING_ENABLED": "false"})
def test_get_ab_model_disabled():

    original_model = "odds_teams_rf_1.0.0"
    result = get_ab_model(original_model)

    print(f"Input: {original_model}")
    print(f"Output: {result}")

    assert result == original_model


@patch.dict(os.environ, {"AB_TESTING_ENABLED": "true"})
def test_get_ab_model_enabled():
    
    original_model = "odds_teams_rf_1.0.0"
    expected_logreg = "odds_teams_logreg_1.0.0"
    
    results = []

    for i in range(50):
        result = get_ab_model(original_model)
        results.append(result)

    unique_results = set(results)
    print(f"Unique Restults: {unique_results}")

    assert original_model in results, f"Should return original rf model at some frequency"
    assert expected_logreg in results, f"Should return LogReg model at some frequency"

    assert len(unique_results) == 2, f"Only expecting two variants"