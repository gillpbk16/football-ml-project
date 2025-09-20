import pandas as pd
import json 
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate import load_data, prepare_features_labels, evaluate_model
from src.config import MODELS_DIR, LABEL_MAPPING
import joblib





def compare_versions():
    _, test_df = load_data("teamstats")
    _, test_X, _, test_y, feature_cols = prepare_features_labels(_, test_df, "teamstats")

    for model_type in ['rf', 'logreg']:
        model_v1_0 = joblib.load(MODELS_DIR / "teamstats" / model_type / "1.0.0" / "model.pkl")
        model_v1_1 = joblib.load(MODELS_DIR / "teamstats" / model_type / "1.1.0" / "model.pkl")

        
        print(f"Available features: {feature_cols}")

        v1_0_features = [
            'rolling_goals_for_home', 'rolling_goals_against_home',  
            'rolling_points_home', 'rolling_win_rate_home',
            'rolling_goals_for_away', 'rolling_goals_against_away',  
            'rolling_points_away', 'rolling_win_rate_away'
        ]

        v1_1_features = feature_cols

        test_X_v1_0 = test_X[v1_0_features]
        test_X_v1_1 = test_X[v1_1_features]

        metrics_v1_0 = evaluate_model(model_v1_0, test_X_v1_0, test_y)
        metrics_v1_1 = evaluate_model(model_v1_1, test_X_v1_1, test_y)

        print(f"\n{model_type.upper()} Comparison:")
        print(f"  v1.0.0 (8 features): {metrics_v1_0['accuracy']:.4f}")
        print(f"  v1.1.0 (14 features): {metrics_v1_1['accuracy']:.4f}")
        print(f"  Improvement: {metrics_v1_1['accuracy'] - metrics_v1_0['accuracy']:+.4f}")

def generate_comparison_report():
    pass  

if __name__ == "__main__":
    compare_versions()


