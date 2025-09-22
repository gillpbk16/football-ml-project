import pandas as pd
import json
from pathlib import Path

from src.evaluate import (
    load_data,
    evaluate_model,
    create_evaluation_report,
)
from src.config import MODELS_DIR, LABEL_MAPPING
import joblib


def compare_versions():
    """Compare model performance between model versions"""
    _, test_df = load_data("teamstats")
    test_y = test_df["FTR"].map(LABEL_MAPPING)
    feature_cols = [col for col in test_df.columns if col.startswith("rolling_")]
    test_X = test_df[feature_cols].copy()

    for model_type in ["rf", "logreg"]:
        try:

            model_v1_0 = joblib.load(
                MODELS_DIR / "teamstats" / model_type / "1.0.0" / "model.pkl"
            )
            model_v1_1 = joblib.load(
                MODELS_DIR / "teamstats" / model_type / "1.1.0" / "model.pkl"
            )

            meta_path_v1_0 = (
                MODELS_DIR / "teamstats" / model_type / "1.0.0" / "metadata.json"
            )
            meta_v1_0 = json.load(open(meta_path_v1_0))
            v1_0_features = meta_v1_0["dataset_info"]["features"]

            v1_1_features = feature_cols

            test_X_v1_0 = test_X[v1_0_features]
            test_X_v1_1 = test_X[v1_1_features]

            metrics_v1_0 = evaluate_model(model_v1_0, test_X_v1_0, test_y)
            metrics_v1_1 = evaluate_model(model_v1_1, test_X_v1_1, test_y)

            print(f"\n{model_type.upper()} Comparison:")
            print(f"  v1.0.0 (8 features): {metrics_v1_0['accuracy']:.4f}")
            print(f"  v1.1.0 (14 features): {metrics_v1_1['accuracy']:.4f}")
            print(
                f"  Improvement: {metrics_v1_1['accuracy'] - metrics_v1_0['accuracy']:+.4f}"
            )
        except FileNotFoundError:
            print(f"Missing files for {model_type} - skipping comparison")
            continue
        except Exception as e:
            print(f"Error comparing {model_type}: {e}")
            continue


def generate_comparison_report():
    """Generate comparison report"""
    results = {}

    for dataset_type in ("teamstats", "odds_teams"):
        _, test_df = load_data(dataset_type)
        y = test_df["FTR"].map(LABEL_MAPPING).astype(int)

        for family in ("logreg", "rf"):
            for version in ["1.0.0", "1.1.0"]:
                model_dir = MODELS_DIR / dataset_type / family / version
                model_path = model_dir / "model.pkl"
                meta_path = model_dir / "metadata.json"

                if not model_path.exists() or not meta_path.exists():
                    continue

                meta = json.load(open(meta_path))
                feature_order = meta["dataset_info"]["features"]
                X = test_df[feature_order].copy()

                model = joblib.load(model_path)
                metrics = evaluate_model(model, X, y)
                results[f"{dataset_type}_{family}_{version}"] = {"metrics": metrics}

    create_evaluation_report(results)
    print("Version comparison report generated")


if __name__ == "__main__":
    compare_versions()
