import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import joblib
import json
from datetime import datetime

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    MODEL_VERSION,
    MODELS,
    LABEL_MAPPING,
    TARGET_CLASSES,
    DATASETS,
)


def load_data(dataset_type="teamstats"):
    if dataset_type not in DATASETS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    config = DATASETS[dataset_type]
    train_path = DATA_DIR / config["train_file"]
    test_path = DATA_DIR / config["test_file"]
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing data files for {dataset_type}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def validate_data(train_df, test_df, feature_cols):
    missing_train = [col for col in feature_cols if col not in train_df.columns]
    missing_test = [col for col in feature_cols if col not in test_df.columns]

    if missing_train or missing_test:
        raise ValueError("Missing Features")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Empty dataset detected")

    if train_df[feature_cols].isnull().any().any():
        raise ValueError("NaN values detected in training features")

    if test_df[feature_cols].isnull().any().any():
        raise ValueError("NaN values detected in test features")


def prepare_features_labels(train_df, test_df, dataset_type):
    train_y = train_df["FTR"].map(LABEL_MAPPING)
    test_y = test_df["FTR"].map(LABEL_MAPPING)

    if train_y.isnull().any() or test_y.isnull().any():
        bad_train = set(train_df.loc[train_y.isnull(), "FTR"])
        bad_test = set(test_df.loc[test_y.isnull(), "FTR"])
        raise ValueError(
            f"Unmapped labels found. Train: {bad_train}  Test: {bad_test}. "
            f"Check LABEL_MAPPING and FTR values."
        )

    if dataset_type == "teamstats":
        feature_cols = [col for col in train_df.columns if col.startswith("rolling_")]
        feature_cols = [
            col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])
        ]
    elif dataset_type == "odds_teams":
        prob_cols = ["prob_home", "prob_draw", "prob_away"]
        home_cols = sorted([col for col in train_df.columns if col.startswith("home_")])
        away_cols = sorted([col for col in train_df.columns if col.startswith("away_")])
        feature_cols = prob_cols + home_cols + away_cols
        feature_cols = [col for col in feature_cols if col in test_df.columns]

    train_X = train_df[feature_cols].copy()
    test_X = test_df[feature_cols].copy()

    return train_X, test_X, train_y, test_y, feature_cols


def train_model(train_X, train_y, model_name, dataset_type):
    if len(train_X) != len(train_y):
        raise ValueError("X and Y must have the same number of samples")

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_config = MODELS[model_name]

    if f"{dataset_type}_params" in model_config:
        params = model_config[f"{dataset_type}_params"].copy()
    elif "params" in model_config:
        params = model_config["params"].copy()
    else:
        raise ValueError(f"No parameters found for {model_name}")

    use_pipeline = params.pop("use_pipeline", False)

    base_model = model_config["class"](**params)

    if use_pipeline:
        model = Pipeline([("scaler", StandardScaler()), ("classifier", base_model)])
    else:
        model = base_model

    model.fit(train_X, train_y)
    return model


def save_model(model, dataset_type, model_family, features, version=MODEL_VERSION):
    save_dir = MODELS_DIR / dataset_type / model_family / version
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "model_info": {
            "name": model_family,
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "random_state": 42,
        },
        "dataset_info": {
            "name": dataset_type,
            "feature_count": len(features),
            "features": features,
        },
        "labels": {"mapping": LABEL_MAPPING, "class_order": TARGET_CLASSES},
    }

    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def train_models():
    results = {}

    for dataset_type in DATASETS.keys():

        train_df, test_df = load_data(dataset_type)
        train_X, test_X, train_y, test_y, features = prepare_features_labels(
            train_df, test_df, dataset_type
        )
        validate_data(train_df, test_df, features)
        for model_name in MODELS.keys():

            # Train Model
            model = train_model(train_X, train_y, model_name, dataset_type)

            # Store results
            key = f"{dataset_type}_{model_name}"
            results[key] = {"model": model, "features": features}
    return results


if __name__ == "__main__":
    try:
        results = train_models()

        for model_name, data in results.items():
            dataset_type, model_family = model_name.rsplit("_", 1)
            save_model(
                model=data["model"],
                dataset_type=dataset_type,
                model_family=model_family,
                features=data["features"],
            )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
