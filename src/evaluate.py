import pandas as pd
import numpy as np 
import json 
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.baselines import MajorityClassBaseline, UniformRandomBaseline, BookmakerBaseline
from src.config import (REPORTS_DIR, DATA_DIR, MODELS_DIR, MODEL_VERSION, LABEL_MAPPING, TARGET_CLASSES, 
                        RANDOM_BASELINE, DATASETS)

import joblib


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

def prepare_features_labels(train_df, test_df, dataset_type):
    train_y = train_df['FTR'].map(LABEL_MAPPING)
    test_y = test_df['FTR'].map(LABEL_MAPPING)

    if dataset_type == "teamstats":
        exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season']
        feature_cols = [col for col in train_df.columns if col.startswith('rolling_')]
    elif dataset_type == "odds_teams":
        prob_cols = ['prob_home', 'prob_draw', 'prob_away']
        home_cols = sorted([col for col in train_df.columns if col.startswith('home_')])
        away_cols = sorted([col for col in train_df.columns if col.startswith('away_')])
        feature_cols = prob_cols + home_cols + away_cols
    
    train_X = train_df[feature_cols].copy()
    test_X = test_df[feature_cols].copy()
    
    return train_X, test_X, train_y, test_y, feature_cols

def evaluate_baselines():
    results = {}

    train_df_odds, test_df_odds = load_data("odds_teams")
    _, _, train_y_odds, test_y_odds, _ = prepare_features_labels(train_df_odds, test_df_odds, "odds_teams")

    baselines = {
        'majority_baseline': MajorityClassBaseline(),
        'uniform_baseline': UniformRandomBaseline(),
        'bookmaker_baseline': BookmakerBaseline()
    }

    for baseline_name, baseline in baselines.items():
        if baseline_name == 'majority_baseline':
            baseline.fit(train_y_odds)
            probs = baseline.predict_proba(test_df_odds)
        elif baseline_name == 'uniform_baseline':
            probs = baseline.predict_proba(test_df_odds)
        else:  
            probs = baseline.predict_proba(test_df_odds)
        assert probs.shape[1] == len(TARGET_CLASSES), "baseline probs must be (n, 3) in H,D,A order"

        preds = np.argmax(probs, axis=1)
        metrics = {
            'accuracy': accuracy_score(test_y_odds, preds),
            'log_loss': log_loss(test_y_odds, probs),
            'baseline_improvement': accuracy_score(test_y_odds, preds) - RANDOM_BASELINE,
            'f1_macro': f1_score(test_y_odds, preds, average='macro')
        }
        
        results[baseline_name] = {'metrics': metrics}
    
    return results

def evaluate_model(model, test_X, test_y):
    predictions = model.predict(test_X)

    probabilities = model.predict_proba(test_X)
    canonical_ids = np.array([LABEL_MAPPING[c] for c in TARGET_CLASSES])
    idx = [int(np.where(model.classes_ == k)[0][0]) for k in canonical_ids]
    probabilities = probabilities[:, idx]

    accuracy = accuracy_score(test_y, predictions)

    logloss = log_loss(test_y, probabilities)
    conf_matrix = confusion_matrix(test_y, predictions)

    f1_macro = f1_score(test_y, predictions, average='macro')
    precision_macro = precision_score(test_y, predictions, average='macro')
    recall_macro = recall_score(test_y, predictions, average='macro')
    
    improvement = accuracy - RANDOM_BASELINE
    max_probs = probabilities.max(axis=1)

    return {
        'accuracy': accuracy, 
        'log_loss': logloss,
        'f1_macro' : f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'baseline_improvement': improvement,
        'confusion_matrix': conf_matrix,
        'prob_confidence_mean': max_probs.mean(),
        'prob_confidence_std': max_probs.std()
    }


def create_evaluation_report(results):
    reports_dir = REPORTS_DIR / "baseline"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Model Evaluation Report\n"]
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report_lines.append("## Model Performance Summary\n")
    report_lines.append("| Dataset | Model | Accuracy | F1-Macro | Baseline Improvement |\n")
    report_lines.append("|---------|-------|----------|----------|---------------------|\n")

    for model_name, data in results.items():
        dataset_type, model_family = model_name.rsplit('_', 1)
        metrics = data['metrics']
        report_lines.append(
            f"| {dataset_type} | {model_family} | "
            f"{metrics['accuracy']:.3f} | {metrics['f1_macro']:.3f} | "
            f"{metrics['baseline_improvement']:.3f} |\n"
        )
    
    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['accuracy'])
    best_acc = results[best_model]['metrics']['accuracy']

    report_lines.append(f"\n## Best Model\n")
    report_lines.append(f"**{best_model}** - Accuracy: {best_acc:.3f}\n")

    report_path = reports_dir / "eval_log.md"
    with open(report_path, 'w') as f: 
        f.writelines(report_lines)
    
    json_metrics = {}
    for model_name, data in results.items():
        metrics = data['metrics']
        json_metrics[model_name] = {
            'accuracy': round(metrics['accuracy'], 4),
            'log_loss': round(metrics['log_loss'], 4),
            'f1_macro': round(metrics['f1_macro'], 4),
            'baseline_improvement': round(metrics['baseline_improvement'], 4)
        }

    # Save JSON format
    json_path = reports_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)


if __name__ == "__main__":
    results = {}

    for dataset_type in ("teamstats", "odds_teams"):
        _, test_df = load_data(dataset_type)
        y = test_df["FTR"].map(LABEL_MAPPING).astype(int)

        for family in ("logreg", "rf"):
            model_dir = MODELS_DIR / dataset_type / family / MODEL_VERSION
            model_path = model_dir / "model.pkl"
            meta_path = model_dir / "metadata.json"
            if not model_path.exists() or not meta_path.exists():
                print(f"[skip] Missing artifacts for {dataset_type}/{family}/{MODEL_VERSION}")
                continue

            meta = json.load(open(meta_path))
            feature_order = meta["dataset_info"]["features"]
            X = test_df[feature_order].copy()

            model = joblib.load(model_path)
            metrics = evaluate_model(model, X, y)
            results[f"{dataset_type}_{family}"] = {"metrics": metrics}

    create_evaluation_report(results)
    print(f"Wrote report to {REPORTS_DIR / 'baseline'}")
