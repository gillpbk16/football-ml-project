import pandas as pd  
from pathlib import Path  

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import joblib
import json 
from datetime import datetime

# Configuration Constants

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_VERSION = "1.0.0"
LABEL_MAPPING = {"H": 0, "D": 1, "A": 2}
TARGET_CLASSES = ["H", "D", "A"]
RANDOM_BASELINE = 1.0 / len(TARGET_CLASSES)

DATASETS = {
    "teamstats" : {
        "train_file" : "train.csv", 
        "test_file" : "test.csv",  
        "description" : "Rolling team statistics"
    }, 
    "odds_teams" : {
        "train_file" : "train_odds.csv",
        "test_file" : "test_odds.csv",  
        "description" : "Betting odds with team encodings"
    }
}

MODELS = {
    "logreg": {
        "class" : LogisticRegression,
        "teamstats_params": {"solver": "lbfgs", "random_state": 42, "max_iter": 10000, "use_pipeline" : True},
        "odds_teams_params": {"solver": "lbfgs", "random_state": 42, "max_iter": 1000, "use_pipeline" : False}
    },
    "rf": {
        "class" : RandomForestClassifier, 
        "params": {"n_estimators": 500, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1, "use_pipeline" : False}
    }
}

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
        feature_cols = [col for col in train_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])]
    elif dataset_type == "odds_teams":
        prob_cols = ['prob_home', 'prob_draw', 'prob_away']
        home_cols = sorted([col for col in train_df.columns if col.startswith('home_')])
        away_cols = sorted([col for col in train_df.columns if col.startswith('away_')])
        feature_cols = prob_cols + home_cols + away_cols
    
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
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
    else:
        model = base_model
    
    model.fit(train_X, train_y)
    return model

def evaluate_model(model, test_X, test_y):
    predictions = model.predict(test_X)
    probabilities = model.predict_proba(test_X)
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

def save_model(model, dataset_type, model_family, features, metrics, version=MODEL_VERSION):

    save_dir = MODELS_DIR / dataset_type / model_family / version
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "model_info": {
            "name": model_family,
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "random_state": 42
        },
        "dataset_info": {
            "name": dataset_type,
            "feature_count": len(features),
            "features": features
        },
        "performance": {
            "accuracy": round(metrics['accuracy'], 4),
            "baseline_improvement": round(metrics['baseline_improvement'], 4),
            "f1_macro": round(metrics['f1_macro'], 4),
            "precision_macro": round(metrics['precision_macro'], 4),
            "log_loss": round(metrics['log_loss'], 4)
        },
        "evaluation": {
            "confusion_matrix": metrics['confusion_matrix'].tolist(),
            "target_classes": TARGET_CLASSES
        }
    }
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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

def train_models():
    results = {}

    for dataset_type in DATASETS.keys():

        train_df, test_df = load_data(dataset_type)
        train_X, test_X, train_y, test_y, features = prepare_features_labels(train_df, test_df, dataset_type)
        validate_data(train_df, test_df, features)
        for model_name in MODELS.keys():
            
            # Train Model
            model = train_model(train_X, train_y, model_name, dataset_type)
            metrics = evaluate_model(model, test_X, test_y)

            # Store results
            key = f"{dataset_type}_{model_name}"
            results[key] = {
                'model' : model, 
                'metrics' : metrics, 
                'features' : features
            }

            acc = metrics['accuracy']
            improvement = metrics['baseline_improvement']

    return results
    

if __name__ == "__main__":
    try: 
        results = train_models()

        for model_name, data in results.items():
            dataset_type, model_family = model_name.rsplit('_', 1)
            save_model(
                model=data['model'],
                dataset_type=dataset_type,
                model_family=model_family,
                features=data['features'],
                metrics=data['metrics']
            )

        create_evaluation_report(results)
    except Exception as e:
        print(f"Failed: {str(e)}")
        raise
