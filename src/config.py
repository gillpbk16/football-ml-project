from pathlib import Path 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Project Structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model Configuration 
MODEL_VERSION = "1.1.0"
LABEL_MAPPING = {"H": 0, "D": 1, "A": 2}
TARGET_CLASSES = ["H", "D", "A"]
RANDOM_BASELINE = 1.0 / len(TARGET_CLASSES)

# Dataset Configuration 
DATASETS = {
    "teamstats": {
        "train_file": "train.csv",
        "test_file": "test.csv",
        "description": "Rolling team statistics"
    },
    "odds_teams": {
        "train_file": "train_odds.csv",
        "test_file": "test_odds.csv",
        "description": "Betting odds with team encodings"
    }
}

# Model Configuration
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