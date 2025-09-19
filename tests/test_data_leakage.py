import pandas as pd
from pathlib import Path  

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def test_temporal_split():
    # Load CSV files
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")
    if not train_path.exists() or not test_path.exists():
        raise ValueError(f"Missing data files")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Convert date columns to date/time
    train_df["Date"] = pd.to_datetime(train_df["Date"], errors="coerce")
    test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")

    # Find max train date and min test date
    max_train_date = train_df["Date"].max()
    min_test_date = test_df["Date"].min()

    # Step 4: Assert max_train < min_test with good error message
    assert max_train_date < min_test_date, f"Leakage, Max train date > Min test date"

def test_rolling_features_exist():
    # Load Train data
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise ValueError(f"Missing data files")
    train_df = pd.read_csv(train_path)

    # Find rolling feature columns
    rolling_cols = [col for col in train_df.columns if col.startswith("rolling_")]
    assert len(rolling_cols) > 0, "No Rolling features found"

    # Check they exist and have data
    for col in rolling_cols: 
        non_null_count = train_df[col].notna().sum()
        # Assert with good error messages
        assert non_null_count > 0, f"Rolling feature {col} has no data (all null)"

        
def test_no_obvious_leakage():
    # Load train data
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise ValueError(f"Missing data files")
    train_df = pd.read_csv(train_path)

    # Get rolling feature columns
    rolling_cols = [col for col in train_df.columns if col.startswith("rolling_")]
    assert len(rolling_cols) > 0, "No Rolling features found"

    # Quick ML test - features shouldn't be too predictive
    X = train_df[rolling_cols].fillna(0)
    y = train_df['FTR']
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')

    # Assert reasonable accuracy threshold
    max_accuracy = scores.max()
    assert max_accuracy < 0.85, f"Suspiciously high accuracy ({max_accuracy:.3f}) - possible data leakage"
