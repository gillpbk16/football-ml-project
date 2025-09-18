from pathlib import Path 
import pandas as pd 
from src.features import (standardise_match_data, create_team_perspective, add_team_results,
                     combine_and_sort, add_rolling_features, create_match_dataset,
                    encode_labels, create_train_test_split
                )

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW = 5
train_seasons = [2020, 2021, 2022]  
test_seasons = [2023, 2024]   

def load_raw_files():
    files = sorted(RAW_DIR.glob("E0_*.csv"))
    if not files: 
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}. Expected E0_YYYY.csv files.")
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        season = f.stem.split("_")[-1]
        df_list.append({ 'dataframe': df, 'season': season })
    return df_list


def save_final_dataset(train_data, test_data, match_data):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    match_data.to_csv(PROCESSED_DIR / "teamstats.csv", index=False)
    train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)

    (PROCESSED_DIR / "README.md").write_text(
        "# Processed Data\n\n"
        "- `teamstats.csv`: one row per match with 5-match rolling features per side (shifted 1 to avoid leakage).\n"
        "- `train.csv`, `test.csv`: split by Season lists.\n\n"
        "## Key columns\n"
        "- Date, HomeTeam, AwayTeam, Season, FTHG, FTAG, FTR (H/D/A)\n"
        "- rolling features: rolling_goals_for_home/away, rolling_goals_against_home/away, "
        "rolling_points_home/away, rolling_win_rate_home/away\n"
    )

def main():
    raw_data = load_raw_files()
    all_matches = standardise_match_data(raw_data)

    home, away = create_team_perspective(all_matches)
    home, away = add_team_results(home, away)
    team_matches = combine_and_sort(home, away)

    team_features = add_rolling_features(team_matches, window=WINDOW)
    
    match_dataset = create_match_dataset(team_features)   
    match_dataset = encode_labels(match_dataset)

    train_data, test_data = create_train_test_split(match_dataset, train_seasons, test_seasons)
    save_final_dataset(train_data, test_data, match_dataset) 

if __name__ == "__main__":
    main()
