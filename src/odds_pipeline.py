from pathlib import Path
import pandas as pd 

from src.features import (load_raw_files, extract_odds_data, convert_odds_to_probs,
                         add_team_one_hot_encoding, create_train_test_split, save_odds_dataset)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW = 5
train_seasons = [2020, 2021, 2022]
test_seasons = [2023, 2024]

def main():
    raw_data = load_raw_files(RAW_DIR)
    odds_matches = extract_odds_data(raw_data) 
    prob_data = convert_odds_to_probs(odds_matches)  
    final_data = add_team_one_hot_encoding(prob_data) 
    train_data, test_data = create_train_test_split(final_data, train_seasons, test_seasons)  
    save_odds_dataset(train_data, test_data, final_data, PROCESSED_DIR) 


if __name__ == "__main__":
    main()