from pathlib import Path 
import pandas as pd 
from src.data_processing import (load_raw_files, standardise_match_data, create_team_perspective, 
                          add_team_results, combine_and_sort, add_rolling_features, 
                          create_match_dataset, encode_labels, create_train_test_split, save_final_dataset)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW = 5
train_seasons = [2020, 2021, 2022]  
test_seasons = [2023, 2024]   

def main():
    raw_data = load_raw_files(RAW_DIR)
    all_matches = standardise_match_data(raw_data)

    home, away = create_team_perspective(all_matches)
    home, away = add_team_results(home, away)
    team_matches = combine_and_sort(home, away)

    team_features = add_rolling_features(team_matches, window=WINDOW)
    
    match_dataset = create_match_dataset(team_features)   
    match_dataset = encode_labels(match_dataset)

    train_data, test_data = create_train_test_split(match_dataset, train_seasons, test_seasons)
    save_final_dataset(train_data, test_data, match_dataset, PROCESSED_DIR) 

if __name__ == "__main__":
    main()
