import pandas as pd 

def standardise_match_data(df_list):
    standardised_dfs = []
    for dfs in df_list:
        df = dfs['dataframe']
        season = dfs['season']

        df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, errors = "coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        df["Season"] = int(season)

        standardised_dfs.append(df)
    return pd.concat(standardised_dfs, ignore_index=True)

def create_team_perspective(df):
    home = df.rename(columns = {
        "HomeTeam" : "Team", 
        "AwayTeam": "Opponent",
        "FTHG": "GF", 
        "FTAG" : "GA"
    }).copy()
    home["Venue"] = "H"

    away = df.rename(columns = {
        "AwayTeam" : "Team", 
        "HomeTeam": "Opponent",
        "FTAG": "GF", 
        "FTHG" : "GA"
    }).copy()
    away["Venue"] = "A"

    keep_cols = ["Date", "Team", "Opponent", "Venue", "GF", "GA", "FTR", "Season"]
    home = home[keep_cols]
    away = away[keep_cols]

    return home, away

def add_team_results(home, away):
    map_home = {"H": "W", "D": "D", "A": "L"}
    map_away = {"H": "L", "D": "D", "A": "W"}

    home["Result"] = home["FTR"].map(map_home)
    away["Result"] = away["FTR"].map(map_away)

    map_points = {"W": 3, "D" : 1, "L": 0}
    home["Points"] = home["Result"].map(map_points)
    away["Points"] = away["Result"].map(map_points)

    return home, away

def combine_and_sort(home, away):
    team_matches = pd.concat([home, away], ignore_index = True)
    return team_matches.sort_values(["Team", "Date"]).reset_index(drop=True)


def add_rolling_features(df, window=5):
    g = df.sort_values(["Team", "Date"]).copy()
    
    g["GF_lag"] = g.groupby("Team")["GF"].shift(1)
    g["GA_lag"] = g.groupby("Team")["GA"].shift(1)
    g["Pts_lag"] = g.groupby("Team")["Points"].shift(1)
    g["Win_lag"] = (g["Result"] == "W").shift(1)

    g["rolling_goals_for"] = g.groupby("Team")["GF_lag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    g["rolling_goals_against"] = g.groupby("Team")["GA_lag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    g["rolling_points"] = g.groupby("Team")["Pts_lag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    g["rolling_win_rate"] = g.groupby("Team")["Win_lag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)

    return g.drop(columns=["GF_lag","GA_lag","Pts_lag","Win_lag"])


def create_match_dataset(team_features):
    home_records = team_features[team_features["Venue"] == "H"].copy()
    away_records = team_features[team_features["Venue"] == "A"].copy()

    rolling_cols = ['rolling_goals_for', 'rolling_goals_against', 
                   'rolling_points', 'rolling_win_rate']
    
    home_data = home_records[['Date', 'Team', 'Opponent', 'Season', 
                             'GF', 'GA', 'Points', 'Result'] + rolling_cols].copy()
    
    for col in rolling_cols:
        home_data[f"{col}_home"] = home_data.pop(col)
    home_data.rename(columns={'Team':'HomeTeam','Opponent':'AwayTeam'}, inplace=True)

    away_data = away_records[['Date','Team','Opponent','Season'] + rolling_cols].copy()
    for col in rolling_cols:
        away_data[f"{col}_away"] = away_data.pop(col)
    away_data.rename(columns={'Team':'AwayTeam','Opponent':'HomeTeam'}, inplace=True)

    assert not home_data.duplicated(['Date','HomeTeam','AwayTeam','Season']).any(), "Duplicate home keys found"
    assert not away_data.duplicated(['Date','HomeTeam','AwayTeam','Season']).any(), "Duplicate away keys found"
    
    match_data = home_data.merge(
    away_data,
    on=['Date','HomeTeam','AwayTeam','Season'],
    how='inner',
    validate='one_to_one',   # <— add this
    )

    need = [f"{c}_home" for c in rolling_cols] + [f"{c}_away" for c in rolling_cols]
    match_data = match_data.dropna(subset=need)

    match_data = match_data.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    return match_data

def encode_labels(match_data):
    label_mapping = {'W' : 'H','D' : 'D', "L" : 'A'}
    match_data['FTR'] = match_data['Result'].map(label_mapping).astype('category')

    return match_data

def create_train_test_split(match_data, train_seasons, test_seasons):
    match_data = match_data.sort_values('Date').reset_index(drop=True)
      
    train_data = match_data[match_data['Season'].isin(train_seasons)].copy()
    test_data = match_data[match_data['Season'].isin(test_seasons)].copy()
    
    return train_data, test_data