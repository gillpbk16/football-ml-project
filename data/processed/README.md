# Processed Data

- `teamstats.csv`: one row per match with 5-match rolling features per side (shifted 1 to avoid leakage).
- `train.csv`, `test.csv`: split by Season lists.

## Key columns
- Date, HomeTeam, AwayTeam, Season, FTHG, FTAG, FTR (H/D/A)
- rolling features: rolling_goals_for_home/away, rolling_goals_against_home/away, rolling_points_home/away, rolling_win_rate_home/away


# Betting Odds & Team Encoding
- `teamstats_odds.csv`: odds probabilities + team dummies
- `train_odds.csv`, `test_odds.csv`: train/test splits

