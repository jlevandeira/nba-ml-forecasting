import pandas as pd 
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the dir of the current script
csv_path = os.path.join(SCRIPT_DIR, "data", "raw", "nba_games_2018-2026.csv")
df = pd.read_csv(csv_path)

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(by="GAME_DATE").reset_index(drop=True)

# Stats from the last 10 games for each team, with a minimum of 3 games played
stats_cols = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "STL", "BLK", "TOV"]

team_stats = []

for team_id, team_df in df.groupby("TEAM_ID"):
    team_df = team_df.sort_values("GAME_DATE").copy()

    for col in stats_cols:
        team_df[f"{col}_ROLLING_10"] = team_df[col].shift(1).rolling(10, min_periods=3).mean()

    team_stats.append(team_df)

df = pd.concat(team_stats).sort_values("GAME_DATE").reset_index(drop=True)

home = df[df["MATCHUP"].str.contains("vs\\.")].copy()
away = df[df["MATCHUP"].str.contains("@")].copy()

def define_winner(result):
    if result == "W":
        return 1
    else:
        return 0

home["HOME_WIN"] = home["WL"].apply(define_winner)

games_merged = pd.merge(home, away, on="GAME_ID", suffixes=("_HOME", "_AWAY"))

games_final = games_merged.dropna(subset=["PTS_ROLLING_10_HOME", "PTS_ROLLING_10_AWAY"])

output_dir = os.path.join(SCRIPT_DIR, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

final_path = os.path.join(output_dir, "nba_games_processed.csv")
games_final.to_csv(final_path, index=False)

print(f"Total of games merged: {len(games_merged)}")
print(games_merged[["GAME_DATE_HOME", "MATCHUP_HOME", "PTS_HOME", "PTS_AWAY", "HOME_WIN"]].head())