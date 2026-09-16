import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder #Module that finds games based on certain criteria

SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26"
]

def fetch_season(season_name):
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_name,
                                               season_type_nullable="Regular Season",
                                               league_id_nullable="00")
    df=finder.get_data_frames()[0]
    return df

def main():
    os.makedirs("data/raw", exist_ok=True)

    all_seasons_data = []

    for s in SEASONS:
        df_season = fetch_season(s)
        all_seasons_data.append(df_season)
        time.sleep(1.5)  # Sleep for 1.5 second to avoid getting temporary ban from API

    final_df = pd.concat(all_seasons_data, ignore_index=True)

    path = "data/raw/nba_games_2018-2026.csv"
    final_df.to_csv(path, index=False)
    print(f"Data saved to {path}, with {len(final_df)} rows.")

if __name__ == "__main__":
    main()