import os
import joblib
import pandas as pd
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Load the model and scaler
models_path = os.path.join(SCRIPT_PATH, "models")
model = joblib.load(os.path.join(models_path, "nba_model.joblib"))
scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))

# Load the data
csv_path = os.path.join(SCRIPT_PATH, "data", "processed", "nba_games_processed.csv")
df = pd.read_csv(csv_path)
df["GAME_DATE_HOME"] = pd.to_datetime(df["GAME_DATE_HOME"])
df = df.sort_values("GAME_DATE_HOME")

stats = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "AST", "REB", "STL", "BLK", "TOV"]

team_profile = {}
all_teams = df["TEAM_ABBREVIATION_HOME"].unique()

for team in all_teams:
    team_games = df[df["TEAM_ABBREVIATION_HOME"] == team] # Filtering the games for a specific team 
    last_game = team_games.iloc[-1] # Getting the last game played by the team

# Getting the values for the last game played by a team and storing them in a list
    team_values = {}
    for r in stats:
        row = f"{r}_ROLLING_10_HOME"
        team_values[r] = last_game[row]
        

    team_profile[team] = team_values # Storing a list associated with a team

# Function to calculate the probability of a home team winning against an away team
def game_probability(home_team, away_team):
    features = []

    # Individual stats, home, away and difference
    for s in stats:
        home_value = team_profile[home_team][s]
        away_value = team_profile[away_team][s]
        diff_value = home_value - away_value

        features.append(home_value)
        features.append(away_value)
        features.append(diff_value)

    # Converts to a numpy matrix, not as heavy as pandas
    features_array = np.array([features])

    # Normalize and calculate the probability
    features_scaled = scaler.transform(features_array)
    probability = model.predict_proba(features_scaled)
    probability_home = probability[0][1]

    return probability_home

def simulate_game(home_team, away_team):
    prob = game_probability(home_team, away_team)
    rand_num = random.random() # Generating a random number 0-1

    # if the random number is less than the probability then home team wins
    if rand_num < prob:
        return home_team
    else:
        return away_team

# All the teams from each conference
east_teams = ["BOS", "MIL", "CLE", "ORL", "IND", "PHI", "MIA", "CHI", "ATL", "BKN", "TOR", "CHA", "WAS", "DET", "NYK"]
west_teams = ["OKC", "DEN", "MIN", "LAC", "DAL", "PHX", "NOP", "LAL", "SAC", "GSW", "HOU", "UTA", "POR", "SAS", "MEM"]

# Function that determines the top 8 teams from each conference
def top_conference(teams_list):
    # Make a diccionary to count the amount of wins of each team
    wins_board = {}
    for t in teams_list:
        wins_board[t] = 0

    for teamA in teams_list:
        for teamB in teams_list:
            if teamA != teamB:
                winner = simulate_game(teamA, teamB)

                wins_board[winner] += 1

    # Array of pairs [wins, team] to order by wins
    pairs = []

    for t in wins_board:
        total = wins_board[t]
        pairs.append([total, t])

    pairs.sort(reverse=True) # Orders from highest to lowest winning team

    top_8 = []
    for i in range(8):
        team_name = pairs[i][1]
        top_8.append(team_name)

    return top_8, wins_board

# Simulating a series of playoffs
def simulate_series(team1, team2):
    team_wins1 = 0
    team_wins2 = 0

    # Order of where the games are being played during 1 series
    being_played_at = [team1, team1, team2, team2, team1, team2, team1]

    for i in being_played_at:
        if i == team1:
            game_winner = simulate_game(team1, team2)
        else:
            game_winner = simulate_game(team2, team1)

        if game_winner == team1:
            team_wins1 += 1
        else:
            team_wins2 += 1

        if team_wins1 == 4:
            return team1
        elif team_wins2 == 4:
            return team2 

# Simulate one side of the playoffs bracket (conference)
def simulate_conference(pos_team):
    """
    First Round:
        1st vs 8th, 2nd vs 7th, 3rd vs 6th, 4th vs 5th 
    """
    round1_m1 = simulate_series(pos_team[0], pos_team[7])
    round1_m2 = simulate_series(pos_team[1], pos_team[6])
    round1_m3 = simulate_series(pos_team[2], pos_team[5])
    round1_m4 = simulate_series(pos_team[3], pos_team[4])

    semi1 = simulate_series(round1_m1, round1_m4)
    semi2 = simulate_series(round1_m2, round1_m3)

    conference_champ = simulate_series(semi1, semi2)
    return conference_champ

# Funcrtion to simulate the whole season, regular season and playoffs
def simulate_full_season():
    # Model calculates the top 8 of each conference
    playoffs_east, wins_east = top_conference(east_teams)
    playoffs_west, wins_west = top_conference(west_teams)

    # Simulates each conference, indicating each conferences champ
    east_champ = simulate_conference(playoffs_east)
    west_champ = simulate_conference(playoffs_west)

    # Deciding who plays at home first, more wins during regular season = Game 1,3,5,7 at home
    east_champ_w = wins_east[east_champ]
    west_champ_w = wins_west[west_champ]

    if west_champ_w >= east_champ_w:
        champ = simulate_series(west_champ, east_champ)
    else:
        champ = simulate_series(east_champ, west_champ)

    return champ

# MCTS cicle at least 10000 simulations
n = 1000
print("\nStarting the simulation with Monte Carlo during Regular season and Playoffs...")
print(f"Simulating {n} seasons...")

champions_count = {}

for i in range(n):
    # Prints an update for each 200 season that gets simulated, to see the progress
    if(i+1) % 200 == 0: 
        print(f"Seasons simulated: {i+1}/{n}")

    champion = simulate_full_season()
    if champion in champions_count:
        champions_count[champion] += 1
    else:
        champions_count[champion] = 1

print("\n-----------------------------------------------")
print(" Final Result: NEXT NBA CHAMPION PROBABILITY")
print("-----------------------------------------------")

# Order the results in pairs[titles, team]
final_rank = []
for team in champions_count:
    titles = champions_count[team]
    final_rank.append([titles, team])

final_rank.sort(reverse=True)

for i in final_rank:
    titles = i[0]
    team = i[1]
    per = (titles/n)*100
    print(f"{team}: {per:.1f}% probability({titles} titles in {n} simulations)")
