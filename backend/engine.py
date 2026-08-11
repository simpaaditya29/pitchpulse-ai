import pandas as pd
import glob
import os
import csv
from vault import vault_1983_final

def load_and_clean_data(folder_path):
    list_df = []
    
    # Grab only ball-by-ball files, completely ignoring the _info metadata files
    all_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if not f.endswith("_info.csv")]
    
    for file in all_files:
        df = pd.read_csv(file)
        
        # Standardize column names across formats
        df.rename(columns={'runs_off_bat': 'runs_batter', 'extras': 'runs_extras'}, inplace=True)
        
        # Extract the file name (without .csv) to use as match_id
        match_id = os.path.splitext(os.path.basename(file))[0]
        df['match_id'] = match_id
        list_df.append(df)

    combined_df = pd.concat(list_df, ignore_index=True)

    # --- DYNAMIC PHASE OF PLAY FEATURE ENGINEERING ---
    def get_phase(over, path):
        over_num = int(float(over))
        path_lower = path.lower()
        
        # ODI Phase Logic (50 Overs)
        if 'odi' in path_lower:
            if over_num < 10:
                return 'Powerplay (1-10)'
            elif over_num < 40:
                return 'Middle Overs (11-40)'
            else:
                return 'Death Overs (41-50)'
                
        # Test Match Ball Condition Logic
        elif 'test' in path_lower:
            if over_num < 30:
                return 'New Ball (Overs 1-30)'
            elif over_num < 80:
                return 'Old Ball (Overs 31-80)'
            else:
                return 'Second New Ball (80+)'
                
        # T20 / IPL Phase Logic (20 Overs)
        else:
            if over_num < 6:
                return 'Powerplay (1-6)'
            elif over_num < 16:
                return 'Middle Overs (7-15)'
            else:
                return 'Death Overs (16-20)'

    combined_df['phase_of_play'] = combined_df['ball'].apply(lambda x: get_phase(x, folder_path))
    return combined_df

def filter_by_year(df):
    print("\nDo you want to filter this data by a specific year? (e.g., 2023)")
    print("Press Enter to skip and use all-time data.")
    
    year_input = input("Enter Year: ").strip()
    
    if year_input == "":
        return df  # Return the full dataset if they press enter
        
    if 'start_date' in df.columns:
        # Filter the dataframe to only include rows where the date starts with the chosen year
        filtered_df = df[df['start_date'].astype(str).str.startswith(year_input)].copy()
        
        if filtered_df.empty:
            print(f" No matches found for the year {year_input}. Reverting to all-time data.")
            return df
            
        print(f" Successfully filtered data! Now analyzing {len(filtered_df)} deliveries from {year_input}.")
        return filtered_df
    else:
        print(" Error: 'start_date' column missing. Using all-time data.")
        return df

    
from difflib import get_close_matches

from difflib import get_close_matches

def resolve_player_name(input_name, dataset):
    """
    Finds the official Cricsheet player name from flexible user input.
    """
    if not input_name:
        return None
        
    all_players = set(dataset['striker'].unique()).union(set(dataset['bowler'].unique()))
    clean_input = input_name.strip().lower()
    
    # 1. Exact case-insensitive match
    for player in all_players:
        if str(player).lower() == clean_input:
            return player
            
    # 2. Full Substring match (e.g., "dhoni" in "MS Dhoni")
    substring_matches = [p for p in all_players if clean_input in str(p).lower()]
    if len(substring_matches) == 1:
        return substring_matches[0]
        
    # 3. Last Name & Initial Match
    parts = clean_input.split()
    if len(parts) > 1:
        last_word = parts[-1] 
        last_word_matches = [p for p in all_players if last_word in str(p).lower()]
        
        if len(last_word_matches) == 1:
            return last_word_matches[0]
        elif len(last_word_matches) > 1:
            first_initial = parts[0][0]
            initial_matches = [p for p in last_word_matches if str(p).lower().startswith(first_initial)]
            
            if len(initial_matches) == 1:
                return initial_matches[0]
            elif len(initial_matches) > 1:
                # NEW: Don't panic on a tie! Just grab the first matching player.
                return initial_matches[0]
                
    # 4. Fuzzy match for typos (Increased cutoff to 0.6 so it stops making wild guesses!)
    fuzzy_matches = get_close_matches(input_name, all_players, n=1, cutoff=0.6)
    if fuzzy_matches:
        return fuzzy_matches[0]
        
    return None


def analyze_matchup(df, batter_name, bowler_name):
    print(f"\n--- MATCHUP SCOUT: {batter_name} vs {bowler_name} ---")
    
    matchup_data = df[(df['striker'] == batter_name) & (df['bowler'] == bowler_name)].copy()
    
    if matchup_data.empty:
        print("No historical data found for this matchup.")
        return
        
    balls_faced = len(matchup_data)
    matchup_data['runs_batter'] = matchup_data['runs_batter'].astype(int)
    
    runs_scored = matchup_data['runs_batter'].sum()
    fours = len(matchup_data[matchup_data['runs_batter'] == 4])
    sixes = len(matchup_data[matchup_data['runs_batter'] == 6])
    strike_rate = (runs_scored / balls_faced) * 100
    dismissals = len(matchup_data[matchup_data['player_dismissed'] == batter_name])
    
    print(f"Overall Balls: {balls_faced}")
    print(f"Overall Runs:  {runs_scored} (4s: {fours}, 6s: {sixes})")
    print(f"Strike Rate:   {strike_rate:.2f}")
    print(f"Dismissals:    {dismissals}")
    
    # --- NEW: PHASE OF PLAY BREAKDOWN ---
    print("\n--- TACTICAL PHASE BREAKDOWN ---")
    phases = ['Powerplay', 'Middle Overs', 'Death Overs']
    
    for phase in phases:
        # Filter the matchup data down to just this specific phase
        phase_data = matchup_data[matchup_data['phase_of_play'] == phase]
        
        if not phase_data.empty:
            p_balls = len(phase_data)
            p_runs = phase_data['runs_batter'].sum()
            p_sr = (p_runs / p_balls) * 100
            p_outs = len(phase_data[phase_data['player_dismissed'] == batter_name])
            
            # Formatting it nicely so it reads like a clean report
            print(f"{phase.ljust(15)} | Runs: {str(p_runs).ljust(3)} | Balls: {str(p_balls).ljust(3)} | SR: {p_sr:>6.2f} | Outs: {p_outs}")


def analyze_bowler(df, bowler_name):
    print(f"\n--- CAPTAIN'S BOWLER SCOUT: {bowler_name} ---")
    
    # Isolate all balls bowled by this specific player
    bowler_data = df[df['bowler'] == bowler_name].copy()
    
    if bowler_data.empty:
        print("No historical data found for this bowler.")
        return
        
    # Convert runs and extras to integers for math
    bowler_data['runs_batter'] = bowler_data['runs_batter'].astype(int)
    bowler_data['runs_extras'] = bowler_data['runs_extras'].astype(int)
    bowler_data['total_runs_conceded'] = bowler_data['runs_batter'] + bowler_data['runs_extras']
    
    print("\n--- PHASE-BY-PHASE RELIABILITY ---")
    print(f"{'PHASE'.ljust(15)} | {'OVERS'.ljust(6)} | {'RUNS'.ljust(5)} | {'WKTS'.ljust(4)} | {'ECON'.ljust(5)}")
    print("-" * 48)
    
    phases = ['Powerplay', 'Middle Overs', 'Death Overs']
    
    for phase in phases:
        phase_data = bowler_data[bowler_data['phase_of_play'] == phase]
        
        if not phase_data.empty:
            balls = len(phase_data)
            overs = balls / 6  # Rough estimate of overs bowled
            runs = phase_data['total_runs_conceded'].sum()
            
            # Count wickets (if the player_dismissed column is not empty, it's a wicket)
            wickets = len(phase_data[phase_data['player_dismissed'] != ''])
            
            # Calculate Economy Rate (Runs per Over)
            economy = runs / overs if overs > 0 else 0
            
            print(f"{phase.ljust(15)} | {overs:>6.1f} | {runs:>5} | {wickets:>4} | {economy:>5.2f}")


def analyze_batter(df, batter_name):
    print(f"\n--- CAPTAIN'S BATTER SCOUT: {batter_name} ---")
    
    # Isolate all balls faced by this specific player
    batter_data = df[df['striker'] == batter_name].copy()
    
    if batter_data.empty:
        print("No historical data found for this batter.")
        return
        
    batter_data['runs_batter'] = batter_data['runs_batter'].astype(int)
    
    print("\n--- PHASE-BY-PHASE PERFORMANCE ---")
    print(f"{'PHASE'.ljust(15)} | {'RUNS'.ljust(5)} | {'BALLS'.ljust(5)} | {'SR'.ljust(6)} | {'4s/6s'}")
    print("-" * 50)
    
    phases = ['Powerplay', 'Middle Overs', 'Death Overs']
    
    for phase in phases:
        phase_data = batter_data[batter_data['phase_of_play'] == phase]
        
        if not phase_data.empty:
            balls = len(phase_data)
            runs = phase_data['runs_batter'].sum()
            
            fours = len(phase_data[phase_data['runs_batter'] == 4])
            sixes = len(phase_data[phase_data['runs_batter'] == 6])
            boundaries = f"{fours}/{sixes}"
            
            strike_rate = (runs / balls) * 100 if balls > 0 else 0
            
            print(f"{phase.ljust(15)} | {runs:>5} | {balls:>5} | {strike_rate:>6.2f} | {boundaries:>5}")

def analyze_workload(df, bowler_name):
    print(f"\n--- FAST BOWLER WORKLOAD TRACKER: {bowler_name} ---")
    
    bowler_data = df[df['bowler'] == bowler_name].copy()
    
    if bowler_data.empty:
        print("No historical data found for this bowler.")
        return
        
    if 'match_id' not in bowler_data.columns:
        print(" Error: 'match_id' column is missing from the dataset. Cannot track workload.")
        return
        
    # Convert runs to integers
    bowler_data['runs_conceded'] = bowler_data['runs_batter'].astype(int) + bowler_data['runs_extras'].astype(int)
    
    # Calculate which ball of the match this is for the bowler
    # By grouping by match_id, the counter resets to 1 for every new game
    bowler_data['delivery_count'] = bowler_data.groupby('match_id').cumcount() + 1
    
    # Split the data: Fresh (Balls 1-12) vs Fatigued (Balls 13+)
def analyze_workload(df, bowler_name):
    print(f"\n--- FAST BOWLER WORKLOAD TRACKER: {bowler_name} ---")
    
    bowler_data = df[df['bowler'] == bowler_name].copy()
    
    if bowler_data.empty:
        print("No historical data found for this bowler.")
        return
        
    if 'match_id' not in bowler_data.columns:
        print(" Error: 'match_id' column is missing from the dataset. Cannot track workload.")
        return
        
    bowler_data['runs_conceded'] = bowler_data['runs_batter'].astype(int) + bowler_data['runs_extras'].astype(int)
    
    bowler_data['delivery_count'] = bowler_data.groupby('match_id').cumcount() + 1
    
    fresh_data = bowler_data[bowler_data['delivery_count'] <= 12]
    fatigue_data = bowler_data[bowler_data['delivery_count'] > 12]
    
    def print_stamina_stats(label, data):
        if data.empty:
            return
        balls = len(data)
        overs = balls / 6
        runs = data['runs_conceded'].sum()
        wickets = len(data[data['player_dismissed'] != ''])
        econ = runs / overs if overs > 0 else 0
        print(f"{label.ljust(20)} | Overs: {overs:>6.1f} | Wickets: {wickets:>4} | Econ: {econ:>5.2f}")

    print("\n--- STAMINA & FATIGUE BREAKDOWN ---")
    print("Comparing performance in the first 2 overs vs subsequent overs:")
    print("-" * 65)
    print_stamina_stats("Fresh (Overs 1-2)", fresh_data)
    print_stamina_stats("Fatigued (Overs 3-4+)", fatigue_data)
    print("-" * 65)


def analyze_team(df, team_name):
    print(f"\n--- FRANCHISE ANALYTICS HUB: {team_name.upper()} ---")
    
    batting_data = df[df['batting_team'].str.contains(team_name, case=False, na=False)].copy()
    bowling_data = df[df['bowling_team'].str.contains(team_name, case=False, na=False)].copy()
    
    if batting_data.empty and bowling_data.empty:
        print("No historical data found. Ensure you type the full name (e.g., 'Chennai Super Kings').")
        return
        
    batting_data['runs_batter'] = batting_data['runs_batter'].astype(int)
    
    print("\n[ ALL-TIME TOP 3 RUN SCORERS ]")
    top_batters = batting_data.groupby('striker')['runs_batter'].sum().sort_values(ascending=False).head(3)
    
    for batter, runs in top_batters.items():
        print(f"- {batter.ljust(20)} : {runs} runs")
        
    print("\n[ ALL-TIME TOP 3 WICKET TAKERS ]")
    wickets_data = bowling_data[bowling_data['player_dismissed'].notna() & (bowling_data['player_dismissed'] != '')]
    
    top_bowlers = wickets_data.groupby('bowler').size().sort_values(ascending=False).head(3)
    
    for bowler, wkts in top_bowlers.items():
        print(f"- {bowler.ljust(20)} : {wkts} wickets")


def export_ml_features(df):
    print("\n--- INITIATING AI/ML FEATURE EXTRACTION ---")
    print("Aggregating historical matchup data for predictive modeling...")
    
    # Ensure runs are treated as integers
    df['runs_batter'] = df['runs_batter'].astype(int)
    
    # Create a simple 1 or 0 flag for whether a wicket fell on that delivery
    df['is_dismissed'] = df['player_dismissed'].notna() & (df['player_dismissed'] != '')
    df['is_dismissed'] = df['is_dismissed'].astype(int)
    
    # Group by every unique Batter vs Bowler matchup
    ml_data = df.groupby(['striker', 'bowler']).agg(
        balls_faced=('ball', 'count'),
        runs_scored=('runs_batter', 'sum'),
        dismissals=('is_dismissed', 'sum')
    ).reset_index()
    
    # Noise Reduction: Only keep matchups where they have faced at least 10 balls
    ml_data = ml_data[ml_data['balls_faced'] >= 10].copy()
    
    # Calculate derived features for the ML model to learn from
    ml_data['strike_rate'] = (ml_data['runs_scored'] / ml_data['balls_faced']) * 100
    
    # Round the strike rate for clean data formatting
    ml_data['strike_rate'] = ml_data['strike_rate'].round(2)
    
    # Export to CSV
    output_filename = "ml_matchup_features.csv"
    ml_data.to_csv(output_filename, index=False)
    
    print(f" Success! Extracted {len(ml_data)} robust head-to-head matchups.")
    print(f" Dataset exported to your workspace as '{output_filename}'.")        
    

if __name__ == "__main__":
    print("\n" + "="*40)
    print("      CRICKET MATCHUP ENGINE          ")
    print("="*40)
    print(" * Database Scope: 2004 - Present") # <--- THE FIX
    print(" * Note: Pre-2004 historical metrics are limited.")

    # --- DYNAMIC MAIN MENU ---
    print("\nSelect Dataset to Load:")
    print("1: Indian Premier League (IPL)")
    print("2: T20 Internationals (T20i)")
    print("3: One Day Internationals (ODI)")
    print("4: Test Matches")
    
    dataset_choice = input("\nEnter 1, 2, 3, or 4: ").strip()
    
    if dataset_choice == '2':
        data_folder = "data/t20i"
        print(f"\nTargeting folder: {data_folder}. Initializing T20i data...")
    elif dataset_choice == '3':
        data_folder = "data/odi"
        print(f"\nTargeting folder: {data_folder}. Initializing ODI data...")
    elif dataset_choice == '4':
        data_folder = "data/test"
        print(f"\nTargeting folder: {data_folder}. Initializing Test data...")
    else:
        # Defaults to IPL if 1 or an unrecognized key is entered
        data_folder = "data/ipl"
        print(f"\nTargeting folder: {data_folder}. Initializing IPL data...")
        
    # Load the chosen dataset
    match_data = load_and_clean_data(data_folder)
    
    # Filter by Year
    match_data = filter_by_year(match_data)

    # --- THE INTERACTIVE SCOUT LOOP ---
    while True:
        print("\n" + "-"*75)
        print("Options: [1] Matchup  [2] Bowler  [3] Batter  [4] Vault  [5] Workload  [6] Team Hub  [7] ML Export  [exit] Quit")
        mode = input("Select Mode: ").strip().lower()
        
        if mode == 'exit':
            print("\nShutting down Matchup Engine. Great session!")
            break
            
        # --- OPTION 4: CLASSIC MATCHES VAULT ---
        if mode == '4':
            print("\n--- CLASSIC MATCHES VAULT ---")
            print("[1] 1983 World Cup Final (IND vs WI)")
            vault_choice = input("Select Match: ").strip()
            
            if vault_choice == '1':
                vault_1983_final()
            else:
                print("Invalid selection or match not yet added to Vault.")
            continue

        # --- OPTION 2: BOWLER SCOUT ---
        if mode == '2':
            raw_bowler = input("\nEnter Bowler Name: ").strip()
            bowler = resolve_player_name(raw_bowler, match_data)
            if bowler:
                analyze_bowler(match_data, bowler)
            else:
                print(f"Could not find any player matching '{raw_bowler}'. Please try again.")
            continue
            
        # --- OPTION 3: BATTER SCOUT ---
        if mode == '3':
            raw_batter = input("\nEnter Batter Name: ").strip()
            batter = resolve_player_name(raw_batter, match_data)
            if batter:
                analyze_batter(match_data, batter)
            else:
                print(f"Could not find any player matching '{raw_batter}'. Please try again.")
            continue

        # --- OPTION 5: WORKLOAD TRACKER ---
        if mode == '5':
            raw_bowler = input("\nEnter Bowler Name: ").strip()
            bowler = resolve_player_name(raw_bowler, match_data)
            if bowler:
                analyze_workload(match_data, bowler)
            else:
                print(f"Could not find any player matching '{raw_bowler}'. Please try again.")
            continue

        # --- OPTION 6: TEAM HUB ---
        if mode == '6':
            team_input = input("\nEnter Full Team Name (e.g., Chennai Super Kings): ").strip()
            analyze_team(match_data, team_input)
            continue
        
        # --- OPTION 7: ML EXPORT PIPELINE ---
        if mode == '7':
            export_ml_features(match_data)
            continue

        # --- OPTION 1: MATCHUP SCOUT (DEFAULT) ---
        print("\n--- MATCHUP SCOUT ---")
        raw_batter = input("Enter Batter Name: ").strip()
        if raw_batter.lower() == 'exit':
            break
            
        raw_bowler = input("Enter Bowler Name: ").strip()
        if raw_bowler.lower() == 'exit':
            break
            
        batter = resolve_player_name(raw_batter, match_data)
        bowler = resolve_player_name(raw_bowler, match_data)
        
        if not batter or not bowler:
            print("Player not found. Please try again.")
            continue
            
        if batter != raw_batter or bowler != raw_bowler:
            print(f"-> Resolved to: {batter} vs {bowler}")
            
        analyze_matchup(match_data, batter, bowler)