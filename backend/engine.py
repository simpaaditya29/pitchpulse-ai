import os
import glob
import pandas as pd
import csv
from vault import vault_1983_final

def get_phase(over, path):
    """Assigns phase of play based on match format and over number."""
    path_lower = str(path).lower()
    
    if 'ipl' in path_lower or 't20' in path_lower:
        if over <= 6:
            return '1. Powerplay (1-6)'
        elif over <= 15:
            return '2. Middle (7-15)'
        else:
            return '3. Death (16-20)'
    elif 'odi' in path_lower:
        if over <= 10:
            return '1. Powerplay (1-10)'
        elif over <= 40:
            return '2. Middle (11-40)'
        else:
            return '3. Death (41-50)'
    else:
        return 'Standard'

def load_and_clean_data(folder_path):
    list_df = []

    # Grab only ball-by-ball match CSVs, ignoring metadata _info files
    all_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if not f.endswith("_info.csv")]
    
    total_files = len(all_files)
    if total_files > 0:
        print(f"-> Processing {total_files} match files. Please wait...")

    # Using enumerate gives us the current file number (i) alongside the filename
    for i, file in enumerate(all_files, 1):
        # Prints progress every 500 files or when it hits the very last file
        if i % 500 == 0 or i == total_files:
            print(f"   Loaded {i}/{total_files} files...")
            
        try:
            # low_memory=False prevents dtype warnings on large datasets
            df = pd.read_csv(file, low_memory=False)

            # Standardize column headers across various Cricsheet schemas
            df.rename(columns={'runs_off_bat': 'runs_batter', 'extras': 'runs_extras', 'striker': 'batter'}, inplace=True)

            # --- CRICSHEET EXTRAS & WICKET TRANSLATOR ---
            if 'extras_type' not in df.columns:
                df['extras_type'] = None
                if 'wides' in df.columns:
                    df.loc[df['wides'].notna(), 'extras_type'] = 'wides'
                if 'noballs' in df.columns:
                    df.loc[df['noballs'].notna(), 'extras_type'] = 'noballs'

            # Preemptively catch missing wicket columns
            if 'player_dismissed' not in df.columns:
                df['player_dismissed'] = None
            if 'wicket_type' not in df.columns:
                df['wicket_type'] = None
            # -------------------------------------------

            # Extract match ID from filename
            match_id = os.path.splitext(os.path.basename(file))[0]
            df['match_id'] = match_id
            list_df.append(df)

        except pd.errors.ParserError:
            print(f"-> Warning: Skipping irregularly formatted file [{os.path.basename(file)}]")
            continue

    # --- YOU WERE MISSING THIS BLOCK ---
    if not list_df:
        print(f"Warning: No valid CSV files found in {folder_path}!")
        return pd.DataFrame()

    combined_df = pd.concat(list_df, ignore_index=True)
    # -----------------------------------

    # --- RESTORE PHASE OF PLAY LOGIC ---
    if 'ball' in combined_df.columns:
        # Convert ball (e.g., 0.1) to over number (e.g., 1)
        combined_df['over'] = combined_df['ball'].astype(float).apply(lambda x: int(x) + 1)
        # Apply your custom format-aware get_phase function
        combined_df['phase_of_play'] = combined_df['over'].apply(lambda x: get_phase(x, folder_path))
        
    return combined_df


def filter_by_year(df):
    if df.empty:
        return df

    print("\nDo you want to filter this data by a specific year? (e.g., 2023)")
    print("Press Enter to skip and use all-time data.")
    year_input = input("Enter Year: ").strip()
    
    if year_input == "":
        return df
    
    if 'start_date' in df.columns:
        filtered_df = df[df['start_date'].astype(str).str.startswith(year_input)].copy()
        if filtered_df.empty:
            print(f"No matches found for the year '{year_input}'. Reverting to all-time data.")
            return df
        print(f"Data filtered successfully for year: {year_input}")
        return filtered_df
    else:
        print("Note: 'start_date' column not found in dataset. Using all available records.")
        return df


def resolve_player_name(query, df):
    if not query or df.empty:
        return None
        
    query_clean = query.strip().lower()
    query_parts = query_clean.split()
    
    # Safely extract all unique players from BOTH the batter and bowler columns
    batters = set(df['batter'].dropna().unique()) if 'batter' in df.columns else set()
    bowlers = set(df['bowler'].dropna().unique()) if 'bowler' in df.columns else set()
    all_players = list(batters.union(bowlers))
    
    # 1. Exact match
    for p in all_players:
        if p.lower() == query_clean:
            return p
            
    # 2. Simple Substring match (e.g., typing "Bumrah" matches "JJ Bumrah")
    matches = [p for p in all_players if query_clean in p.lower()]
    if len(matches) == 1:
        return matches[0]
        
    # 3. Smart Last Name Match (Handles "Jasprit Bumrah" -> "JJ Bumrah")
    last_name_query = query_parts[-1]
    last_name_matches = [p for p in all_players if p.lower().endswith(last_name_query)]
    
    if len(last_name_matches) == 1:
        return last_name_matches[0]
    elif len(last_name_matches) > 1:
        first_initial = query_parts[0][0]
        for p in last_name_matches:
            if p.lower().startswith(first_initial):
                return p
        return last_name_matches[0]
        
    return None


def analyze_matchup(df, batter, bowler):
    matchup_df = df[(df['batter'] == batter) & (df['bowler'] == bowler)].copy()
    
    if matchup_df.empty:
        print(f"\nNo head-to-head records found between {batter} and {bowler} in this dataset.")
        return
        
    runs = int(matchup_df['runs_batter'].sum())
    balls = len(matchup_df[matchup_df['extras_type'] != 'wides'])
    fours = len(matchup_df[matchup_df['runs_batter'] == 4])
    sixes = len(matchup_df[matchup_df['runs_batter'] == 6])
    dots = len(matchup_df[(matchup_df['runs_batter'] == 0) & (matchup_df['extras_type'].isna())])
    dismissals = len(matchup_df[matchup_df['player_dismissed'] == batter])
    
    sr = round((runs / balls * 100), 2) if balls > 0 else 0.0
    dot_pct = round((dots / balls * 100), 2) if balls > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"       MATCHUP SCOUT: {batter} vs {bowler}")
    print("="*50)
    print(f"Balls Faced    : {balls}")
    print(f"Runs Scored    : {runs}")
    print(f"Strike Rate    : {sr}")
    print(f"Dismissals     : {dismissals}")
    print(f"Dot Ball %     : {dot_pct}%")
    print(f"Boundaries     : {fours} Fours | {sixes} Sixes")
    print("-" * 50)


def analyze_bowler(df, bowler):
    b_df = df[df['bowler'] == bowler].copy()
    if b_df.empty:
        print(f"\nNo records found for bowler: {bowler}")
        return
        
    legal_deliveries = b_df[~b_df['extras_type'].isin(['wides', 'noballs'])]
    legal_balls = len(legal_deliveries)
    overs = f"{legal_balls // 6}.{legal_balls % 6}"
    
    runs_conceded = int(b_df['runs_batter'].sum() + b_df[b_df['extras_type'].isin(['wides', 'noballs'])]['runs_extras'].sum())
    wickets = len(b_df[b_df['wicket_type'].notna() & ~b_df['wicket_type'].isin(['run out', 'retired hurt'])])
    dots = len(b_df[(b_df['runs_batter'] == 0) & (b_df['runs_extras'] == 0)])
    
    econ = round((runs_conceded / (legal_balls / 6)), 2) if legal_balls > 0 else 0.0
    avg = round((runs_conceded / wickets), 2) if wickets > 0 else "N/A"
    sr = round((legal_balls / wickets), 2) if wickets > 0 else "N/A"
    dot_pct = round((dots / len(b_df) * 100), 2) if len(b_df) > 0 else 0.0

    print("\n" + "="*50)
    print(f"       BOWLER PROFILE: {bowler}")
    print("="*50)
    print(f"Overs Bowled   : {overs}")
    print(f"Wickets        : {wickets}")
    print(f"Runs Conceded  : {runs_conceded}")
    print(f"Economy Rate   : {econ}")
    print(f"Bowling Average: {avg}")
    print(f"Bowling SR     : {sr}")
    print(f"Dot Ball %     : {dot_pct}%")
    print("-" * 50)


def analyze_batter(df, batter):
    b_df = df[df['batter'] == batter].copy()
    if b_df.empty:
        print(f"\nNo records found for batter: {batter}")
        return
        
    runs = int(b_df['runs_batter'].sum())
    balls = len(b_df[b_df['extras_type'] != 'wides'])
    dismissals = len(b_df[b_df['player_dismissed'] == batter])
    fours = len(b_df[b_df['runs_batter'] == 4])
    sixes = len(b_df[b_df['runs_batter'] == 6])
    dots = len(b_df[(b_df['runs_batter'] == 0) & (b_df['extras_type'].isna())])
    
    sr = round((runs / balls * 100), 2) if balls > 0 else 0.0
    avg = round((runs / dismissals), 2) if dismissals > 0 else "N/A"
    dot_pct = round((dots / balls * 100), 2) if balls > 0 else 0.0

    print("\n" + "="*50)
    print(f"       CAPTAIN'S BATTER SCOUT: {batter}")
    print("="*50)
    print(f"Runs Scored    : {runs}")
    print(f"Balls Faced    : {balls}")
    print(f"Strike Rate    : {sr}")
    print(f"Batting Average: {avg}")
    print(f"Dot Ball %     : {dot_pct}%")
    print(f"Boundaries     : {fours} Fours | {sixes} Sixes")
    print("-" * 50)
    
    print("\n--- PHASE-BY-PHASE PERFORMANCE ---")
    print(f"{'PHASE':<22} | {'RUNS':<6} | {'BALLS':<6} | {'SR':<7} | {'4s/6s':<8}")
    print("-" * 55)
    
    for phase in b_df['phase_of_play'].unique():
        p_df = b_df[b_df['phase_of_play'] == phase]
        p_runs = int(p_df['runs_batter'].sum())
        p_balls = len(p_df[p_df['extras_type'] != 'wides'])
        p_sr = round((p_runs / p_balls * 100), 2) if p_balls > 0 else 0.0
        p_4s = len(p_df[p_df['runs_batter'] == 4])
        p_6s = len(p_df[p_df['runs_batter'] == 6])
        print(f"{phase:<22} | {p_runs:<6} | {p_balls:<6} | {p_sr:<7} | {p_4s}/{p_6s}")
    print("-" * 55)


def analyze_workload(df, bowler):
    b_df = df[df['bowler'] == bowler].copy()
    if b_df.empty:
        print(f"\nNo records found for bowler: {bowler}")
        return
        
    print("\n" + "="*50)
    print(f"       WORKLOAD & SPELL TRACKER: {bowler}")
    print("="*50)
    
    matches = b_df['match_id'].unique()
    total_balls = len(b_df[~b_df['extras_type'].isin(['wides', 'noballs'])])
    matches_played = len(matches)
    avg_overs_per_match = round((total_balls / 6) / matches_played, 1) if matches_played > 0 else 0.0
    
    print(f"Matches Tracked    : {matches_played}")
    print(f"Total Legal Balls  : {total_balls} ({total_balls // 6} overs)")
    print(f"Avg Overs / Match  : {avg_overs_per_match}")
    print("-" * 50)


def advanced_bowling_tactics(df, bowler):
    b_df = df[df['bowler'] == bowler].copy()
    if b_df.empty:
        print(f"\nNo records found for bowler: {bowler}")
        return
        
    print("\n" + "="*65)
    print(f"       ADVANCED TACTICAL SCOUT: {bowler}")
    print("="*65)
    print(f"{'PHASE OF PLAY':<22} | {'OVERS':<6} | {'WKTS':<5} | {'ECON':<5} | {'DOT %':<6} | {'SR (Balls/Wkt)'}")
    print("-" * 65)
    
    # Sort phases so Powerplay appears before Middle, and Middle before Death
    phases = b_df['phase_of_play'].unique()
    
    for phase in sorted(phases):
        p_df = b_df[b_df['phase_of_play'] == phase]
        
        legal_deliveries = p_df[~p_df['extras_type'].isin(['wides', 'noballs'])]
        legal_balls = len(legal_deliveries)
        if legal_balls == 0:
            continue
            
        overs = round(legal_balls / 6, 1)
        runs_conceded = int(p_df['runs_batter'].sum() + p_df[p_df['extras_type'].isin(['wides', 'noballs'])]['runs_extras'].sum())
        wickets = len(p_df[p_df['wicket_type'].notna() & ~p_df['wicket_type'].isin(['run out', 'retired hurt'])])
        dots = len(p_df[(p_df['runs_batter'] == 0) & (p_df['runs_extras'] == 0)])
        
        econ = round((runs_conceded / overs), 2) if overs > 0 else 0.0
        dot_pct = round((dots / len(p_df) * 100), 1)
        sr = round((legal_balls / wickets), 1) if wickets > 0 else "N/A"
        
        print(f"{phase:<22} | {overs:<6} | {wickets:<5} | {econ:<5} | {dot_pct:<5}% | {sr}")
        
    print("-" * 65)


def advanced_batting_tactics(df, batter):
    # Filter the dataset for when this specific player was batting
    b_df = df[df['batter'] == batter].copy()
    if b_df.empty:
        print(f"\nNo records found for batter: {batter}")
        return
        
    print("\n" + "="*75)
    print(f"       ADVANCED TACTICAL SCOUT: {batter}")
    print("="*75)
    print(f"{'PHASE OF PLAY':<22} | {'RUNS':<5} | {'BF':<4} | {'SR':<6} | {'4s':<3} | {'6s':<3} | {'DOT %'}")
    print("-" * 75)
    
    phases = b_df['phase_of_play'].unique()
    
    for phase in sorted(phases):
        p_df = b_df[b_df['phase_of_play'] == phase]
        
        # Wides do not count as balls faced for a batter
        legal_deliveries = p_df[p_df['extras_type'] != 'wides']
        bf = len(legal_deliveries)
        
        if bf == 0:
            continue
            
        # Calculate granular batting metrics
        runs = int(p_df['runs_batter'].sum())
        fours = len(p_df[p_df['runs_batter'] == 4])
        sixes = len(p_df[p_df['runs_batter'] == 6])
        dots = len(p_df[p_df['runs_batter'] == 0])
        
        # Avoid division by zero errors
        sr = round((runs / bf) * 100, 1) if bf > 0 else 0.0
        dot_pct = round((dots / bf * 100), 1) if bf > 0 else 0.0
        
        print(f"{phase:<22} | {runs:<5} | {bf:<4} | {sr:<6} | {fours:<3} | {sixes:<3} | {dot_pct:<5}%")
        
    print("-" * 75)



def analyze_team(df, team_name):
    query = team_name.strip().lower()
    teams = set(df['batting_team'].dropna().unique()) if 'batting_team' in df.columns else set()
    
    matched_team = None
    for t in teams:
        if query in t.lower():
            matched_team = t
            break
            
    if not matched_team:
        print(f"\nNo team matching '{team_name}' found in this dataset.")
        return
        
    t_df = df[df['batting_team'] == matched_team]
    total_runs = int(t_df['runs_batter'].sum() + t_df['runs_extras'].sum()) if 'runs_extras' in t_df.columns else int(t_df['runs_batter'].sum())
    matches = t_df['match_id'].nunique() if 'match_id' in t_df.columns else "N/A"
    
    print("\n" + "="*50)
    print(f"       TEAM HUB: {matched_team}")
    print("="*50)
    print(f"Matches Tracked : {matches}")
    print(f"Total Runs Agg  : {total_runs}")
    print("-" * 50)


def export_ml_features(df):
    print("\n" + "="*65)
    print("      AI FEATURE ENGINEERING: TACTICAL PIPELINE")
    print("="*65)
    print("-> Crunching raw ball-by-ball data into Machine Learning features...")
    
    if 'phase_of_play' not in df.columns:
        print("Error: Tactical phases not found. Please reload the dataset.")
        return

    # Create a copy to avoid warning messages
    ml_df = df.copy()
    
    # 1. Label every single ball for the AI
    ml_df['is_dot'] = ((ml_df['runs_batter'] == 0) & (ml_df['runs_extras'] == 0)).astype(int)
    ml_df['is_boundary'] = ml_df['runs_batter'].isin([4, 6]).astype(int)
    ml_df['is_wicket'] = ml_df['wicket_type'].notna().astype(int)
    
    print("-> Aggregating phase-by-phase tactical metrics...")
    
    # 2. Group the data so the AI can read it match-by-match and phase-by-phase
    phase_stats = ml_df.groupby(['match_id', 'batting_team', 'phase_of_play']).agg(
        total_runs=('runs_batter', 'sum'),
        total_extras=('runs_extras', 'sum'),
        dot_balls=('is_dot', 'sum'),
        boundaries=('is_boundary', 'sum'),
        wickets_lost=('is_wicket', 'sum'),
        balls_faced=('ball', 'count')
    ).reset_index()

    # Save to CSV
    output_name = "ml_tactical_features.csv"
    phase_stats.to_csv(output_name, index=False)
    
    print(f"-> SUCCESS: Extracted advanced features for Random Forest model!")
    print(f"-> Dataset saved directly to your workspace as '{output_name}'")
    print("-" * 65)


# --- COMPARATOR MODULE: BATTER VS BATTER ---
def compare_batters(df, batter1, batter2):
    def get_batter_stats(b_name):
        b_df = df[df['batter'] == b_name].copy()
        if b_df.empty:
            return None
        
        runs = int(b_df['runs_batter'].sum())
        balls = len(b_df[b_df['extras_type'] != 'wides'])
        dismissals = len(b_df[b_df['player_dismissed'] == b_name])
        fours = len(b_df[b_df['runs_batter'] == 4])
        sixes = len(b_df[b_df['runs_batter'] == 6])
        dots = len(b_df[(b_df['runs_batter'] == 0) & (b_df['extras_type'].isna())])
        
        sr = round((runs / balls * 100), 2) if balls > 0 else 0.0
        avg = round((runs / dismissals), 2) if dismissals > 0 else "N/A"
        dot_pct = round((dots / balls * 100), 2) if balls > 0 else 0.0
        boundary_pct = round(((fours * 4 + sixes * 6) / runs * 100), 2) if runs > 0 else 0.0
        
        return {
            'name': b_name, 'runs': runs, 'balls': balls, 'sr': sr, 
            'avg': avg, 'fours': fours, 'sixes': sixes, 
            'dot_pct': dot_pct, 'boundary_pct': boundary_pct
        }

    s1, s2 = get_batter_stats(batter1), get_batter_stats(batter2)
    if not s1 or not s2:
        print("\nCould not generate comparison. Ensure both player names exist in this dataset.")
        return

    s1_dot = f"{s1['dot_pct']}%"
    s2_dot = f"{s2['dot_pct']}%"
    s1_bnd = f"{s1['boundary_pct']}%"
    s2_bnd = f"{s2['boundary_pct']}%"
    s1_fours_sixes = f"{s1['fours']} / {s1['sixes']}"
    s2_fours_sixes = f"{s2['fours']} / {s2['sixes']}"

    print("\n" + "="*65)
    print(f"       HEAD-TO-HEAD BENCHMARK: {s1['name']} vs {s2['name']}")
    print("="*65)
    print(f"{'Metric':<25} | {s1['name']:<18} | {s2['name']:<18}")
    print("-" * 65)
    print(f"{'Total Runs':<25} | {s1['runs']:<18} | {s2['runs']:<18}")
    print(f"{'Balls Faced':<25} | {s1['balls']:<18} | {s2['balls']:<18}")
    print(f"{'Strike Rate':<25} | {s1['sr']:<18} | {s2['sr']:<18}")
    print(f"{'Batting Average':<25} | {str(s1['avg']):<18} | {str(s2['avg']):<18}")
    print(f"{'Dot Ball %':<25} | {s1_dot:<18} | {s2_dot:<18}")
    print(f"{'Boundary Run %':<25} | {s1_bnd:<18} | {s2_bnd:<18}")
    print(f"{'Boundaries (4s / 6s)':<25} | {s1_fours_sixes:<18} | {s2_fours_sixes:<18}")
    print("-" * 65)


# --- COMPARATOR MODULE: BOWLER VS BOWLER ---
def compare_bowlers(df, bowler1, bowler2):
    def get_bowler_stats(b_name):
        b_df = df[df['bowler'] == b_name].copy()
        if b_df.empty:
            return None
            
        legal_deliveries = b_df[~b_df['extras_type'].isin(['wides', 'noballs'])]
        legal_balls = len(legal_deliveries)
        overs = f"{legal_balls // 6}.{legal_balls % 6}"
        
        runs_conceded = int(b_df['runs_batter'].sum() + b_df[b_df['extras_type'].isin(['wides', 'noballs'])]['runs_extras'].sum())
        wickets = len(b_df[b_df['wicket_type'].notna() & ~b_df['wicket_type'].isin(['run out', 'retired hurt'])])
        dots = len(b_df[(b_df['runs_batter'] == 0) & (b_df['runs_extras'] == 0)])
        
        econ = round((runs_conceded / (legal_balls / 6)), 2) if legal_balls > 0 else 0.0
        avg = round((runs_conceded / wickets), 2) if wickets > 0 else "N/A"
        sr = round((legal_balls / wickets), 2) if wickets > 0 else "N/A"
        dot_pct = round((dots / len(b_df) * 100), 2) if len(b_df) > 0 else 0.0
        
        return {
            'name': b_name, 'overs': overs, 'legal_balls': legal_balls,
            'runs': runs_conceded, 'wickets': wickets, 'econ': econ,
            'avg': avg, 'sr': sr, 'dot_pct': dot_pct
        }

    s1, s2 = get_bowler_stats(bowler1), get_bowler_stats(bowler2)
    if not s1 or not s2:
        print("\nCould not generate comparison. Ensure both bowler names exist in this dataset.")
        return

    s1_dot = f"{s1['dot_pct']}%"
    s2_dot = f"{s2['dot_pct']}%"

    print("\n" + "="*65)
    print(f"       HEAD-TO-HEAD BENCHMARK: {s1['name']} vs {s2['name']}")
    print("="*65)
    print(f"{'Metric':<25} | {s1['name']:<18} | {s2['name']:<18}")
    print("-" * 65)
    print(f"{'Overs Bowled':<25} | {s1['overs']:<18} | {s2['overs']:<18}")
    print(f"{'Wickets':<25} | {s1['wickets']:<18} | {s2['wickets']:<18}")
    print(f"{'Economy Rate':<25} | {s1['econ']:<18} | {s2['econ']:<18}")
    print(f"{'Bowling Average':<25} | {str(s1['avg']):<18} | {str(s2['avg']):<18}")
    print(f"{'Strike Rate (Balls/Wkt)':<25} | {str(s1['sr']):<18} | {str(s2['sr']):<18}")
    print(f"{'Dot Ball %':<25} | {s1_dot:<18} | {s2_dot:<18}")
    print("-" * 65)


if __name__ == "__main__":
    print("\n" + "="*40)
    print("      CRICKET MATCHUP ENGINE          ")
    print("="*40)
    print(" * Database Scope: 2004 - Present")
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
        # Default to IPL
        data_folder = "data/ipl"
        print(f"\nTargeting folder: {data_folder}. Initializing IPL data...")

    # Load the chosen dataset
    match_data = load_and_clean_data(data_folder)

    # Filter by Year
    match_data = filter_by_year(match_data)

    # --- THE INTERACTIVE SCOUT LOOP ---
    while True:
        print("\n" + "-"*75)
        print("Options: [1] Matchup  [2] Bowler  [3] Batter  [4] Vault  [5] Workload  [6] Team Hub  [7] ML Export  [8] Compare  [9] Bowl Tactics  [10] Bat Tactics  [exit] Quit")
        mode = input("Select Mode: ").strip().lower()

        if mode == 'exit':
            print("\nShutting down Matchup Engine. Great session!")
            break

        # --- OPTION 4: CLASSIC MATCHES VAULT ---
        elif mode == '4':
            print("\n--- CLASSIC MATCHES VAULT ---")
            print("[1] 1983 World Cup Final (IND vs WI)")
            vault_choice = input("Select Match: ").strip()

            if vault_choice == '1':
                vault_1983_final()
            else:
                print("Invalid selection or match not yet added to Vault.")
            continue

        # --- OPTION 2: BOWLER SCOUT ---
        elif mode == '2':
            raw_bowler = input("\nEnter Bowler Name: ").strip()
            bowler = resolve_player_name(raw_bowler, match_data)
            if bowler:
                analyze_bowler(match_data, bowler)
            else:
                print(f"Could not find any player matching '{raw_bowler}'. Please try again.")
            continue

        # --- OPTION 3: BATTER SCOUT ---
        elif mode == '3':
            raw_batter = input("\nEnter Batter Name: ").strip()
            batter = resolve_player_name(raw_batter, match_data)
            if batter:
                analyze_batter(match_data, batter)
            else:
                print(f"Could not find any player matching '{raw_batter}'. Please try again.")
            continue

        # --- OPTION 5: WORKLOAD TRACKER ---
        elif mode == '5':
            raw_bowler = input("\nEnter Bowler Name: ").strip()
            bowler = resolve_player_name(raw_bowler, match_data)
            if bowler:
                analyze_workload(match_data, bowler)
            else:
                print(f"Could not find any player matching '{raw_bowler}'. Please try again.")
            continue

        # --- OPTION 6: TEAM HUB ---
        elif mode == '6':
            team_input = input("\nEnter Full Team Name (e.g., Chennai Super Kings): ").strip()
            analyze_team(match_data, team_input)
            continue

        # --- OPTION 7: ML EXPORT PIPELINE ---
        elif mode == '7':
            export_ml_features(match_data)
            continue


        # --- OPTION 8: HEAD-TO-HEAD COMPARATOR ---
        elif mode == '8':
            comp_type = input("\nCompare [1] Batters or [2] Bowlers? ").strip()
            
            if comp_type == '1':
                p1_raw = input("Enter Batter 1 Name: ").strip()
                p2_raw = input("Enter Batter 2 Name: ").strip()
                p1 = resolve_player_name(p1_raw, match_data)
                p2 = resolve_player_name(p2_raw, match_data)
                if p1 and p2:
                    compare_batters(match_data, p1, p2)
                else:
                    print("Could not resolve one or both batter names.")
                    
            elif comp_type == '2':
                p1_raw = input("Enter Bowler 1 Name: ").strip()
                p2_raw = input("Enter Bowler 2 Name: ").strip()
                p1 = resolve_player_name(p1_raw, match_data)
                p2 = resolve_player_name(p2_raw, match_data)
                if p1 and p2:
                    compare_bowlers(match_data, p1, p2)
                else:
                    print("Could not resolve one or both bowler names.")
            continue

        # --- OPTION 9: ADVANCED BOWLING TACTICS ---
        elif mode == '9':
            raw_bowler = input("\nEnter Bowler Name: ").strip()
            bowler = resolve_player_name(raw_bowler, match_data)
            if bowler:
                advanced_bowling_tactics(match_data, bowler)
            else:
                print(f"Could not find any player matching '{raw_bowler}'. Please try again.")
            continue

            # --- OPTION 10: ADVANCED BATTING TACTICS ---
        elif mode == '10':
            raw_batter = input("\nEnter Batter Name for Tactical Breakdown: ").strip()
            batter = resolve_player_name(raw_batter, match_data)
            if batter:
                advanced_batting_tactics(match_data, batter)
            else:
                print(f"Could not find any player matching '{raw_batter}'. Please try again.")
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