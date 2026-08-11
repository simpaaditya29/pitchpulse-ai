import pandas as pd
import glob
import os
import csv

def load_and_clean_data(folder_path):
    print("Starting Matchup Engine Data Ingestion...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} match files. Cleaning and compiling...\n")
    
    all_deliveries = []
    
    # Loop through every single match file in the data folder
    for file in csv_files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Cricsheet mixes metadata and delivery data. 
                # We ONLY want the rows that start with the word 'ball'.
                if row and row[0] == 'ball':
                    all_deliveries.append(row)
                    
    # Define the exact columns for the data we just extracted
    headers = ['type', 'innings', 'over_ball', 'batting_team', 'striker', 'non_striker', 'bowler', 'runs_batter', 'runs_extras', 'dismissal_type', 'player_dismissed']
    
    # Convert our cleaned raw data into a powerful Pandas DataFrame
    # We slice row[:11] to ensure we only grab the core columns and avoid any weird extra commas
    cleaned_data = [row[:11] for row in all_deliveries]
    df = pd.DataFrame(cleaned_data, columns=headers)
    
    print(f"Success! Engine has extracted {len(df)} clean deliveries.")
    return df

if __name__ == "__main__":
    data_folder = "data" 
    
    # Run the ingestion tool
    match_data = load_and_clean_data(data_folder)
    
    print("\nData Preview:")
    print(match_data.head())