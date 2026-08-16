import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re
import warnings

# Suppress minor pandas warnings
warnings.filterwarnings('ignore')

def clean_name(name):
    """Removes Wikipedia citations like [11], [b], and symbols like † or ^"""
    name = re.sub(r'\[.*?\]', '', str(name))
    name = name.replace('†', '').replace('^', '').strip()
    return name

def scrape_wikipedia_records(url, table_index, display_columns, filename, title):
    print(f"\n-> Connecting to Wikipedia: {title}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table', class_='wikitable')
        target_table = tables[table_index]
        
        html_wrapper = StringIO(str(target_table))
        df = pd.read_html(html_wrapper)[0]
        
        # 1. Clean the Player Names
        if 'Player' in df.columns:
            df['Player'] = df['Player'].apply(clean_name)
        elif 'Bowler' in df.columns: # Specifically for the wickets table
            df['Bowler'] = df['Bowler'].apply(clean_name)
            df.rename(columns={'Bowler': 'Player'}, inplace=True)
            
        # 2. Clean numeric columns (convert floats like 463.0 to clean integers like 463)
        for col in display_columns:
            if col in df.columns:
                if col in ['Mat.', 'Runs', 'Wkts']:
                    # Remove commas and decimals
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace('.0', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Save the full list to CSV in the background
        df.to_csv(filename, index=False)
        print(f"-> Success! Full dataset saved to your workspace as '{filename}'.\n")
        
        # 3. Let the user choose how much to see
        print(f"--- {title.upper()} ---")
        limit = input("How many players do you want to see? (Enter a number or 'all'): ").strip().lower()
        
        if limit == 'all':
            print_df = df[display_columns]
        else:
            try:
                print_df = df[display_columns].head(int(limit))
            except ValueError:
                print_df = df[display_columns].head(10) # Default to 10 if invalid input
                
        # 4. Print beautifully formatted, center-justified table
        print("\n" + print_df.to_string(index=False, justify='center'))
        print("-" * 65)
        
    except Exception as e:
        print(f"\nError encountered during scraping: {e}")

def main():
    while True:
        print("\n" + "="*50)
        print("      WEB SCRAPER: WIKIPEDIA TELESCOPE")
        print("="*50)
        print("Select a Record Database to Extract:")
        print("[1] Top All-Time ODI Run Scorers (10,000+ Club)")
        print("[2] Top All-Time Test Run Scorers (10,000+ Club)")
        print("[3] Top All-Time ODI Wicket Takers (300+ Club)")
        print("[exit] Quit Scraper")
        
        choice = input("\nEnter Choice: ").strip().lower()
        
        if choice == '1':
            scrape_wikipedia_records(
                url="https://en.wikipedia.org/wiki/List_of_players_who_have_scored_10,000_or_more_runs_in_One_Day_International_cricket",
                table_index=0,
                display_columns=['Player', 'Team', 'Mat.', 'Runs', 'Avg.'],
                filename="odi_run_legends.csv",
                title="Top ODI Run Scorers"
            )
        elif choice == '2':
            scrape_wikipedia_records(
                url="https://en.wikipedia.org/wiki/List_of_players_who_have_scored_10,000_or_more_runs_in_Test_cricket",
                table_index=0,
                display_columns=['Player', 'Team', 'Mat.', 'Runs', 'Avg.'],
                filename="test_run_legends.csv",
                title="Top Test Run Scorers"
            )
        elif choice == '3':
            scrape_wikipedia_records(
                url="https://en.wikipedia.org/wiki/List_of_bowlers_who_have_taken_300_or_more_wickets_in_One_Day_International_cricket",
                table_index=0,
                display_columns=['Player', 'Team', 'Mat.', 'Wkts', 'Ave.', 'Econ.'],
                filename="odi_wicket_legends.csv",
                title="Top ODI Wicket Takers"
            )
        elif choice == 'exit':
            print("Shutting down Telescope. Great session!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()