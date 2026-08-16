import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Cricket Analytics Pro", layout="wide")

# 2. Inject Custom CSS for styling
st.markdown("""
    <style>
    .main-title { font-size: 42px; font-weight: 800; color: #1E88E5; margin-bottom: 0px;}
    .sub-title { font-size: 18px; color: #616161; margin-bottom: 30px;}
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# 3. Dashboard Header
st.markdown('<p class="main-title">🏏 Tactical Cricket Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Data Science & Machine Learning Pipeline</p>', unsafe_allow_html=True)
st.divider()

# 4. Sidebar Navigation
st.sidebar.header("Navigation Hub")
view_mode = st.sidebar.radio("Select Module:", ["Historical Vault (Scraper)", "Tactical AI Pipeline", "Live AI Predictor"])

# --- MODULE 1: THE SCRAPER DATA ---
if view_mode == "Historical Vault (Scraper)":
    st.subheader("All-Time ODI Records")
    try:
        # Load the CSV we generated earlier
        df = pd.read_csv("all_time_odi_legends.csv")
        
        # Display the data as an interactive table
        st.dataframe(df, width="stretch")
        
        # Build a quick visual bar chart
        st.markdown("### Top 10 Run Scorers (Visualized)")
        st.bar_chart(data=df.head(10), x='Player', y='Runs', color="#1E88E5")
        
    except FileNotFoundError:
        st.error("⚠️ No scraper data found. Run 'backend/scraper.py' first to generate the dataset!")

# --- MODULE 2: THE ML FEATURES ---
elif view_mode == "Tactical AI Pipeline":
    st.subheader("Phase-by-Phase Machine Learning Features")
    try:
        # Load the newly exported ML features
        ml_df = pd.read_csv("ml_tactical_features.csv")
        
        # --- NEW: INTERACTIVE DROPDOWN ---
        st.markdown("### 🎯 Interactive Tactical Scout")
        
        # Grab every unique team from the dataset automatically
        teams = sorted(ml_df['batting_team'].unique())
        selected_team = st.selectbox("Select a Team to Analyze:", teams)
        
        # Filter the data based on what you clicked in the dropdown
        team_data = ml_df[ml_df['batting_team'] == selected_team]
        
        # --- DYNAMIC METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches Processed", team_data['match_id'].nunique())
        col2.metric("Total Boundaries", team_data['boundaries'].sum())
        col3.metric("Total Wickets Lost", team_data['wickets_lost'].sum())
        
        # --- DYNAMIC CHART ---
        st.markdown(f"**{selected_team} - Runs & Boundaries by Phase**")
        
        # Group the data by phase so the chart can read it
        phase_chart_data = team_data.groupby('phase_of_play')[['total_runs', 'boundaries']].sum()
        
        # Generate the visual chart
        st.bar_chart(phase_chart_data, color=["#1E88E5", "#FF4B4B"])
        
        st.markdown("### Filtered AI Training Data")
        st.dataframe(team_data, width="stretch")
        
    except FileNotFoundError:
        st.error("⚠️ No tactical AI data found. Run Option 7 in 'backend/engine.py' to extract the features!")

# --- MODULE 3: LIVE AI PREDICTOR ---
elif view_mode == "Live AI Predictor":
    st.subheader("🤖 Live Matchup Prediction Engine")
    st.markdown("Enter batter vs bowler matchup stats to predict who wins the battle.")
    
    import joblib
    
    try:
        # Load the AI's brain from the file you just created
        model = joblib.load('cricket_ai_model.pkl')
        
       # --- THE CRICBUZZ MULTI-FORMAT ARCHITECTURE ---
        # 1. Format Selector Dropdown
        format_choice = st.selectbox(
            "🏆 Select Match Format:", 
            ["T20 / IPL", "One Day Internationals (ODI)", "Test Matches"]
        )
        
        # Map formats to dataset files
        format_file_map = {
            "T20 / IPL": "t20_matchup_data.csv",
            "One Day Internationals (ODI)": "odi_matchup_data.csv",
            "Test Matches": "test_matchup_data.csv"
        }
        
        selected_file = format_file_map[format_choice]
        
        # 2. Dynamic Caching Loader
        @st.cache_data
        def load_raw_data(file_path):
            return pd.read_csv(file_path, low_memory=False)
            
        try:
            raw_df = load_raw_data(selected_file)
        except FileNotFoundError:
            st.error(f"⚠️ Dataset file '{selected_file}' not found. Please export it from the backend engine first.")
            st.stop()
        
        # 2. Create the Search Dropdowns
        col1, col2 = st.columns(2)
        with col1:
            batter_list = sorted(raw_df['batter'].dropna().unique())
            selected_batter = st.selectbox("🏏 Select Batter", batter_list)
        with col2:
            bowler_list = sorted(raw_df['bowler'].dropna().unique())
            selected_bowler = st.selectbox("🥎 Select Bowler", bowler_list)
            
        # 3. Automatically filter the dataset for this exact head-to-head matchup
        matchup_df = raw_df[(raw_df['batter'] == selected_batter) & (raw_df['bowler'] == selected_bowler)]
        
        if matchup_df.empty:
            st.warning(f"No historical data found for {selected_batter} vs {selected_bowler}.")
            # Provide default 0s so the app doesn't crash if they haven't played each other
            balls_faced, dot_pct, total_boundaries, dismissals = 0, 0.0, 0, 0
        else:
            # 4. Calculate the exact stats on the fly!
            balls_faced = len(matchup_df[matchup_df['extras_type'] != 'wides'])
            dots = len(matchup_df[matchup_df['runs_batter'] == 0])
            dot_pct = round((dots / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
            total_boundaries = len(matchup_df[matchup_df['runs_batter'].isin([4, 6])])
            dismissals = matchup_df['wicket_type'].notna().sum()
            
            # 5. Display the real stats in beautiful metric cards
            st.markdown("### 📊 Head-to-Head Record")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balls Faced", balls_faced)
            c2.metric("Dot %", f"{dot_pct}%")
            c3.metric("Boundaries", total_boundaries)
            c4.metric("Dismissals", dismissals)      
        # Prediction Button
        if st.button("Run AI Prediction", type="primary"):
            # Structure the input exactly how the Random Forest expects it
            input_data = pd.DataFrame([[balls_faced, dot_pct, total_boundaries, dismissals]], 
                                      columns=['balls_faced', 'dot_pct', 'total_boundaries', 'dismissals'])
            
            # Ask the AI for its prediction and confidence score
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0].max()
            
           # Translate machine code to human text
        if prediction == 1:
            verdict_text = "Batter Dominates (High Strike Rate expected)"
            verdict_color = "green"
        else:
            verdict_text = "Bowler Dominates (Low Scoring / Wicket likely)"
            verdict_color = "red"
            
        st.divider()
        st.markdown(f"### AI Verdict: :{verdict_color}[**{verdict_text}**]")
        st.progress(float(confidence))
        st.markdown(f"**AI Confidence Score:** {round(confidence * 100, 1)}%")
            
    except FileNotFoundError:
        st.error("⚠️ AI Model not found. Please train and save the model in the backend first!")        