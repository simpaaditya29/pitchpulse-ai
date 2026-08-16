import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
# Suppress minor warnings for a clean terminal output
warnings.filterwarnings('ignore')

def train_matchup_model(csv_path="ml_matchup_features.csv"):
    print("\n" + "="*55)
    print("      AI PREDICTIVE MODEL: MATCHUP ANALYSIS")
    print("="*55)

    # 1. Load the Data
    try:
        df = pd.read_csv(csv_path)
        print(f"-> Successfully loaded {len(df)} historical matchups.")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        print("Please run the ML Export Pipeline (Option 7) in the engine first!")
        return

    # 2. Feature Engineering: Define the Target Variable
    # We will train the AI to classify who "wins" the historical matchup.
    # We define a Batter win as maintaining a Strike Rate > 130. 
    df['Matchup_Winner'] = df['strike_rate'].apply(lambda x: 1 if x > 130 else 0)

    # 3. Select Features (X) and Target (y)
    df['total_boundaries'] = df['fours'] + df['sixes']
    
    # These are the stats the AI will use to find patterns
    features = ['balls_faced', 'dot_pct', 'total_boundaries', 'dismissals']
    X = df[features]
    y = df['Matchup_Winner']

    # 4. Train-Test Split (80% training data, 20% testing data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize and Train the Random Forest AI
    print("-> Training Random Forest Classifier across 100 decision trees...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Make Predictions and Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n--- MODEL EVALUATION ---")
    print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
    print("\nDetailed Classification Report:")
    
    # This prints out the precision and recall for both outcomes
    report = classification_report(y_test, predictions, target_names=['Bowler Wins (SR <= 130)', 'Batter Wins (SR > 130)'])
    print(report)

    # 7. Feature Importance (What stats matter most to the AI?)
    print("--- AI FEATURE IMPORTANCE ---")
    print("Which stats have the biggest impact on winning a matchup?")
    importances = model.feature_importances_
    for feature, imp in zip(features, importances):
        print(f"{feature:<20}: {round(imp * 100, 2)}%")
        
    print("-" * 55)
    print("Model training complete! The AI has successfully learned the dataset.")

    # 8. Save the Trained Model to Disk
    import joblib
    joblib.dump(model, 'cricket_ai_model.pkl')
    print("-> AI Model successfully saved as 'cricket_ai_model.pkl'")

if __name__ == "__main__":
    train_matchup_model()