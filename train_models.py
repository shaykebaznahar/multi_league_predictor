import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

print("🤖 Training models for each league...\n")

# Load combined data
df = pd.read_csv('data/combined/all_leagues_2005_2024.csv')

# Create output folders
os.makedirs('models', exist_ok=True)
os.makedirs('stats', exist_ok=True)

# Features to use
features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 
           'HY', 'AY', 'HR', 'AR', 'Home_Percentile', 'Away_Percentile']

# Get all leagues
leagues = sorted(df['Div'].unique())

print(f"Found {len(leagues)} leagues: {leagues}\n")

# Extract year from YEAR column (you have 'YEAR' not 'Date')
df['Year'] = df['YEAR']

# Train model for each league
for league in leagues:
    print(f"{'='*50}")
    print(f"Training: {league}")
    print(f"{'='*50}")
    
    # Filter data for this league
    league_data = df[df['Div'] == league].copy()
    
    print(f"Total matches: {len(league_data)}")
    
    # Split into train (2005-2019) and test (2020-2024)
    train_data = league_data[league_data['Year'] < 2020]
    test_data = league_data[(league_data['Year'] >= 2020) & (league_data['Year'] < 2025)]
    
    print(f"Training matches: {len(train_data)}")
    print(f"Testing matches: {len(test_data)}")
    
    # Skip if not enough data
    if len(train_data) < 100:
        print(f"⚠️ Not enough training data for {league}, skipping...\n")
        continue
    
    # Remove rows with missing values
    train_clean = train_data[features + ['FTR']].dropna()
    test_clean = test_data[features + ['FTR']].dropna()
    
    print(f"After cleaning - Train: {len(train_clean)}, Test: {len(test_clean)}")
    
    if len(test_clean) == 0:
        print(f"⚠️ No test data for {league}, skipping...\n")
        continue
    
    # Train model
    X_train = train_clean[features]
    y_train = train_clean['FTR']
    
    X_test = test_clean[features]
    y_test = test_clean['FTR']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Accuracy: {accuracy:.2%}\n")
    
    # Save model
    model_path = f'models/model_{league}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path}")
    
    # Calculate stats for current season (2024 or 2025)
    current_season = league_data[league_data['Year'] == 2025]
    
    if len(current_season) == 0:
        current_season = league_data[league_data['Year'] == 2024]
    
    if len(current_season) > 0:
        home_stats = current_season.groupby('HomeTeam')[features].mean()
        away_stats = current_season.groupby('AwayTeam')[features].mean()
        
        home_stats.to_csv(f'stats/home_stats_{league}_2025.csv')
        away_stats.to_csv(f'stats/away_stats_{league}_2025.csv')
        
        print(f"✅ Stats saved for {len(home_stats)} teams")
    else:
        print(f"⚠️ No recent season data for {league}")
    
    print(f"\n")

print("🎉 All models trained!")