#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚽ Football Match Predictor - Multi-League Web Application
17 Leagues, Separate Models for Each
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load all models and stats
LEAGUES = {}
models_folder = 'models'
stats_folder = 'stats'

print("📂 Loading models and stats...")

# Get all model files
for model_file in os.listdir(models_folder):
    if model_file.startswith('model_') and model_file.endswith('.pkl'):
        league = model_file.replace('model_', '').replace('.pkl', '')
        
        try:
            # Load model
            with open(os.path.join(models_folder, model_file), 'rb') as f:
                model = pickle.load(f)
            
            # Load home and away stats
            home_stats_file = f'stats/home_stats_{league}_2025.csv'
            away_stats_file = f'stats/away_stats_{league}_2025.csv'
            
            if os.path.exists(home_stats_file) and os.path.exists(away_stats_file):
                home_stats = pd.read_csv(home_stats_file, index_col=0)
                away_stats = pd.read_csv(away_stats_file, index_col=0)
                
                LEAGUES[league] = {
                    'model': model,
                    'home_stats': home_stats,
                    'away_stats': away_stats,
                }
                print(f"✅ {league}: Model + Stats loaded")
        except Exception as e:
            print(f"❌ {league}: Error - {str(e)}")

print(f"\n🎉 Total leagues loaded: {len(LEAGUES)}\n")

# League names for display
LEAGUE_NAMES = {
    'B1': '🇧🇪 Belgium - First Division',
    'D1': '🇩🇪 Germany - Bundesliga',
    'D2': '🇩🇪 Germany - 2. Bundesliga',
    'E0': '🇬🇧 England - Premier League',
    'E1': '🇬🇧 England - Championship',
    'E2': '🇬🇧 England - League Two',
    'F1': '🇫🇷 France - Ligue 1',
    'F2': '🇫🇷 France - Ligue 2',
    'G1': '🇬🇷 Greece - Super League',
    'I1': '🇮🇹 Italy - Serie A',
    'I2': '🇮🇹 Italy - Serie B',
    'N1': '🇳🇱 Netherlands - Eredivisie',
    'P1': '🇵🇹 Portugal - Primeira Liga',
    'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland - Premiership',
    'SP1': '🇪🇸 Spain - La Liga',
    'SP2': '🇪🇸 Spain - Segunda División',
    'T1': '🇹🇷 Turkey - Super Lig',
}

features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 
           'HY', 'AY', 'HR', 'AR', 'Home_Percentile', 'Away_Percentile']

@app.route('/')
def index():
    """Main page with league selector"""
    leagues_list = sorted(list(LEAGUES.keys()))
    leagues_info = [(league, LEAGUE_NAMES.get(league, league)) for league in leagues_list]
    return render_template('index_multi_league.html', leagues=leagues_info)

@app.route('/api/teams/<league>')
def get_teams(league):
    """Get teams for selected league"""
    if league not in LEAGUES:
        return jsonify({'error': 'League not found'}), 400
    
    teams = sorted(list(LEAGUES[league]['home_stats'].index))
    return jsonify({'teams': teams})

@app.route('/api/predict/<league>', methods=['POST'])
def predict(league):
    """Predict for selected league"""
    try:
        if league not in LEAGUES:
            return jsonify({'error': 'League not found'}), 400
        
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        league_data = LEAGUES[league]
        model = league_data['model']
        home_stats = league_data['home_stats']
        away_stats = league_data['away_stats']
        
        # Check teams exist
        if home_team not in home_stats.index:
            return jsonify({'error': f'Team "{home_team}" not found'}), 400
        
        if away_team not in away_stats.index:
            return jsonify({'error': f'Team "{away_team}" not found'}), 400
        
        # Get stats
        home_team_stats = home_stats.loc[home_team]
        away_team_stats = away_stats.loc[away_team]
        
        # Build prediction data
        future_match = pd.DataFrame({
            feature: [home_team_stats[feature]] if feature.startswith('H') else [away_team_stats[feature]]
            for feature in features
        })
        
        # Predict
        pred = model.predict(future_match)
        pred_proba = model.predict_proba(future_match)
        
        # Model classes: ['A', 'D', 'H']
        prob_away = float(pred_proba[0][0])
        prob_draw = float(pred_proba[0][1])
        prob_home = float(pred_proba[0][2])
        
        result = {
            'league': league,
            'league_name': LEAGUE_NAMES.get(league, league),
            'home_team': home_team,
            'away_team': away_team,
            'prediction': pred[0],
            'probabilities': {
                'H': round(prob_home * 100, 1),
                'D': round(prob_draw * 100, 1),
                'A': round(prob_away * 100, 1)
            },
            'home_stats': {
                'HS': float(home_team_stats['HS']),
                'HST': float(home_team_stats['HST']),
                'HC': float(home_team_stats['HC']),
                'HF': float(home_team_stats['HF']),
                'HY': float(home_team_stats['HY']),
                'Home_Percentile': float(home_team_stats['Home_Percentile'])
            },
            'away_stats': {
                'AS': float(away_team_stats['AS']),
                'AST': float(away_team_stats['AST']),
                'AC': float(away_team_stats['AC']),
                'AF': float(away_team_stats['AF']),
                'AY': float(away_team_stats['AY']),
                'Away_Percentile': float(away_team_stats['Away_Percentile'])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leagues', methods=['GET'])
def get_leagues():
    """Get list of available leagues"""
    leagues_list = sorted(list(LEAGUES.keys()))
    leagues_info = {league: LEAGUE_NAMES.get(league, league) for league in leagues_list}
    return jsonify(leagues_info)

if __name__ == '__main__':
    print("🚀 Starting Multi-League Football Predictor...")
    print("📱 Open: http://localhost:5000")
    print(f"⚽ Ready to predict! ({len(LEAGUES)} leagues available)")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
