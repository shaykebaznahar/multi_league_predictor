#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

LEAGUES = {}
models_folder = 'models'
stats_folder = 'stats'

print("📂 Loading models and stats...")

try:
    current_file = os.path.join(stats_folder, 'All_Matches_current_with_Percentiles_2025_Current.xlsx')
    all_matches_df = pd.read_excel(current_file)
    print(f"✅ Loaded current stats file: {all_matches_df.shape[0]} rows")
except Exception as e:
    print(f"❌ Error loading current stats: {e}")
    all_matches_df = None

for model_file in os.listdir(models_folder):
    if model_file.startswith('model_') and model_file.endswith('.pkl'):
        league = model_file.replace('model_', '').replace('.pkl', '')
        
        try:
            with open(os.path.join(models_folder, model_file), 'rb') as f:
                model = pickle.load(f)
            
            if all_matches_df is not None:
                league_matches = all_matches_df[all_matches_df['Div'] == league].copy()
                
                if len(league_matches) > 0:
                    numeric_cols = league_matches.select_dtypes(include=['number']).columns
                    league_numeric = league_matches[numeric_cols].copy()
                    league_numeric['HomeTeam'] = league_matches['HomeTeam']
                    league_numeric['AwayTeam'] = league_matches['AwayTeam']
                    
                    home_avg = league_numeric.groupby('HomeTeam')[numeric_cols].mean()
                    away_avg = league_numeric.groupby('AwayTeam')[numeric_cols].mean()
                    
                    home_avg = home_avg.dropna(subset=['Home_Percentile'])
                    away_avg = away_avg.dropna(subset=['Away_Percentile'])
                    
                    if len(home_avg) > 0 and len(away_avg) > 0:
                        home_teams = set(home_avg.index)
                        away_teams = set(away_avg.index)
                        all_teams = sorted(list(home_teams | away_teams))
                        
                        LEAGUES[league] = {
                            'model': model,
                            'home_stats': home_avg,
                            'away_stats': away_avg,
                            'teams': all_teams
                        }
                        print(f"✅ {league}: Model + Stats loaded ({len(all_teams)} teams)")
                    else:
                        print(f"⚠️  {league}: No valid stats found")
                else:
                    print(f"⚠️  {league}: No matches found")
        except Exception as e:
            print(f"❌ {league}: Error - {str(e)}")

print(f"\n🎉 Total leagues loaded: {len(LEAGUES)}\n")

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
    leagues_list = sorted(list(LEAGUES.keys()))
    leagues_info = [(league, LEAGUE_NAMES.get(league, league)) for league in leagues_list]
    return render_template('index_multi_league.html', leagues=leagues_info)

@app.route('/api/teams/<league>')
def get_teams(league):
    if league not in LEAGUES:
        return jsonify({'error': 'League not found'}), 400
    
    teams = LEAGUES[league]['teams']
    return jsonify({'teams': teams})

@app.route('/api/predict/<league>', methods=['POST'])
def predict(league):
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
        
        if home_team not in home_stats.index:
            return jsonify({'error': f'Team "{home_team}" not found'}), 400
        
        if away_team not in away_stats.index:
            return jsonify({'error': f'Team "{away_team}" not found'}), 400
        
        home_team_stats = home_stats.loc[home_team]
        away_team_stats = away_stats.loc[away_team]
        
        future_match = pd.DataFrame({
            feature: [home_team_stats[feature]] if feature.startswith('H') else [away_team_stats[feature]]
            for feature in features
            if feature in home_team_stats.index and feature in away_team_stats.index
        })
        
        pred = model.predict(future_match)
        pred_proba = model.predict_proba(future_match)
        
        prob_away = float(pred_proba[0][0]) if pred_proba.shape[1] > 0 else 0
        prob_draw = float(pred_proba[0][1]) if pred_proba.shape[1] > 1 else 0
        prob_home = float(pred_proba[0][2]) if pred_proba.shape[1] > 2 else 0
        
        result = {
            'league': league,
            'league_name': LEAGUE_NAMES.get(league, league),
            'home_team': home_team,
            'away_team': away_team,
            'prediction': str(pred[0]),
            'probabilities': {
                'H': round(prob_home * 100, 1),
                'D': round(prob_draw * 100, 1),
                'A': round(prob_away * 100, 1)
            },
            'home_stats': {
                'HS': round(float(home_team_stats.get('HS', 0)), 1),
                'HST': round(float(home_team_stats.get('HST', 0)), 1),
                'HC': round(float(home_team_stats.get('HC', 0)), 1),
                'HF': round(float(home_team_stats.get('HF', 0)), 1),
                'HY': round(float(home_team_stats.get('HY', 0)), 1),
                'HR': round(float(home_team_stats.get('HR', 0)), 1),
                'Home_Percentile': round(float(home_team_stats.get('Home_Percentile', 0)), 1)
            },
            'away_stats': {
                'AS': round(float(away_team_stats.get('AS', 0)), 1),
                'AST': round(float(away_team_stats.get('AST', 0)), 1),
                'AC': round(float(away_team_stats.get('AC', 0)), 1),
                'AF': round(float(away_team_stats.get('AF', 0)), 1),
                'AY': round(float(away_team_stats.get('AY', 0)), 1),
                'AR': round(float(away_team_stats.get('AR', 0)), 1),
                'Away_Percentile': round(float(away_team_stats.get('Away_Percentile', 0)), 1)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leagues', methods=['GET'])
def get_leagues():
    leagues_list = sorted(list(LEAGUES.keys()))
    leagues_info = {league: LEAGUE_NAMES.get(league, league) for league in leagues_list}
    return jsonify(leagues_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
