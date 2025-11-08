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
    all_stats_df = pd.read_csv(os.path.join(stats_folder, 'all_leagues_2005_2024.csv'))
    print(f"✅ Loaded combined stats file: {all_stats_df.shape[0]} rows")
except Exception as e:
    print(f"❌ Error loading combined stats: {e}")
    all_stats_df = None

for model_file in os.listdir(models_folder):
    if model_file.startswith('model_') and model_file.endswith('.pkl'):
        league = model_file.replace('model_', '').replace('.pkl', '')
        
        try:
            with open(os.path.join(models_folder, model_file), 'rb') as f:
                model = pickle.load(f)
            
            if all_stats_df is not None:
                league_stats = all_stats_df[all_stats_df['Div'] == league].copy()
                
                if len(league_stats) > 0:
                    home_stats = league_stats.copy()
                    away_stats = league_stats.copy()
                    
                    if 'HomeTeam' in home_stats.columns:
                        home_stats = home_stats.set_index('HomeTeam')
                        away_stats = away_stats.set_index('AwayTeam')
                    elif 'Team' in home_stats.columns:
                        home_stats = home_stats.set_index('Team')
                        away_stats = away_stats.set_index('Team')
                    
                    LEAGUES[league] = {
                        'model': model,
                        'home_stats': home_stats,
                        'away_stats': away_stats,
                        'teams': sorted(list(set(list(home_stats.index) + list(away_stats.index))))
                    }
                    print(f"✅ {league}: Model + Stats loaded ({len(LEAGUES[league]['teams'])} teams)")
                else:
                    print(f"⚠️  {league}: No stats found in combined file")
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
        
        if isinstance(home_team_stats, pd.DataFrame):
            home_team_stats = home_team_stats.mean()
        if isinstance(away_team_stats, pd.DataFrame):
            away_team_stats = away_team_stats.mean()
        
        future_match = pd.DataFrame({
            feature: [home_team_stats[feature]] if feature.startswith('H') else [away_team_stats[feature]]
            for feature in features
            if feature in home_team_stats.index and feature in away_team_stats.index
        })
        
        pred = model.predict(future_match)
        pred_proba = model.predict_proba(future_match)
        
        prob_away = float(pred_proba[0][0]) if len(pred_proba[0]) > 0 else 0
        prob_draw = float(pred_proba[0][1]) if len(pred_proba[0]) > 1 else 0
        prob_home = float(pred_proba[0][2]) if len(pred_proba[0]) > 2 else 0
        
        result = {
            'league': league,
            'league_name': LEAGUE_NAMES.get(league, league),
            'home_team': home_team,
            'away_team': away_team,
            'prediction': pred[0],
            'probabilities': {
                'H': prob_home,
                'D': prob_draw,
                'A': prob_away
            },
            'home_stats': {
                'HS': float(home_team_stats['HS']) if 'HS' in home_team_stats.index else 0,
                'HST': float(home_team_stats['HST']) if 'HST' in home_team_stats.index else 0,
                'HC': float(home_team_stats['HC']) if 'HC' in home_team_stats.index else 0,
                'HF': float(home_team_stats['HF']) if 'HF' in home_team_stats.index else 0,
                'HY': float(home_team_stats['HY']) if 'HY' in home_team_stats.index else 0,
                'Home_Percentile': float(home_team_stats['Home_Percentile']) if 'Home_Percentile' in home_team_stats.index else 0
            },
            'away_stats': {
                'AS': float(away_team_stats['AS']) if 'AS' in away_team_stats.index else 0,
                'AST': float(away_team_stats['AST']) if 'AST' in away_team_stats.index else 0,
                'AC': float(away_team_stats['AC']) if 'AC' in away_team_stats.index else 0,
                'AF': float(away_team_stats['AF']) if 'AF' in away_team_stats.index else 0,
                'AY': float(away_team_stats['AY']) if 'AY' in away_team_stats.index else 0,
                'Away_Percentile': float(away_team_stats['Away_Percentile']) if 'Away_Percentile' in away_team_stats.index else 0
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
