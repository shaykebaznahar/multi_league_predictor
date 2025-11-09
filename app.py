#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚽ Football Match Predictor - Web Application
אפליקציית ווב לחיזוי תוצאות משחקי כדורגל
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# טעינת המודל והנתונים
with open('/home/claude/prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

home_stats = pd.read_csv('/home/claude/home_stats_2024_2025.csv', index_col=0)
away_stats = pd.read_csv('/home/claude/away_stats_2024_2025.csv', index_col=0)

features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 
            'HY', 'AY', 'HR', 'AR', 'Home_Percentile', 'Away_Percentile']

teams = sorted(list(set(list(home_stats.index) + list(away_stats.index))))

@app.route('/')
def index():
    """דף ראשי"""
    return render_template('index.html', teams=teams)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API לחיזוי משחק"""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        # בדיקה שהקבוצות קיימות
        if home_team not in home_stats.index:
            return jsonify({'error': f'קבוצת בית "{home_team}" לא נמצאה'}), 400
        
        if away_team not in away_stats.index:
            return jsonify({'error': f'קבוצת חוץ "{away_team}" לא נמצאה'}), 400
        
        # קח את הממוצעים
        home_avg = home_stats.loc[home_team]
        away_avg = away_stats.loc[away_team]
        
        # בנה את הנתונים לחיזוי
        future_match = pd.DataFrame({
            feature: [home_avg[feature]] if feature.startswith('H') else [away_avg[feature]]
            for feature in features
        })
        
        # עשה חיזוי
        pred = model.predict(future_match)
        pred_proba = model.predict_proba(future_match)
        
        # בנה את התשובה
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': pred[0],
            'probabilities': {
                'H': float(pred_proba[0][0]),
                'D': float(pred_proba[0][1]),
                'A': float(pred_proba[0][2])
            },
            'home_stats': {
                'HS': float(home_avg['HS']),
                'HST': float(home_avg['HST']),
                'HC': float(home_avg['HC']),
                'HF': float(home_avg['HF']),
                'HY': float(home_avg['HY']),
                'Home_Percentile': float(home_avg['Home_Percentile'])
            },
            'away_stats': {
                'AS': float(away_avg['AS']),
                'AST': float(away_avg['AST']),
                'AC': float(away_avg['AC']),
                'AF': float(away_avg['AF']),
                'AY': float(away_avg['AY']),
                'Away_Percentile': float(away_avg['Away_Percentile'])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """קבלת רשימת קבוצות"""
    return jsonify({'teams': teams})

if __name__ == '__main__':
    print("🚀 אפליקציה מתחילה...")
    print("📱 גש ל: http://localhost:5000")
    print("⚽ משחקים זמינים!")
    app.run(debug=True, host='localhost', port=5000)
