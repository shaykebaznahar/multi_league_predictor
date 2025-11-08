#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚽ Update Stats from 2025 Current Data
עדכון סטטיסטיקות מנתוני 2025 הנוכחיים
"""

import pandas as pd
import os

print("🔄 עדכון סטטיסטיקות 2025...")
print("=" * 60)

# Features we need
features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 
           'HY', 'AY', 'HR', 'AR', 'Home_Percentile', 'Away_Percentile']

# Read 2025 current data
print("\n📂 קריאת נתוני 2025 הנוכחיים...")

try:
    current_data = pd.read_excel('data/current/All_Matches_current_with_Percentiles_2025_Current.xlsx')
    print("✅ קובץ Excel נקרא בהצלחה")
except:
    current_data = pd.read_csv('data/current/all_leagues_2025_current.csv')
    print("✅ קובץ CSV נקרא בהצלחה")

print(f"✅ כולל {len(current_data)} משחקים")

# Get all leagues
leagues = sorted(current_data['Div'].unique())
print(f"✅ {len(leagues)} ליגות נמצאו: {leagues}\n")

# Create stats folder if not exists
os.makedirs('stats', exist_ok=True)

# Update stats for each league
for league in leagues:
    print(f"{'='*60}")
    print(f"📊 עדכון: {league}")
    print(f"{'='*60}")
    
    league_data = current_data[current_data['Div'] == league].copy()
    print(f"📈 משחקים בליגה: {len(league_data)}")
    
    league_data_clean = league_data[features + ['HomeTeam', 'AwayTeam']].dropna()
    print(f"✅ משחקים תקינים: {len(league_data_clean)}")
    
    home_stats = league_data_clean.groupby('HomeTeam')[features].mean()
    away_stats = league_data_clean.groupby('AwayTeam')[features].mean()
    
    print(f"🏟️ טימים בבית: {len(home_stats)}")
    print(f"🚗 טימים בחוץ: {len(away_stats)}")
    
    home_path = f'stats/home_stats_{league}_2025.csv'
    away_path = f'stats/away_stats_{league}_2025.csv'
    
    home_stats.to_csv(home_path)
    away_stats.to_csv(away_path)
    
    print(f"✅ שמור: {home_path}")
    print(f"✅ שמור: {away_path}")
    print()

print("=" * 60)
print("🎉 כל הסטטיסטיקות עודכנו בהצלחה!")
print("=" * 60)
print("\n🔄 הפעל את app.py מחדש להשתמש בנתונים החדשים!")


