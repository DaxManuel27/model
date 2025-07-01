import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


API_KEY = "82ebde4cb6c34ab6bb75d6f910fc27d7"
HEADERS = {
    'X-Auth-Token': API_KEY
}
BASE = 'https://api.football-data.org/v4'
season = "2024"

params = {'season': 2024}
url = f"{BASE}/competitions/PL/matches"
resp = requests.get(url, headers=HEADERS, params=params)
matches = resp.json()['matches']
df = pd.json_normalize(matches)

# Save to CSV
df.to_csv('premier_league_matches_2024.csv', index=False)
print("Data saved to premier_league_matches_2024.csv")

# Clean and prepare data
df['result'] = df['score.winner'].replace({'DRAW': 0, 'HOME_TEAM': 1, 'AWAY_TEAM': 2})
df['home_team_score'] = df['score.fullTime.home']
df['away_team_score'] = df['score.fullTime.away']
df = df.dropna(subset=['home_team_score', 'away_team_score'])

# Sort by date to ensure chronological order
df['utcDate'] = pd.to_datetime(df['utcDate'])
df = df.sort_values('utcDate')

print(f"Total matches: {len(df)}")

def calculate_recent_form(df, team, date, n_games=5):
    """Calculate recent form for a team before a given date"""
    # Get recent matches for the team (both home and away)
    recent_matches = df[
        ((df['homeTeam.name'] == team) | (df['awayTeam.name'] == team)) & 
        (df['utcDate'] < date)
    ].tail(n_games)
    
    if len(recent_matches) == 0:
        return {
            'goals_scored': 0, 'goals_conceded': 0, 'points': 0, 
            'wins': 0, 'draws': 0, 'losses': 0, 'games_played': 0
        }
    
    goals_scored = 0
    goals_conceded = 0
    points = 0
    wins = draws = losses = 0
    
    for _, match in recent_matches.iterrows():
        if match['homeTeam.name'] == team:
            # Team playing at home
            team_goals = match['home_team_score']
            opp_goals = match['away_team_score']
        else:
            # Team playing away
            team_goals = match['away_team_score']
            opp_goals = match['home_team_score']
        
        goals_scored += team_goals
        goals_conceded += opp_goals
        
        if team_goals > opp_goals:
            points += 3
            wins += 1
        elif team_goals == opp_goals:
            points += 1
            draws += 1
        else:
            losses += 1
    
    return {
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'points': points,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'games_played': len(recent_matches)
    }

def calculate_home_away_form(df, team, date, venue='home', n_games=10):
    """Calculate home/away specific form"""
    if venue == 'home':
        venue_matches = df[
            (df['homeTeam.name'] == team) & (df['utcDate'] < date)
        ].tail(n_games)
        goals_scored = venue_matches['home_team_score'].sum()
        goals_conceded = venue_matches['away_team_score'].sum()
    else:
        venue_matches = df[
            (df['awayTeam.name'] == team) & (df['utcDate'] < date)
        ].tail(n_games)
        goals_scored = venue_matches['away_team_score'].sum()
        goals_conceded = venue_matches['home_team_score'].sum()
    
    games_played = len(venue_matches)
    if games_played == 0:
        return {
            'goals_scored': 0, 
            'goals_conceded': 0, 
            'games_played': 0,
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0
        }
    
    return {
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'games_played': games_played,
        'avg_goals_scored': goals_scored / games_played,
        'avg_goals_conceded': goals_conceded / games_played
    }

def calculate_head_to_head(df, home_team, away_team, date, n_games=5):
    """Calculate head-to-head record between two teams"""
    h2h_matches = df[
        (((df['homeTeam.name'] == home_team) & (df['awayTeam.name'] == away_team)) |
         ((df['homeTeam.name'] == away_team) & (df['awayTeam.name'] == home_team))) &
        (df['utcDate'] < date)
    ].tail(n_games)
    
    if len(h2h_matches) == 0:
        return {
            'total_goals': 0, 'home_team_goals': 0, 'away_team_goals': 0,
            'home_team_wins': 0, 'away_team_wins': 0, 'draws': 0,
            'games_played': 0, 'avg_total_goals': 0
        }
    
    total_goals = 0
    home_team_goals = away_team_goals = 0
    home_team_wins = away_team_wins = draws = 0
    
    for _, match in h2h_matches.iterrows():
        total_goals += match['home_team_score'] + match['away_team_score']
        
        if match['homeTeam.name'] == home_team:
            home_team_goals += match['home_team_score']
            away_team_goals += match['away_team_score']
            if match['home_team_score'] > match['away_team_score']:
                home_team_wins += 1
            elif match['home_team_score'] < match['away_team_score']:
                away_team_wins += 1
            else:
                draws += 1
        else:
            home_team_goals += match['away_team_score']
            away_team_goals += match['home_team_score']
            if match['away_team_score'] > match['home_team_score']:
                home_team_wins += 1
            elif match['away_team_score'] < match['home_team_score']:
                away_team_wins += 1
            else:
                draws += 1
    
    return {
        'total_goals': total_goals,
        'home_team_goals': home_team_goals,
        'away_team_goals': away_team_goals,
        'home_team_wins': home_team_wins,
        'away_team_wins': away_team_wins,
        'draws': draws,
        'games_played': len(h2h_matches),
        'avg_total_goals': total_goals / len(h2h_matches) if len(h2h_matches) > 0 else 0
    }

print("Calculating enhanced features for each match...")

# Calculate enhanced features for each match
enhanced_features = []
for i, (idx, row) in enumerate(df.iterrows()):
    if i % 50 == 0:
        print(f"Processing match {i+1}/{len(df)}")
    
    home_team = row['homeTeam.name']
    away_team = row['awayTeam.name']
    match_date = row['utcDate']
    
    # Recent form (last 5 games)
    home_form = calculate_recent_form(df, home_team, match_date, 5)
    away_form = calculate_recent_form(df, away_team, match_date, 5)
    
    # Home/Away specific form
    home_home_form = calculate_home_away_form(df, home_team, match_date, 'home', 10)
    away_away_form = calculate_home_away_form(df, away_team, match_date, 'away', 10)
    
    # Head-to-head
    h2h = calculate_head_to_head(df, home_team, away_team, match_date, 5)
    
    features = {
        'home_team': home_team,
        'away_team': away_team,
        'matchday': row['matchday'],
        
        # Recent form features
        'home_recent_goals_scored': home_form['goals_scored'],
        'home_recent_goals_conceded': home_form['goals_conceded'],
        'home_recent_points': home_form['points'],
        'home_recent_wins': home_form['wins'],
        
        'away_recent_goals_scored': away_form['goals_scored'],
        'away_recent_goals_conceded': away_form['goals_conceded'],
        'away_recent_points': away_form['points'],
        'away_recent_wins': away_form['wins'],
        
        # Home advantage features
        'home_home_avg_scored': home_home_form['avg_goals_scored'],
        'home_home_avg_conceded': home_home_form['avg_goals_conceded'],
        'away_away_avg_scored': away_away_form['avg_goals_scored'],
        'away_away_avg_conceded': away_away_form['avg_goals_conceded'],
        
        # Head-to-head features
        'h2h_avg_total_goals': h2h['avg_total_goals'],
        'h2h_home_advantage': h2h['home_team_wins'] - h2h['away_team_wins'] if h2h['games_played'] > 0 else 0,
        
        # Target variables
        'home_goals': row['home_team_score'],
        'away_goals': row['away_team_score']
    }
    
    enhanced_features.append(features)

# Create enhanced dataframe
enhanced_df = pd.DataFrame(enhanced_features)

print(f"Enhanced features calculated. Shape: {enhanced_df.shape}")

# Prepare features for modeling
feature_columns = [
    'matchday',
    'home_recent_goals_scored', 'home_recent_goals_conceded', 'home_recent_points', 'home_recent_wins',
    'away_recent_goals_scored', 'away_recent_goals_conceded', 'away_recent_points', 'away_recent_wins',
    'home_home_avg_scored', 'home_home_avg_conceded',
    'away_away_avg_scored', 'away_away_avg_conceded',
    'h2h_avg_total_goals', 'h2h_home_advantage'
]

# Create feature matrix
X_enhanced = enhanced_df[['home_team', 'away_team'] + feature_columns].copy()
y_home = enhanced_df['home_goals']
y_away = enhanced_df['away_goals']

# Preprocessing with both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('teams', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team']),
        ('features', StandardScaler(), feature_columns)
    ]
)

# Split data chronologically (last 20%)
split_index = int(len(X_enhanced) * 0.8)
X_train_enhanced = X_enhanced.iloc[:split_index]
X_test_enhanced = X_enhanced.iloc[split_index:]
y_train_home = y_home.iloc[:split_index]
y_test_home = y_home.iloc[split_index:]
y_train_away = y_away.iloc[:split_index]
y_test_away = y_away.iloc[split_index:]

# Fit preprocessor and transform data
X_train_processed = preprocessor.fit_transform(X_train_enhanced)
X_test_processed = preprocessor.transform(X_test_enhanced)

# Convert to numpy arrays
y_train_home = np.array(y_train_home)
y_test_home = np.array(y_test_home)
y_train_away = np.array(y_train_away)
y_test_away = np.array(y_test_away)

# Train enhanced models
print("Training enhanced models...")
lr_home_enhanced = LinearRegression()
lr_home_enhanced.fit(X_train_processed, y_train_home)
y_pred_home_enhanced = lr_home_enhanced.predict(X_test_processed)

lr_away_enhanced = LinearRegression()
lr_away_enhanced.fit(X_train_processed, y_train_away)
y_pred_away_enhanced = lr_away_enhanced.predict(X_test_processed)

# Clip negative predictions to 0 (goals can't be negative)
y_pred_home_enhanced = np.maximum(0, y_pred_home_enhanced)
y_pred_away_enhanced = np.maximum(0, y_pred_away_enhanced)

# Calculate BTTS predictions
btts_pred_enhanced = (y_pred_home_enhanced > 0.5) & (y_pred_away_enhanced > 0.5)
btts_actual_enhanced = (y_test_home > 0) & (y_test_away > 0)


# Calculate BTTS accuracy
correct_predictions_enhanced = np.sum(btts_pred_enhanced == btts_actual_enhanced)
total_predictions_enhanced = len(y_test_home)
accuracy_enhanced = (correct_predictions_enhanced / total_predictions_enhanced) * 100

print(f"\nENHANCED BTTS PREDICTION ACCURACY:")
print(f"Correct predictions: {correct_predictions_enhanced}/{total_predictions_enhanced}")
print(f"Accuracy: {accuracy_enhanced:.1f}%")

print("\nSample Enhanced Predictions:")
for i in range((len(y_test_home))):
    home = X_test_enhanced.iloc[i]['home_team']
    away = X_test_enhanced.iloc[i]['away_team']
    home_pred = y_pred_home_enhanced[i]
    away_pred = y_pred_away_enhanced[i]
    home_actual = y_test_home[i]
    away_actual = y_test_away[i]
    
    btts_prediction = "YES" if btts_pred_enhanced[i] else "NO"
    btts_actual_result = "YES" if btts_actual_enhanced[i] else "NO"
    
    print(f"{home} vs {away}")
    print(f"  Predicted: {home_pred:.1f}-{away_pred:.1f} | BTTS: {btts_prediction}")
    print(f"  Actual: {home_actual:.0f}-{away_actual:.0f} | BTTS: {btts_actual_result}")
    print()

