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

df.to_csv('premier_league_matches_2024.csv', index=False)
print("Data saved to premier_league_matches_2024.csv")

df['result'] = df['score.winner'].replace({'DRAW': 0, 'HOME_TEAM': 1, 'AWAY_TEAM': 2})
df['home_team_score'] = df['score.fullTime.home']
df['away_team_score'] = df['score.fullTime.away']
df = df.dropna(subset=['home_team_score', 'away_team_score'])

df['utcDate'] = pd.to_datetime(df['utcDate'])
df = df.sort_values('utcDate')

print(f"Total matches: {len(df)}")

print("Creating basic features...")
features_list = []
for i, (idx, row) in enumerate(df.iterrows()):
    features = {
        'home_team': row['homeTeam.name'],
        'away_team': row['awayTeam.name'],
        'matchday': row['matchday'],
        'home_goals': row['home_team_score'],
        'away_goals': row['away_team_score']
    }
    features_list.append(features)

df_features = pd.DataFrame(features_list)

print(f"Features created. Shape: {df_features.shape}")

feature_columns = ['matchday']

X = df_features[['home_team', 'away_team'] + feature_columns].copy()
y_home = df_features['home_goals']
y_away = df_features['away_goals']

preprocessor = ColumnTransformer(
    transformers=[
        ('teams', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team']),
        ('features', StandardScaler(), feature_columns)
    ]
)

split_index = int(len(X) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train_home = y_home.iloc[:split_index]
y_test_home = y_home.iloc[split_index:]
y_train_away = y_away.iloc[:split_index]
y_test_away = y_away.iloc[split_index:]

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

y_train_home = np.array(y_train_home)
y_test_home = np.array(y_test_home)
y_train_away = np.array(y_train_away)
y_test_away = np.array(y_test_away)

print("Training models...")
lr_home = LinearRegression()
lr_home.fit(X_train_processed, y_train_home)
y_pred_home = lr_home.predict(X_test_processed)

lr_away = LinearRegression()
lr_away.fit(X_train_processed, y_train_away)
y_pred_away = lr_away.predict(X_test_processed)

y_pred_home = np.maximum(0, y_pred_home)
y_pred_away = np.maximum(0, y_pred_away)

btts_pred = (y_pred_home > 0.5) & (y_pred_away > 0.5)
btts_actual = (y_test_home > 0) & (y_test_away > 0)

correct_predictions = np.sum(btts_pred == btts_actual)
total_predictions = len(y_test_home)
accuracy = (correct_predictions / total_predictions) * 100

print("MODEL RESULTS:")
print("=" * 50)
print("HOME TEAM MODEL:")
print("MSE:", mean_squared_error(y_test_home, y_pred_home))
print("R² score:", r2_score(y_test_home, y_pred_home))

print("\nAWAY TEAM MODEL:")
print("MSE:", mean_squared_error(y_test_away, y_pred_away))
print("R² score:", r2_score(y_test_away, y_pred_away))

print(f"\nBTTS PREDICTION ACCURACY:")
print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {accuracy:.1f}%")

print("\nSample Predictions:")
for i in range(min(10, len(y_test_home))):
    home = X_test.iloc[i]['home_team']
    away = X_test.iloc[i]['away_team']
    home_pred = y_pred_home[i]
    away_pred = y_pred_away[i]
    home_actual = y_test_home[i]
    away_actual = y_test_away[i]
    
    btts_prediction = "YES" if btts_pred[i] else "NO"
    btts_actual_result = "YES" if btts_actual[i] else "NO"
    
    print(f"{home} vs {away}")
    print(f"  Predicted: {home_pred:.1f}-{away_pred:.1f} | BTTS: {btts_prediction}")
    print(f"  Actual: {home_actual:.0f}-{away_actual:.0f} | BTTS: {btts_actual_result}")
    print()

