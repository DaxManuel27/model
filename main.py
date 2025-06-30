import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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

# Print available columns to debug
print("Available columns:")
print(df.columns.tolist())

# Save to CSV
df.to_csv('premier_league_matches_2024.csv', index=False)

print("Data saved to premier_league_matches_2024.csv")


#Model
df['result'] = df['score.winner'].replace({'DRAW': 0, 'HOME_TEAM': 1, 'AWAY_TEAM': 2})
df['home_team_score'] = df['score.fullTime.home']
df['away_team_score'] = df['score.fullTime.away']
df = df.dropna(subset=['home_team_score', 'away_team_score'])

# Sort by matchday to ensure chronological order
df = df.sort_values('matchday')

X = df[['homeTeam.name', 'awayTeam.name', 'matchday']]
y_home = df['score.fullTime.home']
y_away = df['score.fullTime.away']

preprocessor = ColumnTransformer(
    transformers=[
        ('teams', OneHotEncoder(handle_unknown='ignore'), ['homeTeam.name', 'awayTeam.name'])
    ],
    remainder='passthrough' 
)

# Split BEFORE encoding to keep team names
X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42, shuffle=False)
_, _, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42, shuffle=False)

# Now encode
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Convert y to numpy arrays
y_train_home = np.array(y_train_home)
y_test_home = np.array(y_test_home)
y_train_away = np.array(y_train_away)
y_test_away = np.array(y_test_away)

# Train model for home team goals
lr_home = LinearRegression()
lr_home.fit(X_train_encoded, y_train_home)
y_pred_home = lr_home.predict(X_test_encoded)

# Train model for away team goals
lr_away = LinearRegression()
lr_away.fit(X_train_encoded, y_train_away)
y_pred_away = lr_away.predict(X_test_encoded)

# Calculate BTTS predictions
btts_pred = (y_pred_home > 0.5) & (y_pred_away > 0.5)  # Both teams predicted to score > 0.5 goals
btts_actual = (y_test_home > 0) & (y_test_away > 0)  # Both teams actually scored

print("HOME TEAM MODEL:")
print("MSE:", mean_squared_error(y_test_home, y_pred_home))
print("R² score:", r2_score(y_test_home, y_pred_home))

print("\nAWAY TEAM MODEL:")
print("MSE:", mean_squared_error(y_test_away, y_pred_away))
print("R² score:", r2_score(y_test_away, y_pred_away))

print("\nBTTS PREDICTIONS vs ACTUAL:")
X_test_reset = X_test.reset_index(drop=True)

for i in range(min(10, len(y_test_home))):
    home = X_test_reset.loc[i, 'homeTeam.name']
    away = X_test_reset.loc[i, 'awayTeam.name']
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