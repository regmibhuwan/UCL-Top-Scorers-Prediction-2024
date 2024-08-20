# src/final_ml_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
combined_data = pd.read_csv('results/combined_ucl_data.csv')

# Improved feature engineering
def engineer_features(data):
    player_stats = data.groupby('Name').agg({
        'Season': 'count',
        'G': ['sum', 'mean'],
        'P': ['sum', 'mean'],
        'GoalsPerGame': ['mean', 'std']
    }).reset_index()
    
    player_stats.columns = ['Name', 'Seasons_Played', 'Total_Goals', 'Avg_Goals_Per_Season',
                            'Total_Games', 'Avg_Games_Per_Season', 'Avg_GoalsPerGame', 'Std_GoalsPerGame']
    
    # Calculate consistency score
    player_stats['Consistency_Score'] = player_stats['Seasons_Played'] / 5 * (1 - player_stats['Std_GoalsPerGame'])
    
    # Calculate trend
    def calc_trend(player_data):
        if len(player_data) < 2:
            return 0
        first_gpg = player_data.iloc[0]['GoalsPerGame']
        last_gpg = player_data.iloc[-1]['GoalsPerGame']
        return (last_gpg - first_gpg) / first_gpg  # Relative change

    trend_data = data.sort_values('Season').groupby('Name').apply(calc_trend).reset_index(name='Trend')
    player_stats = player_stats.merge(trend_data, on='Name', how='left')
    
    # Get most recent team
    recent_team = data.sort_values('Season').groupby('Name').last()[['Team', 'Season']]
    player_stats = player_stats.merge(recent_team, on='Name', how='left')
    
    return player_stats

features = engineer_features(combined_data)

# Filter out players with less than 2 seasons or less than 10 total games
features = features[(features['Seasons_Played'] >= 2) & (features['Total_Games'] >= 10)]

# Prepare the features and target
X = features[['Seasons_Played', 'Total_Goals', 'Avg_Goals_Per_Season', 'Total_Games', 
              'Avg_Games_Per_Season', 'Avg_GoalsPerGame', 'Consistency_Score', 'Trend']]
y = features['Avg_GoalsPerGame']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores)}")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Make predictions for next season
features['Predicted_GoalsPerGame'] = model.predict(scaler.transform(X))

# Adjust predictions based on consistency and recent performance
features['Final_Prediction'] = features['Predicted_GoalsPerGame'] * (1 + features['Consistency_Score']) * (1 + features['Trend'])

# Top 10 predictions
top_10 = features.nlargest(10, 'Final_Prediction')
print("\nTop 10 Predicted Scorers for 2024-25 Season:")
print(top_10[['Name', 'Team', 'Final_Prediction', 'Avg_GoalsPerGame', 'Consistency_Score', 'Trend', 'Seasons_Played']])

# Save predictions
top_10.to_csv('results/final_ml_predictions.csv', index=False)

print("\nPredictions saved to 'results/final_ml_predictions.csv'")