# src/data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent.parent
data_dir = current_dir / 'data'
results_dir = current_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Read and combine data from all seasons
seasons = range(2019, 2024)
dfs = []

for season in seasons:
    df = pd.read_csv(data_dir / f'ucl_top_scorers_{season}.csv')
    df['Season'] = season
    dfs.append(df)

combined_data = pd.concat(dfs, ignore_index=True)

# Calculate goals per game
combined_data['GoalsPerGame'] = combined_data['G'] / combined_data['P']

# Display basic statistics
print("Basic Statistics:")
print(combined_data.describe())

# Display top 10 goal scorers across all seasons
print("\nTop 10 Goal Scorers Across All Seasons:")
print(combined_data.sort_values('G', ascending=False).head(10))

# Display top 10 players by goals per game (minimum 5 games played)
print("\nTop 10 Players by Goals per Game (min. 5 games):")
print(combined_data[combined_data['P'] >= 5].sort_values('GoalsPerGame', ascending=False).head(10))

# Create visualizations
plt.figure(figsize=(12, 6))
sns.scatterplot(data=combined_data, x='P', y='G', hue='Season', size='GoalsPerGame', sizes=(20, 200))
plt.title('Goals vs. Games Played by Season')
plt.savefig(results_dir / 'goals_vs_games.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=combined_data, x='Season', y='GoalsPerGame')
plt.title('Distribution of Goals per Game by Season')
plt.savefig(results_dir / 'goals_per_game_distribution.png')
plt.close()

# Count appearances for each player
player_appearances = combined_data['Name'].value_counts().reset_index()
player_appearances.columns = ['Name', 'Appearances']

# Get players with multiple appearances
multiple_appearances = player_appearances[player_appearances['Appearances'] > 1]

print("\nPlayers with Multiple Appearances:")
print(multiple_appearances)

# Analyze trend for players with multiple appearances
trend_data = combined_data[combined_data['Name'].isin(multiple_appearances['Name'])]
trend_pivot = trend_data.pivot(index='Name', columns='Season', values='GoalsPerGame')
trend_pivot['Trend'] = trend_pivot[2023] - trend_pivot[2019]
trend_pivot = trend_pivot.sort_values('Trend', ascending=False)

print("\nTrend Analysis for Players with Multiple Appearances:")
print(trend_pivot)

# Save processed data
combined_data.to_csv(results_dir / 'combined_ucl_data.csv', index=False)
trend_pivot.to_csv(results_dir / 'player_trends.csv')

print("\nAnalysis complete. Results saved in the 'results' directory.")