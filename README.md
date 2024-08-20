# UEFA Champions League Top Scorers Prediction

This project uses machine learning to predict the top scorers for the upcoming UEFA Champions League (UCL) season based on historical data from 2019 to 2023.

## Project Overview

We've developed a machine learning model that analyzes past performances of players in the UEFA Champions League to predict potential top scorers for future seasons. The model considers various factors such as consistency across seasons, goal-scoring ratio, and recent performance trends.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/regmibhuwan/UCL-Top-Scorers-Prediction-2024.git
   cd UCL-Top-Scorers-Prediction-2024
   ```

2. Install required packages:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

## Usage

1. Run the data analysis script:
   ```
   python src/data_analysis.py
   ```

2. Run the prediction model:
   ```
   python src/final_ml_prediction.py
   ```

## Model Overview

Our model uses a Random Forest Regressor and considers factors such as:
- Consistency across multiple seasons
- Goal scoring ratio
- Recent performance trends
- Total games played

## Sample Results

Top 5 predicted scorers for the 2024-25 season (based on historical UCL data and current team status):

1. Mohamed Salah (Liverpool)
2. Kylian Mbappé (Paris Saint-Germain)
3. Vinícius Júnior (Real Madrid)
4. Robert Lewandowski (Barcelona)
5. Erling Haaland (Manchester City)

**Note on Predictions:**
- Our model bases predictions on historical UCL performance data from 2019 to 2023.
- The predictions reflect potential performance if the player participates in the UCL, based on their historical data.
- The model does not account for recent transfers or changes in player circumstances that might affect UCL participation.

For full results and detailed analysis, see `results/final_ml_predictions.csv`.

## Key Findings

- Consistent performers over multiple UCL seasons are ranked higher.
- Recent performance trends significantly impact predictions.
- The model balances long-term consistency with recent form.

## Limitations and Future Improvements

- Current Limitation: The model doesn't automatically account for players' current teams or leagues, which may affect UCL eligibility.
- Future Improvement: Incorporate current league and team information to filter out players not eligible for UCL.
- Future Improvement: Update the dataset with the most recent transfer information.
- Future Improvement: Include additional factors such as player age, team strength, and domestic league performance.
- Future Improvement: Implement a system to regularly update player status and eligibility for more accurate predictions.

## Data Sources

- Historical UCL top scorer data from 2019 to 2023 seasons.

## Contributing

We welcome contributions and suggestions to improve the accuracy and relevance of our predictions! Please feel free to open an issue or submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).