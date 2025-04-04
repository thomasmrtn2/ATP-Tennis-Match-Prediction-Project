# 🎾 ATP Tennis Match Outcome Prediction with Machine Learning


## 📖 Description

This project aims to predict the outcomes of ATP men's tennis matches using **machine learning** techniques. It implements a complete pipeline, from loading historical data to evaluating an optimized model, with thorough feature engineering and rigorous temporal handling to avoid data leakage.

The main goal is to demonstrate a methodical approach to solving a complex temporal prediction problem, focusing on:
* Extracting and creating relevant features (players serves, return and under pressure ratings, recent form, H2H, rankings).
* Properly handling temporal data for training and evaluation (using only past data).
* Robust model evaluation using `TimeSeriesSplit`.

---

## 🎯 Problem Statement

Predicting the outcome of a tennis match is challenging due to the multitude of factors influencing performance (current form, surface, head-to-head history, playing conditions, etc.). This project attempts to model these influences using historical data to build a reliable predictor.

---

## 📊 Data

The dataset comes from **Jeff Sackmann's comprehensive GitHub repository**: [tennis_atp](https://github.com/JeffSackmann/tennis_atp).  
The repository includes detailed information on:
* Match results (singles and doubles) since 1968.
* Historical ATP rankings.
* Player data (e.g., date of birth, height, dominant hand, etc.).

For this project, we focus on ATP singles matches between **2000 and 2024**.

---

## 🛠️ Methodology

The project follows a structured pipeline (mostly implemented in `Tennis_Prediction.ipynb`):

1. **Data Loading:**
   - Load match data from CSV files for the selected period.

2. **Initial Preprocessing:**
   - Convert date columns to datetime format.
   - Sort the data by date to maintain chronological order.

3. **Target Handling:**
   - The `flip_dataset` function randomly swaps (50% chance) the `winner_*` and `loser_*` columns and creates a binary target `winner` (0 if Player 1 wins, 1 if Player 2 wins).  
   - This ensures the model doesn’t simply learn that "Player 1 always wins" and creates a balanced binary classification problem.

4. **Feature Engineering:**
   - Numerous features are created to capture different aspects of the game, **ensuring only past information is used for each match to avoid data leakage**:
     - **Minutes Played Recently:** Number of minutes played by each player in the last window matches. (`get_minutes_played_recent`).
     - **Elo rankings:** A global and surface specific elo ranking for each players. The initial elo is 1500 (Can be modified) (`elo_feature`).
     - **Head-to-Head (H2H):** win/loss record between players during their last 10 matches. (`get_h2h`).
     - **Serve Rating:** A rating of each players service for global and surface specific matches. To compute this statistic I used this formula :
       $$

    \text{Serve Rating} = \text{weight}_1 \times (\text{First Serve Ratio}) + \text{weight}_2 \times (\text{1st Serve Points Won}) + \ldots
    
   $$
The coefficient are different depending on the surface. (`get_serve_statistics`).
     - **Break Point Statistics:** Percentages of break points saved/converted, number of break points per match (`get_break_points_statistics`).
     - **Rankings:** Evolution of ATP points over time (`get_players_rank_stats`).

5. **Missing Data Imputation:**
   - Median imputation for serve/break statistics (`ServeStatisticsImputer`).
   - Advanced **KNN imputation** (after standardization) for rankings and other key statistics, integrated **properly** within a `ColumnTransformer` to prevent data leakage during cross-validation and on the test set.

6. **Final Preprocessing (via `ColumnTransformer`):**
   - Drop redundant raw columns.
   - One-hot encode categorical variables (surface, tournament, etc.).
   - Standardize remaining numerical features (`StandardScaler`).

7. **Modeling:**
   - Initial comparison of several models (RandomForest, AdaBoost, GradientBoosting).
   - Selection of `GradientBoostingClassifier` as the main model.

8. **Hyperparameter Optimization:**
   - Use **Bayesian optimization** (`BayesSearchCV` from `scikit-optimize`) to find the best hyperparameters for the entire pipeline (including feature selection with `SelectKBest` and `GradientBoostingClassifier` parameters).
   - Validate with **`TimeSeriesSplit`** to respect the temporal dependency of the data.

9. **Evaluation:**
   - Cross-validation accuracy obtained during Bayesian optimization.
   - Final evaluation on an independent test set (future data), measuring generalization performance.

---

## 🚀 Technologies Used

* **Python 3.x**
* **Pandas**: Data manipulation and analysis.
* **NumPy**: Numerical computation.
* **Scikit-learn**:
  - Pipelines, preprocessing, machine learning models, metrics, cross-validation (`TimeSeriesSplit`), `ColumnTransformer`, `KNNImputer`, `StandardScaler`, and `SelectKBest`.
* **Scikit-optimize (`skopt`)**: Bayesian optimization (`BayesSearchCV`).
* **Matplotlib / Seaborn**: Visualizations (used during exploration and analysis).
* **Re (Regular Expressions)**: For score parsing.

---

## 📈 Results

* **Optimal Model**: `GradientBoostingClassifier` (integrated into a complete Scikit-learn pipeline with preprocessing, KNN imputation, and feature selection).
* **Best Cross-Validation Accuracy**: **67%**  
  (Average accuracy from time-series cross-validation during Bayesian optimization).
* **Test Set Accuracy (2022-2024)**: **65%**  
  (Final accuracy evaluated on unseen matches).

These results highlight the uncertainty inherent in tennis. The sport retains a fundamental layer of unpredictability, where day-to-day form, match dynamics, and mental toughness can overturn predictions. This is exactly what makes tennis so thrilling and suspenseful.

---

## 💡 Future Work

* Integrate more dynamic rating systems (e.g., ELO ratings) to better capture relative player form.
* Add fatigue-related features (recent time on court, match frequency, travel effects).
* Explore deep learning techniques (e.g., RNNs or transformers for match sequences) if the complexity and data volume justify it.
* Conduct error analysis:
  - Identify match types (surface, tournament level, specific H2H pairs) where the model struggles the most.
* Experiment with different rolling time windows for performance metrics.

---

## 👨‍💻 Author

**Thomas Martin** - [GitHub Profile](https://github.com/thomasmrtn2)

---

## 🙏 Acknowledgments

Special thanks to Jeff Sackmann for his exhaustive dataset and its public availability via the [tennis_atp repository](https://github.com/JeffSackmann/tennis_atp).
