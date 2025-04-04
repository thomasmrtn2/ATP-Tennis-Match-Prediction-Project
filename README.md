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

1.  **Data Loading & Initial Processing:**
    * Match data from 2010 to 2024 is loaded.
    * Dates are processed to estimate actual match dates based on tournament start date and round.
    * Matches are sorted chronologically.

2.  **Target Balancing (`flip_dataset`):**
    * To avoid bias from the arbitrary assignment of "winner" and "loser" columns, 50% of the matches are randomly selected to have their Player 1 and Player 2 data swapped. A target variable `winner` (0 or 1) is created accordingly.

3.  **Feature Engineering Pipeline:** A series of functions generate features, carefully avoiding data leakage by using historical data only:
    * **Serve Stats Imputation (`ServeStatsImputerV2`):** Imputes missing serve statistics (aces, double faults, points won, etc.) using historical player averages, falling back to surface averages and then a default value.
    * **Minutes Imputation (`minutes_Imputer`):** Imputes missing match duration (`minutes`) using the median duration based on the number of games and sets played in the match.
    * **Recent Playtime (`get_minutes_played_recent`):** Calculates the total minutes played by each player over a recent time window (e.g., 14 days) to estimate fatigue.
    * **Elo Ratings (`elo_feature`):** Calculates dynamic Elo ratings for each player, both globally and specific to the match surface. Elo differences are key predictors.
    * **Head-to-Head (`get_h2h_v2`):** Computes historical and recent (last N matches) head-to-head win/loss records between the two players.
    * **Serve/Return/Pressure Ratings (`serve_stats_rating`, `return_stats_rating`, `under_pressure_rating`):** Creates composite scores based on weighted rolling averages (e.g., last 365 days) of various statistics:
        * *Serve:* First serve %, serve points won %, aces, double faults, service games won %.
        * *Return:* Return points won %, break points converted %.
        * *Pressure:* Break points saved/converted %, tiebreaks won %, deciding sets won %.
    * **Recent Form (`get_recent_form`):** Calculates the number of matches won by each player in their last N (e.g., 10) matches.
    * **Match Importance (`get_match_importance`):** Assigns a score based on the tournament level (Grand Slam, Masters, etc.) and round (Final, SF, QF, etc.).

4.  **Modeling & Evaluation:**
    * **Models Tested:**
        * Random Forest Classifier
        * AdaBoost Classifier
        * Gradient Boosting Classifier
        * A Dense Neural Network (MLP) using TensorFlow/Keras
    * **Cross-Validation:** `TimeSeriesSplit` is used for cross-validation on the training set to respect data chronology.
    * **Evaluation Metrics:** Accuracy is the primary metric. Confusion matrices are analyzed to understand error patterns.
---

## 🛠️ Feature Engineering Highlights

* **Composite Performance Ratings:** A key element of this project is the creation of unique, composite ratings that combine multiple raw statistics into single, weighted scores reflecting specific aspects of a player's game. These ratings are calculated using rolling time windows (365 days) to capture recent form and are generated both globally and for specific surfaces:
    * **Serve Rating (`serve_stats_rating`):**
        * *Components:* Aggregates rolling averages of 1st serve %, 1st serve points won %, 2nd serve points won %, aces per match (normalized), double faults per match (normalized, negative impact), and service games won %.
        * *Originality:* Combines serve effectiveness (win %) and consistency (% in) with serve weapons (aces) and weaknesses (DFs). Uses **surface-specific weights** (e.g., aces and 1st serve win % weighted higher on Grass) and normalization for count-based stats to create a holistic, context-aware serve performance score.
    * **Return Rating (`return_stats_rating`):**
        * *Components:* Aggregates rolling averages of opponent's 1st serve points won % (inverted to get returner's win %), opponent's 2nd serve points won % (inverted), opponent's service games won % (inverted), and break points converted %. All components are normalized.
        * *Originality:* Focuses on the player's ability to neutralize opponent serves and capitalize on break opportunities. Uses **surface-specific weights** (e.g., break point conversion weighted higher on Clay) and normalization to create a comprehensive return game assessment.
    * **Under Pressure Rating (`under_pressure_rating`):**
        * *Components:* Aggregates rolling averages of break points saved %, break points converted % (using opponent's faced/saved stats), tiebreaks won %, and deciding sets won %.
        * *Originality:* Quantifies player performance in high-stakes situations. Combines clutch serving (BP saved), clutch returning (BP converted), tiebreak nerve, and endurance/focus in decisive sets. Also utilizes **surface-specific weights** (e.g., tiebreak performance weighted higher on faster surfaces like Grass).
        * 
---

## 🚀 Technologies Used

* Python 3.x
* pandas
* numpy
* scikit-learn
* tensorflow (for the Neural Network part)
* matplotlib
* seaborn


---

## 📈 Results

* The tree-based models (Random Forest, AdaBoost, Gradient Boosting) achieved cross-validation accuracies generally in the range of **~67-69%** on the training folds using `TimeSeriesSplit`.
* The Multi-Layer Perceptron (MLP) achieved an accuracy of approximately **64.7%** on its dedicated test set after scaling.
* **Feature Importance:** Analysis (e.g., using Random Forest feature importances) revealed that **Elo-based features** (especially Elo difference and surface-specific Elo difference) are the most significant predictors in the model.
* **Error Analysis:** Predictions are less accurate when the Elo difference between players is small (i.e., matches are predicted to be close). The model performs better when there is a clearer skill gap indicated by Elo ratings.

These results highlight the **uncertainty inherent in tennis**. The sport retains a fundamental layer of unpredictability, where day-to-day form, match dynamics, and mental toughness can overturn predictions. 
This is exactly what **makes tennis so thrilling and suspenseful.**

---

## 💡 Future Work

Potential avenues for future development and improvement include:

* **Advanced Hyperparameter Tuning:** Employ more rigorous optimization techniques like Bayesian Optimization (as hinted in the original script comments) specifically tailored for `TimeSeriesSplit` to fine-tune the best performing models (e.g., Gradient Boosting).
* **Expanded Dataset:**
    * Integrate data from other tours (WTA, Challengers) for broader analysis or transfer learning experiments.
    * Include betting odds data as a market-based feature reflecting perceived win probabilities.
* **Advanced Modeling:**
    * Experiment with sequence models (LSTMs, GRUs) to better capture temporal dependencies in player form.
    * Explore Graph Neural Networks (GNNs) to model player relationships and head-to-head networks.
* **Model Interpretability:** Utilize techniques like SHAP or LIME to gain deeper insights into individual predictions and feature contributions, especially for the custom ratings.
* **Deployment:** Develop a pipeline for making real-time predictions during ongoing tournaments.

---

## 👨‍💻 Author

**Thomas Martin** - [GitHub Profile](https://github.com/thomasmrtn2)

---

## 🙏 Acknowledgments

Special thanks to Jeff Sackmann for his exhaustive dataset and its public availability via the [tennis_atp repository](https://github.com/JeffSackmann/tennis_atp).
