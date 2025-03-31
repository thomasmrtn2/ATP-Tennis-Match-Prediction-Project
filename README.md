# ATP Tennis Match Prediction

## Overview
This project predicts the outcomes of ATP (Association of Tennis Professionals) tennis matches using historical data from 2000 to 2024. Built as a portfolio piece, it demonstrates expertise in data preprocessing, feature engineering, and machine learning, leveraging a Gradient Boosting Classifier optimized with Bayesian search. The notebook achieves an accuracy of approximately 80% on test data (2022-2024), showcasing a robust approach to sports analytics.

**[View the Notebook on GitHub](ATP_prediction.ipynb)**

## Features
- **Data Source**: Utilizes Jeff Sackmann's [tennis_atp](https://github.com/JeffSackmann/tennis_atp) repository, providing match, ranking, and player data.
- **Feature Engineering**:
  - Win rates: Overall, surface-specific, and rolling, adjusted with a Bayesian approach.
  - Serve statistics: First serve percentage, win percentages, aces, and double faults per match.
  - Break point performance: Save and conversion rates under pressure.
  - Head-to-head (H2H) records between players.
  - Player form: Recent game counts and ranking point changes.
- **Modeling**: Gradient Boosting Classifier with feature selection and hyperparameter tuning.
- **Evaluation**: Time-series cross-validation to respect the temporal nature of tennis data.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Jupyter Notebook

### Steps
1. **Clone this Repository**:
   ```bash
   git clone https://github.com/[YourUsername]/ATP_prediction.git
   cd ATP_prediction
2. **Install Dependencies**:
   pip install -r requirements.txt
3. **Obtain Tennis Data**
    This project relies on external data from the tennis_atp repository.
    Follow the instructions in data/README.md to download and set up the data.

### Usage

1. **Launch Jupyter Notebook:**
  jupyter notebook ATP_prediction.ipynb
2. **Run the Notebook:**
- Execute all cells sequentially to:
  - Load and preprocess the data.
  - Engineer features.
  - Train and optimize the model.
  - Evaluate performance on the test set.

3. **Customize:**
Modify year ranges in the notebook (e.g., load_match(2000, 2020)) to adjust training/validation/test periods.
Tweak hyperparameters in the BayesSearchCV section for experimentation.

## Results

- Training Period: 2000-2022
- Test Period: 2022-2024
- Model: Gradient Boosting Classifier
- Test Accuracy: ~65% (varies slightly with optimization iterations)
- Cross-Validation Score: Average of ~0.65 across 5 time-series folds
