# stock-direction-ml
ML project evaluating stock price direction based on technical indicators and python!

## Overview!
Can daily price movement (up/down) be predicted using standard technical indicators (RSI, Simple Moving Averages, Daily Returns)?

## Phase 1 (Initial Implementations)
**Goal:** Predict daily price movements using Logistic Regression and Random Forest models.

**Retrospective Analysis:** Version 1 evaluated binary price classification (Target = 1 or 0) using continuous regression metrics (MAE, MSE, R^2). Calculating R^2 on binary targets resulted in negative scores (~ -1.02). This was a fundamental metric mismatch between regression metrics and classification outcomes.

**Baseline Finding:** The models demonstrated ~50% directional accuracy, aligning with the Efficient Market Hypothesis for shortterm technical indicators.

### Phase 2: Model Refactoring & Classification Evaluation

**Improvements:** Refactored the evaluation pipeline to utilize proper binary classification metrics like Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
**Goal:** Benchmark performance against random coin flip baselines and test additional feature engineering.

## Technologies Used
**Python 3**
**Pandas & NumPy:** Data manipulation and feature extraction
**Scikit-Learn:** Machine learning models and classification evaluation
**Matplotlib & Seaborn:** Error analysis and visual performance metrics
