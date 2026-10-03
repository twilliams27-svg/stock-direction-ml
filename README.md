# stock-direction-ml
ML project evaluating stock price direction based on technical indicators and python!

## Overview!
Can daily price movement (up and down) be predicted using standard technical indicators (RSI, Simple Moving Averages, Daily Returns)?

## Phase 1 (Mini_Project --- completed around 9 months ago for an Algebra 2 project)
* **Approach:** I attempted to predict daily binary price movements using default regression setups.
* **But there was a flaw:** I applied continuous regression metrics ($R^2$) to a binary target variable, but that gave me a confusing negative $R^2$ scores and misleading performance signals.
* **My Recent Realization:** A model predicting direction requires strict binary classification metrics, and not continuous line fitting metrics. But this is a simple fix!

**Phase 1 (baseline) Finding:** The models demonstrated ~50% directional accuracy, aligning with the Efficient Market Hypothesis for shortterm technical indicators.

### Phase 2: (Current Remake~)

**Improvements:** 
* **Improved Code:**I rebuilt the pipeline in Python ('scikit-learn', 'pandas') only using classification metrics: Confusion Matrices, Accuracy, Precision, Recall, and ROC AUC.
* **Evaluation:** Logistic Regression and Random Forest Classifiers.
* **Not so New Findings:** Both models converged around **~50% accuracy** and an **AUC score of~0.50-0.52**. This is still supporting short term market efficiency. In short: basic retail technical indicators alone do not beat random guessing.

## Interactive Web Application ##
I wanted to build an interactive dashboard using **Streamlit Community Cloud** so I could allow users to: 
* Dynamically adjust time series train/test split ratios via slides.
* Retrain Random Forest tree depths.
* Compare live ROC curves and confusion matrices side by side. 
## Technologies Used
* **Python 3**
* **Pandas & NumPy:** Data manipulation and feature extraction
* **Scikit-Learn:** Machine learning models and classification evaluation
* **Matplotlib & Seaborn:** Error analysis and visual performance metrics
