# stock-direction-ml
ML project evaluating stock price direction based on technical indicators and python!

## Overview!
Can daily price movement (up/down) be predicted using standard technical indicators (RSI, Simple Moving Averages, Daily Returns)?

## Phase 1 (Mini_Project --- completed around 9 months ago for an Algebra 2 project)
* **Approach:** Attempted to predict daily binary price movements using default regression setups.
* **Flaw Identified:** I applied continuous regression metrics ($R^2$) to a binary target variable, yielding confusing negative $R^2$ scores and misleading performance signals.
* **Realization:** A model predicting direction requires strict binary classification metrics, not continuous line fitting metrics.

**Baseline Finding:** The models demonstrated ~50% directional accuracy, aligning with the Efficient Market Hypothesis for shortterm technical indicators.

### Phase 2: (Remake)

**Improvements:** 
* **Improved Code:** Rebuilt the pipeline in Python ('scikit-learn', 'pandas') strictly using classification metrics: Confusion Matrices, Accuracy, Precision, Recall, and ROC-AUC.
* **Evaluation:** Logistic Regression and Random Forest Classifiers.
* **Key Findings:** Both models converged around **~50% accuracy** and an **AUC score o f~0.50-0.52**. This provides evidence supporting short term markey efficiency-proving that basic retail technical indicators alone do not beat random guessing!

## Interactive Web Application ##
Built and deployed an interactive dashboard via **Streamlit Community Cloud** allowing users to: 
* Dynamically adjust time series train/test split ratios via slides.
* Retrain Random Forest tree depths in real time.
* Compare live ROC curves and confusion matrices side by side. 
## Technologies Used
**Python 3**
**Pandas & NumPy:** Data manipulation and feature extraction
**Scikit-Learn:** Machine learning models and classification evaluation
**Matplotlib & Seaborn:** Error analysis and visual performance metrics
