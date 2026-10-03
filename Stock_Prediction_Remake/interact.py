import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

#Page setup
st.set_page_config(page_title="Stock Movement ML Predictor", layout="wide")

st.title("Stock Movement Machine Learning Predictor")
st.write(
    "This tool evaluates whether technical indicators (RSI, Moving Averages, Returns) "
    "can or cannot reliably predict short term stock price direction"
)

#Sidebar controls
st.sidebar.header("Interactive Parameters")
split_ratio = st.sidebar.slider("Training Data Split Ratio", min_value=0.50, max_value=0.85, value=0.70, step=0.05)
rf_trees = st.sidebar.slider("Random Forest Trees (n_estimators)", min_value=50, max_value=300, value=200, step=50)

#dataset path relative to interact.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "feature_data.csv")

#Load data helper function
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df.dropna()

try:
    df = load_data()

    features = ["Return", "SMA_5", "SMA_10", "RSI"]
    X = df[features]
    y = df["Target"]

    #Time Series Split controlled by a slider
    split_index = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    #Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Training Models
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=rf_trees, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    #predictions
    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    #KPI metrics display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", f"{(log_pred == y_test).mean():.1%}")
        st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, log_prob):.2f}")

    with col2:
        st.subheader("Random Forest Classifier")
        st.metric("Accuracy", f"{(rf_pred == y_test).mean():.1%}")
        st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, rf_prob):.2f}")

    st.markdown("---")

    #Visual tabs
    tab1, tab2 = st.tabs(["Confusion Matrices", "ROC Curve Analysis"])

    with tab1:
        fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Logistic Regression')
        axes[0].set_xlabel('Predicted (0 = Down, 1 = Up)')
        axes[0].set_ylabel('Actual Label')

        sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Random Forest')
        axes[1].set_xlabel('Predicted (0 = Down, 1 = Up)')
        axes[1].set_ylabel('Actual Label')
        st.pyplot(fig1)

    with tab2:
        fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, log_prob):.2f})', color='blue')
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.2f})', color='green')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing Baseline (AUC = 0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve: Model Performance vs. Coin Flip')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig2)

    st.info(
        "**For Readers:**\n\n"
        "Even though I used two different machine learning architectures, both models have around **~50% accuracy** "
        "and an **AUC score of ~0.50**. So this supports the **Efficient Market Hypothesis (EMH)**: "
        "basic retail indicators (alone) do not have sufficient predictive signal to consistently beat random guesses."
    )

except FileNotFoundError:
    st.error("Missing dataset! Make sure `feature_data.csv` is uploaded to GitHub inside the `Stock_Prediction_Remake` folder.")
