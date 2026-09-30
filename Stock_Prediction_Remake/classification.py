import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Load the Data
df = pd.read_csv("feature_data.csv", index_col=0, parse_dates=True)
df = df.dropna()

features = ["Return", "SMA_5", "SMA_10", "RSI"]
X = df[features]
y = df["Target"]

# 2. Time-Series Split (70-30)
split_index = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 3. Feature Scaling (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Models
# Using scaled data for Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Random Forest does not require scaled data, but we use consistent parameters
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Make Predictions & Get Probabilities
log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1] # Probability of "1" (Up)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# 6. Print Classification Metrics
print("=== Logistic Regression Metrics ===")
print(classification_report(y_test, log_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, log_prob):.4f}\n")

print("=== Random Forest Metrics ===")
print(classification_report(y_test, rf_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_prob):.4f}\n")

# 7. Visualization 1: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Heatmap
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted Label (0 = Down, 1 = Up)')
axes[0].set_ylabel('Actual Label')

# Random Forest Heatmap
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted Label (0 = Down, 1 = Up)')
axes[1].set_ylabel('Actual Label')

plt.tight_layout()
plt.show()

# 8. Visualization 2: ROC Curve Comparison
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, log_prob):.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)') # Baseline coin-flip
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Model Performance vs. Random Guessing')
plt.legend()
plt.grid(alpha=0.3)
plt.show()