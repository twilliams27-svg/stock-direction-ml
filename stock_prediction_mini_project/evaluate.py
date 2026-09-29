import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load data
df= pd.read_csv("feature_data.csv", index_col=0)
X = df.drop("Target", axis=1)
y = df["Target"]

features = ["Return", "SMA_5", "SMA_10", "RSI"]
X = df[features]

#train-test split (70-30)
split_index = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
Y_train, Y_test = y.iloc[:split_index], y.iloc[split_index:]
actual = Y_test.reset_index(drop=True)

#load models
with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

#make predictions
log_pred = pd.Series(logistic_model.predict(X_test))
rf_pred = pd.Series(random_forest_model.predict(X_test))

#compute metrics
def evaluate_model(y_true, y_pred, model_name = "Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} ---")
    print("Mean Absolute Error: (MAE):", round(mae, 4))
    print("Mean Squared Error (MSE):", round(mse, 4))
    print("R^2 Score:", round(r2, 4))
    return {"MAE": mae, "MSE": mse, "R2": r2}

log_metrics = evaluate_model(actual, log_pred, "Logistic Regression")
rf_metrics = evaluate_model(actual, rf_pred, "Random Forest")

#automated comparison
print("\n=== Model Comparison Summary ===")
if log_metrics["MAE"] < rf_metrics["MAE"]:
    print("Logistic Regression performs better based on MAE. (lower is better)")
else:
    print("Random Forest performs better based on MAE. (lower is better)")
if log_metrics["MSE"] < rf_metrics["MSE"]:
    print("Logistic Regression performs better based on MSE. (lower is better)")
else:
    print("Random Forest performs better based on MSE. (lower is better)")
if log_metrics["R2"] > rf_metrics["R2"]:
    print("Logistic Regression performs better based on R^2 Score. (higher is better)")
else:
    print("Random Forest performs better based on R^2 Score. (higher is better)")

#compute per-sample "winner"
log_error = abs(log_pred - actual)
rf_error = abs(rf_pred - actual)

winner = []
for le, re in zip(log_error, rf_error):
    if le < re:
        winner.append("Logistic Regression")
    elif re < le:
        winner.append("Random Forest")
    else:
        winner.append("Tie")

#count wins
log_wins = winner.count("Logistic Regression")
rf_wins = winner.count("Random Forest")
ties = winner.count("Tie")
total = len(winner)

print(f"\nWinner Summary (entire test set of {total} samples):")
print(f"Logistic Regression wins: {log_wins} ({log_wins/total*100:.2f}%)")
print(f"Random Forest wins: {rf_wins} ({rf_wins/total*100:.2f}%)")
print(f"Ties: {ties} ({ties/total*100:.2f}%)")

#plot actual vs predicted
plt.figure(figsize=(12, 4))
plt.plot(actual[:200], label="Actual", marker='o')
plt.plot(log_pred[:200], label="Logistic Predicted", marker='x')
plt.plot(rf_pred[:200], label="Random Forest Predicted", marker='s')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.title("Actual vs Predicted Values (First 200 Samples)")
plt.show()

#add winner markers (presentation purposes)

error_diff = (log_error - rf_error)[:200]
plt.figure(figsize=(12, 4))

#scatter points with colors based on winner
for i, diff in enumerate(error_diff):
    if diff < 0:
        plt.scatter(i, diff, color='green', marker='^', s=100, alpha=0.7)
    elif diff > 0:
        plt.scatter(i, diff, color='orange', marker='v', s=100, alpha=0.7)
    else:
        plt.scatter(i, diff, color='blue', marker='o', s=100, alpha=0.7)

plt.axhline(0, color='black', linestyle='--', linewidth=1.2, label="Tie Line")
plt.xlabel("Sample Index")
plt.ylabel("Error Difference (Logistic - RF)")
plt.title("Model Performance Comparison (First 200 Samples)")
plt.legend(["Tie Line", "Logistic Win (green)", "Random Forest Win (orange)"])
plt.show()
    

#plot prediction errors
plt.figures(figsize=(12, 4))
plt.plot((log_pred - actual)[:200], label="Logistic Regression Errors", marker='x')
plt.plot((rf_pred - actual)[:200], label="Random Forest Errors", marker='s')
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Prediction Error")
plt.title("Prediction Errors (First 200 Samples)")
plt.show()