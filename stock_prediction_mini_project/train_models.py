import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("feature_data.csv", index_col=0, parse_dates=True)

df = df.dropna()

X = df[["Return", "SMA_5", "SMA_10", "RSI"]]
y = df["Target"]

#split
split_index = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
Y_train, Y_test = y.iloc[:split_index], y.iloc[split_index:]

#LR
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(Y_test, log_pred)

#rf Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_pred)

print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)
#Saving the models
pickle.dump(log_model, open("logistic_model.pkl", "wb"))
pickle.dump(rf_model, open("random_forest_model.pkl", "wb"))
