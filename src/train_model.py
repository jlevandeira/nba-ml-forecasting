import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Loading the data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "data", "processed", "nba_games_processed.csv")
df = pd.read_csv(csv_path)

# Order the data to garantee a temporal order
df["GAME_DATE_HOME"] = pd.to_datetime(df["GAME_DATE_HOME"])
df = df.sort_values("GAME_DATE_HOME").reset_index(drop=True)

# Defining features and target variable
feature_cols = []

for col in df.columns:
    if "ROLLING_10" in col:
        feature_cols.append(col)

X = df[feature_cols] #features
Y = df["HOME_WIN"] #target variable

# Temporal split: 80% train and 20% test

split_idx = int(len(df)*0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
Y_train = Y.iloc[:split_idx]
Y_test = Y.iloc[split_idx:]

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}" )

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Making predictions
Y_pred = model.predict(X_test_scaled)

# Evaluating the results
acc = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {acc:.2%}")
print(classification_report(Y_test, Y_pred, target_names=["Vitória Fora", "Vitória Casa"]))

# Saving the model and scaler
models_path = os.path.join(SCRIPT_DIR, "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(model, os.path.join(models_path, "nba_model.joblib"))
joblib.dump(scaler, os.path.join(models_path, "scaler.joblib"))

print(f"Model and scaler saved to {models_path}")