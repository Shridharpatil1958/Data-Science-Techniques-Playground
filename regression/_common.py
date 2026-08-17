"""
Shared helpers for regression scripts.
Not a standalone script — imported by the individual model files.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "datasets/titanic.csv"
TARGET = "Fare"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch"]


def load_and_prepare(path: str = DATA_PATH, target: str = TARGET, features: list = FEATURES):
    df = pd.read_csv(path)

    df = df.copy()
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def evaluate_regressor(model, X_test, y_test, model_name: str = "Model", n_features: int = None):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"{model_name} — Evaluation")
    print(f"{'=' * 50}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")

    if n_features:
        n = len(y_test)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        print(f"Adjusted R2: {adj_r2:.4f}")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
