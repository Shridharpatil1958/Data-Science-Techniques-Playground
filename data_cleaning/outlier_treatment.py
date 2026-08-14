"""
Outlier Treatment
==================
Detect and treat outliers using the IQR method, with boxplot
visualization support.

Usage:
    python data_cleaning/outlier_treatment.py
"""

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def iqr_bounds(df: pd.DataFrame, column: str, k: float = 1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    lower, upper = iqr_bounds(df, column, k)
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"{column}: {len(outliers)} outliers found (bounds: {lower:.2f}, {upper:.2f})")
    return outliers


def cap_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    lower, upper = iqr_bounds(df, column, k)
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    lower, upper = iqr_bounds(df, column, k)
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()


def plot_boxplot(df: pd.DataFrame, column: str, save_path: str = None):
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[column].dropna(), vert=False)
    plt.title(f"Boxplot: {column}")
    plt.xlabel(column)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved boxplot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    df = load_data()
    for col in ["Fare", "Age"]:
        if col in df.columns:
            detect_outliers_iqr(df, col)
            plot_boxplot(df, col, save_path=f"screenshots/boxplot_{col}.png")
    df_capped = cap_outliers_iqr(df, "Fare") if "Fare" in df.columns else df
    print("\nOutlier treatment complete.")
