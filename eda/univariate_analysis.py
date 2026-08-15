"""
Univariate Analysis
====================
Histograms, distribution plots, and boxplots for individual
numeric and categorical features.

Usage:
    python eda/univariate_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30, save_path: str = None):
    plt.figure(figsize=(7, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    _finish(save_path)


def plot_boxplot(df: pd.DataFrame, column: str, save_path: str = None):
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[column].dropna())
    plt.title(f"Boxplot of {column}")
    _finish(save_path)


def plot_countplot(df: pd.DataFrame, column: str, save_path: str = None):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[column].dropna())
    plt.title(f"Count of {column}")
    _finish(save_path)


def _finish(save_path: str = None):
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    df = load_data()

    numeric_cols = [c for c in ["Age", "Fare"] if c in df.columns]
    categorical_cols = [c for c in ["Sex", "Pclass", "Embarked", "Survived"] if c in df.columns]

    for col in numeric_cols:
        plot_histogram(df, col, save_path=f"screenshots/hist_{col}.png")
        plot_boxplot(df, col, save_path=f"screenshots/box_{col}.png")

    for col in categorical_cols:
        plot_countplot(df, col, save_path=f"screenshots/count_{col}.png")
