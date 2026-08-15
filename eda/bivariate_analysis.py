"""
Bivariate Analysis
===================
Scatterplots, countplots, and group-wise analysis between pairs
of features and the target variable.

Usage:
    python eda/bivariate_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None, save_path: str = None):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.7)
    plt.title(f"{x} vs {y}" + (f" by {hue}" if hue else ""))
    _finish(save_path)


def plot_grouped_countplot(df: pd.DataFrame, x: str, hue: str, save_path: str = None):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=x, hue=hue)
    plt.title(f"{x} vs {hue}")
    _finish(save_path)


def groupwise_summary(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    summary = df.groupby(group_col)[target_col].agg(["mean", "median", "count"])
    print(f"\n{target_col} summary grouped by {group_col}:")
    print(summary)
    return summary


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

    if {"Age", "Fare"}.issubset(df.columns):
        hue = "Survived" if "Survived" in df.columns else None
        plot_scatter(df, "Age", "Fare", hue=hue, save_path="screenshots/scatter_age_fare.png")

    if {"Pclass", "Survived"}.issubset(df.columns):
        plot_grouped_countplot(df, "Pclass", "Survived", save_path="screenshots/count_pclass_survived.png")
        groupwise_summary(df, "Pclass", "Survived")

    if {"Sex", "Survived"}.issubset(df.columns):
        plot_grouped_countplot(df, "Sex", "Survived", save_path="screenshots/count_sex_survived.png")
        groupwise_summary(df, "Sex", "Survived")
