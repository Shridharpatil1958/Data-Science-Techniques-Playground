"""
Multivariate Analysis
=======================
Pairplots and multi-feature relationship analysis.

Usage:
    python eda/multivariate_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_pairplot(df: pd.DataFrame, columns: list, hue: str = None, save_path: str = None):
    subset = df[columns + ([hue] if hue and hue not in columns else [])].dropna()
    grid = sns.pairplot(subset, hue=hue, diag_kind="kde")
    grid.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    if save_path:
        grid.savefig(save_path, bbox_inches="tight")
        print(f"Saved pairplot to {save_path}")
    else:
        plt.show()
    plt.close("all")


def plot_facet_relationship(df: pd.DataFrame, x: str, y: str, row: str, col: str, save_path: str = None):
    grid = sns.FacetGrid(df, row=row, col=col, margin_titles=True)
    grid.map_dataframe(sns.scatterplot, x=x, y=y, alpha=0.7)
    if save_path:
        grid.savefig(save_path, bbox_inches="tight")
        print(f"Saved facet grid to {save_path}")
    else:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    df = load_data()

    numeric_cols = [c for c in ["Age", "Fare", "Pclass"] if c in df.columns]
    hue = "Survived" if "Survived" in df.columns else None

    if len(numeric_cols) >= 2:
        plot_pairplot(df, numeric_cols, hue=hue, save_path="screenshots/pairplot.png")

    if {"Age", "Fare", "Sex", "Pclass"}.issubset(df.columns):
        plot_facet_relationship(df, "Age", "Fare", row="Sex", col="Pclass", save_path="screenshots/facet_age_fare.png")
