"""
Correlation Heatmap
=====================
Computes and visualizes the correlation matrix for numeric
features in the dataset.

Usage:
    python eda/correlation_heatmap.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr(method=method)


def plot_heatmap(corr: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close()


def top_correlated_with_target(corr: pd.DataFrame, target: str, n: int = 5) -> pd.Series:
    if target not in corr.columns:
        raise ValueError(f"{target} not found in correlation matrix")
    ranked = corr[target].drop(target).abs().sort_values(ascending=False)
    print(f"\nTop {n} features correlated with {target}:")
    print(ranked.head(n))
    return ranked.head(n)


if __name__ == "__main__":
    df = load_data()
    corr = compute_correlation(df)
    plot_heatmap(corr, save_path="screenshots/correlation_heatmap.png")

    if "Survived" in corr.columns:
        top_correlated_with_target(corr, "Survived")
