"""
Visualizations module for PriceWise AI.
Creates modern, publication-grade Matplotlib and Seaborn charts for Streamlit.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "axes.edgecolor": "#E2E8F0",
    "axes.linewidth": 1.2,
    "grid.color": "#EDF2F7",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], category: str) -> plt.Figure:
    """
    Renders comparative bar charts for R2, MAE, and RMSE across all 4 models.
    """
    models = list(metrics_dict.keys())
    r2_scores = [metrics_dict[m]["R2"] for m in models]
    mae_scores = [metrics_dict[m]["MAE"] for m in models]
    rmse_scores = [metrics_dict[m]["RMSE"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. R2 Score (Higher is better)
    ax0 = axes[0]
    bars0 = ax0.bar(models, r2_scores, color="#2563EB", alpha=0.85, width=0.55)
    ax0.set_title(f"R² Score (Higher is Better) - {category}", fontsize=12, fontweight="bold", pad=10)
    ax0.set_ylabel("R² Score", fontsize=10)
    ax0.set_ylim([min(0, min(r2_scores) - 0.1), 1.05])
    ax0.tick_params(axis="x", rotation=25)
    for bar in bars0:
        h = bar.get_height()
        ax0.annotate(f"{h:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -12),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. MAE (Lower is better)
    ax1 = axes[1]
    bars1 = ax1.bar(models, mae_scores, color="#10B981", alpha=0.85, width=0.55)
    ax1.set_title(f"Mean Absolute Error (MAE in ₹)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("MAE (₹)", fontsize=10)
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars1:
        h = bar.get_height()
        val_str = f"₹{h/1e5:.1f}L" if h >= 1e5 else f"₹{h/1e3:.1f}k" if h >= 1e3 else f"₹{h:.0f}"
        ax1.annotate(val_str,
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. RMSE (Lower is better)
    ax2 = axes[2]
    bars2 = ax2.bar(models, rmse_scores, color="#F59E0B", alpha=0.85, width=0.55)
    ax2.set_title(f"Root Mean Squared Error (RMSE in ₹)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("RMSE (₹)", fontsize=10)
    ax2.tick_params(axis="x", rotation=25)
    for bar in bars2:
        h = bar.get_height()
        val_str = f"₹{h/1e5:.1f}L" if h >= 1e5 else f"₹{h/1e3:.1f}k" if h >= 1e3 else f"₹{h:.0f}"
        ax2.annotate(val_str,
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: List[float],
    y_pred: List[float],
    model_name: str,
    category: str
) -> plt.Figure:
    """
    Renders actual vs predicted scatter plot with 45-degree reference line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())

    ax.scatter(y_t, y_p, alpha=0.6, color="#2563EB", edgecolors="none", s=40, label="Holdout Samples")
    ax.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=2, label="Ideal (Actual = Predicted)")

    ax.set_title(f"Actual vs Predicted Prices: {model_name} ({category})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Price (₹)", fontsize=11)
    ax.set_ylabel("Predicted Price (₹)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper left")

    # Format ticks to readable format
    ax.ticklabel_format(style="plain", axis="both")
    return fig


def plot_residuals(
    y_true: List[float],
    y_pred: List[float],
    model_name: str,
    category: str
) -> plt.Figure:
    """
    Renders residuals scatter plot (residuals vs predicted) and residual distribution.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    y_p = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter of residuals vs predicted
    ax0 = axes[0]
    ax0.scatter(y_p, residuals, alpha=0.55, color="#8B5CF6", edgecolors="none", s=35)
    ax0.axhline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    ax0.set_title(f"Residuals vs Predicted ({category})", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Predicted Price (₹)", fontsize=10)
    ax0.set_ylabel("Residual Error (Actual - Predicted) (₹)", fontsize=10)

    # Residuals distribution histogram
    ax1 = axes[1]
    sns.histplot(residuals, kde=True, ax=ax1, color="#10B981", bins=25, alpha=0.6)
    ax1.axvline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    ax1.set_title(f"Residual Distribution ({model_name})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Residual Error (₹)", fontsize=10)
    ax1.set_ylabel("Density / Count", fontsize=10)

    plt.tight_layout()
    return fig


def plot_feature_importance(importances: Dict[str, float], model_name: str, category: str) -> plt.Figure:
    """
    Renders horizontal bar chart of top feature importances or linear coefficients.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if not importances:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        return fig

    # Sort importances descending
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [x[0] for x in sorted_items][::-1]
    values = [x[1] for x in sorted_items][::-1]

    bars = ax.barh(features, values, color="#3B82F6", alpha=0.85, height=0.6)
    ax.set_title(f"Top 10 Feature Drivers: {model_name} ({category})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Importance / Absolute Weight", fontsize=10)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_price_distribution(prices: pd.Series, category: str) -> plt.Figure:
    """
    Renders distribution histogram and box plot for prices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram + KDE
    ax0 = axes[0]
    sns.histplot(prices, kde=True, ax=ax0, color="#2563EB", bins=30, alpha=0.6)
    ax0.axvline(prices.median(), color="#EF4444", linestyle="--", label=f"Median: ₹{prices.median():,.0f}")
    ax0.set_title(f"{category} Price Distribution", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Price (INR)", fontsize=10)
    ax0.legend()

    # Boxplot
    ax1 = axes[1]
    sns.boxplot(x=prices, ax=ax1, color="#60A5FA", fliersize=3)
    ax1.set_title(f"{category} Price Quartiles & Outliers", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Price (INR)", fontsize=10)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, category: str) -> plt.Figure:
    """
    Renders correlation heatmap for numerical features of the category.
    """
    num_df = df.select_dtypes(include=["int64", "float64"])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, square=True,
                linewidths=0.5, linecolor="#E2E8F0")
    ax.set_title(f"Numerical Correlation Heatmap: {category}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig
