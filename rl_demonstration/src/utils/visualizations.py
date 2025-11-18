"""
Visualization utilities for trading analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_price_series(
    data: pd.DataFrame,
    ticker: str = None,
    save_path: Optional[str] = None
):
    """
    Plot OHLC price series.

    Args:
        data: DataFrame with OHLC data
        ticker: Ticker symbol for title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price plot
    ax1.plot(data.index, data['close'], label='Close', linewidth=1.5)
    if 'sma_50' in data.columns:
        ax1.plot(data.index, data['sma_50'], label='SMA 50', alpha=0.7)
    if 'sma_200' in data.columns:
        ax1.plot(data.index, data['sma_200'], label='SMA 200', alpha=0.7)

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker if ticker else "Asset"} Price History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Volume plot
    ax2.bar(data.index, data['volume'], alpha=0.5, color='gray')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Price chart saved to {save_path}")

    plt.show()


def plot_feature_distributions(
    data: pd.DataFrame,
    features: List[str],
    n_cols: int = 3,
    save_path: Optional[str] = None
):
    """
    Plot distributions of multiple features.

    Args:
        data: DataFrame with features
        features: List of feature column names
        n_cols: Number of columns in subplot grid
        save_path: Optional path to save figure
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if feature in data.columns:
            data[feature].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature distributions saved to {save_path}")

    plt.show()


def plot_correlation_matrix(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix heatmap.

    Args:
        data: DataFrame with features
        features: Optional list of features to include
        save_path: Optional path to save figure
    """
    if features is None:
        # Use all numeric columns
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = data[features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )

    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")

    plt.show()


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None
):
    """
    Plot returns distribution with statistics.

    Args:
        returns: Series of returns
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    ax1.axvline(returns.median(), color='g', linestyle='--', label=f'Median: {returns.median():.4f}')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title} - Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Returns distribution saved to {save_path}")

    plt.show()


def plot_drawdown(
    portfolio_values: pd.Series,
    title: str = "Drawdown Analysis",
    save_path: Optional[str] = None
):
    """
    Plot portfolio value and drawdown.

    Args:
        portfolio_values: Series of portfolio values over time
        title: Plot title
        save_path: Optional path to save figure
    """
    # Calculate drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Portfolio value
    ax1.plot(portfolio_values.index, portfolio_values, linewidth=1.5)
    ax1.fill_between(portfolio_values.index, portfolio_values, running_max, alpha=0.3, color='green')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'{title} - Portfolio Value')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(drawdown.index, 0, drawdown, alpha=0.5, color='red')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.set_title(f'Maximum Drawdown: {abs(drawdown.min()):.2%}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Drawdown analysis saved to {save_path}")

    plt.show()


def plot_walk_forward_splits(
    splits: List[Any],
    save_path: Optional[str] = None
):
    """
    Visualize walk-forward cross-validation splits.

    Args:
        splits: List of Split objects from WalkForwardSplitter
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, len(splits) * 0.5 + 2))

    for idx, split in enumerate(splits):
        # Train period
        ax.barh(
            idx,
            (split.train_end - split.train_start).days,
            left=split.train_start.toordinal(),
            color='blue',
            alpha=0.6,
            label='Train' if idx == 0 else ''
        )

        # Validation period
        ax.barh(
            idx,
            (split.val_end - split.val_start).days,
            left=split.val_start.toordinal(),
            color='orange',
            alpha=0.6,
            label='Validation' if idx == 0 else ''
        )

    # Format x-axis as dates
    ax.set_xlabel('Date')
    ax.set_ylabel('Fold')
    ax.set_title('Walk-Forward Cross-Validation Splits')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

    # Convert ordinal back to dates for x-axis labels
    import matplotlib.dates as mdates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Walk-forward splits visualization saved to {save_path}")

    plt.show()
