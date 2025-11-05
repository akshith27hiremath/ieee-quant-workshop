"""
Data splitting module for walk-forward cross-validation.
Implements purged and embargoed splits to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Split:
    """Represents a single train/validation split."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    fold: int


class WalkForwardSplitter:
    """
    Implements walk-forward cross-validation with purging and embargo.

    This simulates a realistic retraining schedule where:
    1. Train on a fixed window of historical data
    2. Validate on the next period
    3. Step forward and repeat
    4. Apply purging (remove data after train to prevent leakage)
    5. Apply embargo (remove data before validation to prevent lookahead)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: CV configuration dictionary
        """
        self.config = config['walk_forward']
        self.test_config = config['test_set']

    def split(self, data: pd.DataFrame) -> List[Split]:
        """
        Generate walk-forward splits.

        Args:
            data: Full dataset

        Returns:
            List of Split objects, each containing train/val date ranges
        """
        # Get date range (excluding test set)
        test_start = pd.Timestamp(self.test_config['start_date'])

        # Get unique dates
        if 'ticker' in data.columns:
            # For multi-ticker, get dates from first ticker
            ticker = data['ticker'].unique()[0]
            available_data = data[data['ticker'] == ticker]
            available_data = available_data[available_data.index < test_start]
            dates = available_data.index.unique().sort_values()
        else:
            available_data = data[data.index < test_start]
            dates = available_data.index.unique().sort_values()

        train_window = self.config['train_window_days']
        val_window = self.config['val_window_days']
        step = self.config['step_days']
        purge = self.config['purge_days']
        embargo = self.config['embargo_days']

        splits = []
        fold = 0

        # Start from first possible training window
        current_start = dates[0]

        while True:
            # Define train period
            train_start = current_start
            train_end = train_start + pd.Timedelta(days=train_window)

            # Apply purging (remove N days after train set)
            purge_end = train_end + pd.Timedelta(days=purge)

            # Apply embargo (start val N days later)
            val_start = purge_end + pd.Timedelta(days=embargo)
            val_end = val_start + pd.Timedelta(days=val_window)

            # Check if we have enough data
            if val_end >= test_start:
                break  # Would overlap with test set

            # Find actual dates in data (market days only)
            train_start_actual = dates[dates >= train_start][0] if any(dates >= train_start) else None
            train_end_actual = dates[dates <= train_end][-1] if any(dates <= train_end) else None
            val_start_actual = dates[dates >= val_start][0] if any(dates >= val_start) else None
            val_end_actual = dates[dates <= val_end][-1] if any(dates <= val_end) else None

            if None not in [train_start_actual, train_end_actual, val_start_actual, val_end_actual]:
                split = Split(
                    train_start=train_start_actual,
                    train_end=train_end_actual,
                    val_start=val_start_actual,
                    val_end=val_end_actual,
                    fold=fold
                )
                splits.append(split)
                fold += 1

            # Step forward
            current_start = current_start + pd.Timedelta(days=step)

        logger.info(f"Generated {len(splits)} walk-forward splits")

        return splits

    def get_train_val_data(
        self,
        data: pd.DataFrame,
        split: Split
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and validation data for a specific split.

        Args:
            data: Full dataset
            split: Split object defining the date ranges

        Returns:
            Tuple of (train_data, val_data)
        """
        train_data = data[(data.index >= split.train_start) & (data.index <= split.train_end)]
        val_data = data[(data.index >= split.val_start) & (data.index <= split.val_end)]

        logger.info(f"Fold {split.fold}: Train={len(train_data)} rows, Val={len(val_data)} rows")

        return train_data, val_data

    def get_test_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get held-out test set.

        Args:
            data: Full dataset

        Returns:
            Test set DataFrame
        """
        test_start = pd.Timestamp(self.test_config['start_date'])
        test_end = pd.Timestamp(self.test_config['end_date'])

        test_data = data[(data.index >= test_start) & (data.index <= test_end)]

        logger.info(f"Test set: {len(test_data)} samples from {test_start.date()} to {test_end.date()}")

        return test_data

    def summary(self, splits: List[Split]) -> pd.DataFrame:
        """
        Generate summary table of all splits.

        Args:
            splits: List of Split objects

        Returns:
            DataFrame with split information
        """
        summary_data = []
        for split in splits:
            summary_data.append({
                'fold': split.fold,
                'train_start': split.train_start.date(),
                'train_end': split.train_end.date(),
                'val_start': split.val_start.date(),
                'val_end': split.val_end.date(),
                'train_days': (split.train_end - split.train_start).days,
                'val_days': (split.val_end - split.val_start).days,
                'gap_days': (split.val_start - split.train_end).days
            })

        return pd.DataFrame(summary_data)


class SimpleSplitter:
    """Simple time-series split as fallback."""

    def __init__(self, config: dict):
        """
        Args:
            config: CV configuration dictionary
        """
        self.config = config['simple_split']
        self.test_config = config['test_set']

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Simple train/val/test split.

        Args:
            data: Full dataset

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Exclude test set
        test_start = pd.Timestamp(self.test_config['start_date'])

        # Get unique dates
        if 'ticker' in data.columns:
            # For multi-ticker, get dates from first ticker
            ticker = data['ticker'].unique()[0]
            available_data = data[data['ticker'] == ticker]
            available_data = available_data[available_data.index < test_start]
        else:
            available_data = data[data.index < test_start]

        n = len(available_data.index.unique())
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']

        # Get date thresholds
        dates = sorted(available_data.index.unique())
        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))

        train_end_date = dates[train_end_idx - 1]
        val_end_date = dates[val_end_idx - 1]

        # Split data
        train_data = data[data.index <= train_end_date]
        val_data = data[(data.index > train_end_date) & (data.index <= val_end_date)]
        test_data = data[data.index >= test_start]

        logger.info(f"Simple split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        return train_data, val_data, test_data
