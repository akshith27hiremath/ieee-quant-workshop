"""
Data scaling utilities with proper train/test separation.
Prevents information leakage from future data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesScaler:
    """
    Scaler for time-series data that prevents lookahead bias.

    Key principle: Fit scaler ONLY on training data, then transform
    validation and test sets using the same fitted parameters.
    """

    def __init__(self, method: str = 'standard', feature_columns: Optional[List[str]] = None):
        """
        Args:
            method: 'standard' or 'minmax'
            feature_columns: List of columns to scale (if None, scales all numeric)
        """
        self.method = method
        self.feature_columns = feature_columns

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'TimeSeriesScaler':
        """
        Fit scaler on training data ONLY.

        Args:
            data: Training data DataFrame

        Returns:
            Self for method chaining
        """
        # Select columns to scale
        if self.feature_columns is None:
            # Scale all numeric columns
            self.feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Fit scaler
        self.scaler.fit(data[self.feature_columns])
        self.is_fitted = True

        logger.info(f"Fitted {self.method} scaler on {len(self.feature_columns)} features")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.

        Args:
            data: Data to transform

        Returns:
            Scaled DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        # Create copy
        scaled_data = data.copy()

        # Scale selected columns
        scaled_data[self.feature_columns] = self.scaler.transform(data[self.feature_columns])

        return scaled_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step (use only on training data).

        Args:
            data: Training data

        Returns:
            Scaled DataFrame
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.

        Args:
            data: Scaled data

        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")

        # Create copy
        original_data = data.copy()

        # Inverse transform
        original_data[self.feature_columns] = self.scaler.inverse_transform(
            data[self.feature_columns]
        )

        return original_data

    def save(self, path: str):
        """
        Save fitted scaler to disk.

        Args:
            path: File path to save scaler
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'scaler': self.scaler,
            'method': self.method,
            'feature_columns': self.feature_columns
        }, save_path)

        logger.info(f"Saved scaler to {path}")

    def load(self, path: str):
        """
        Load fitted scaler from disk.

        Args:
            path: File path to load scaler from

        Returns:
            Self for method chaining
        """
        loaded = joblib.load(path)

        self.scaler = loaded['scaler']
        self.method = loaded['method']
        self.feature_columns = loaded['feature_columns']
        self.is_fitted = True

        logger.info(f"Loaded scaler from {path}")

        return self
