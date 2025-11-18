"""
Data preprocessing module.
Handles cleaning, alignment, and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean and preprocess market data."""

    def __init__(self, config: dict):
        """
        Args:
            config: Data configuration dictionary
        """
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            data: Raw data DataFrame

        Returns:
            Cleaned and preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")

        # 1. Remove duplicates
        data = self._remove_duplicates(data)

        # 2. Handle missing values
        data = self._handle_missing(data)

        # 3. Align dates (for multi-ticker data)
        data = self._align_dates(data)

        # 4. Validate final data
        self._validate_processed(data)

        logger.info(f"Preprocessing complete: {len(data)} rows")

        return data

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame without duplicates
        """
        before = len(data)

        if 'ticker' in data.columns:
            # Multi-ticker data
            data = data.reset_index()
            data = data.drop_duplicates(subset=['date', 'ticker'], keep='first')
            data = data.set_index('date')
        else:
            # Single ticker data
            data = data[~data.index.duplicated(keep='first')]

        after = len(data)
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")

        return data

    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to config.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        method = self.config.get('handle_missing', 'forward_fill')
        max_fill = self.config.get('max_fill_limit', 5)

        # Check missing before
        missing_before = data.isnull().sum().sum()

        if missing_before > 0:
            logger.info(f"Found {missing_before} missing values")

        if method == 'forward_fill':
            # Forward fill with limit
            if 'ticker' in data.columns:
                # Group by ticker for multi-asset data
                data = data.groupby('ticker', group_keys=False).apply(lambda x: x.ffill(limit=max_fill))
            else:
                data = data.ffill(limit=max_fill)

        elif method == 'drop':
            data = data.dropna()

        # Check missing after
        missing_after = data.isnull().sum().sum()

        if missing_after > 0:
            logger.warning(f"Still have {missing_after} missing values after imputation")
            logger.warning("Dropping remaining rows with missing values")
            # Drop remaining missing
            data = data.dropna()

        logger.info(f"Handled {missing_before - data.isnull().sum().sum()} missing values")

        return data

    def _align_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align dates across multiple tickers (inner join).
        Only keeps dates present for ALL tickers.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with aligned dates
        """
        if 'ticker' not in data.columns:
            return data  # Single ticker, no alignment needed

        logger.info("Aligning dates across tickers")

        # Get unique tickers
        tickers = data['ticker'].unique()
        logger.info(f"Aligning {len(tickers)} tickers")

        # Reset index to have date as column
        data = data.reset_index()

        # For each OHLCV column, create separate dataframe
        aligned_dfs = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                # Pivot to wide format
                pivot = data.pivot(index='date', columns='ticker', values=col)
                # Inner join - only keep dates present for ALL tickers
                aligned_dfs[col] = pivot.dropna()

        # Get common dates
        common_dates = aligned_dfs['close'].index

        # Stack back to long format
        aligned_data = []
        for ticker in tickers:
            ticker_data = pd.DataFrame({
                col: aligned_dfs[col][ticker]
                for col in aligned_dfs.keys()
            }, index=common_dates)
            ticker_data['ticker'] = ticker
            aligned_data.append(ticker_data)

        result = pd.concat(aligned_data, axis=0)
        result.index.name = 'date'

        logger.info(f"Aligned to {len(common_dates)} common dates across {len(tickers)} tickers")

        return result

    def _validate_processed(self, data: pd.DataFrame):
        """
        Final validation of processed data.

        Args:
            data: Processed DataFrame

        Raises:
            ValueError: If data doesn't meet minimum requirements
        """
        # Check minimum trading days
        min_days = self.config.get('min_trading_days', 252)

        if 'ticker' in data.columns:
            # Check each ticker
            for ticker in data['ticker'].unique():
                ticker_data = data[data['ticker'] == ticker]
                if len(ticker_data) < min_days:
                    logger.warning(f"{ticker}: Only {len(ticker_data)} days (< {min_days})")
        else:
            if len(data) < min_days:
                logger.error(f"Insufficient data: {len(data)} < {min_days} days")
                raise ValueError(f"Insufficient data: {len(data)} days")

        # Check no missing values
        missing = data.isnull().sum().sum()
        if missing > 0:
            logger.error(f"Processed data still contains {missing} missing values!")
            raise ValueError("Data contains missing values after preprocessing")

        # Check no infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(data[numeric_cols]).sum().sum()
        if inf_count > 0:
            logger.error(f"Processed data contains {inf_count} infinite values!")
            raise ValueError("Data contains infinite values")

        logger.info("Data validation passed")
