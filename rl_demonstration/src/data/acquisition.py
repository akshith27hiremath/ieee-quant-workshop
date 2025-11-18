"""
Data acquisition module for downloading market data.
Handles caching, corporate actions, and data validation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Download and cache market data from Yahoo Finance."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If no data is downloaded
        """
        # Check cache
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            logger.info(f"Loading {ticker} from cache")
            return pd.read_parquet(cache_file)

        # Download from yfinance
        logger.info(f"Downloading {ticker} from {start_date} to {end_date}")

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,  # Adjust for splits and dividends
                progress=False
            )
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            raise

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # Handle multi-index columns (newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-index columns
            data.columns = data.columns.get_level_values(0)

        # Standardize column names
        data.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in data.columns]

        # Add ticker column
        data['ticker'] = ticker

        # Reset index to have date as column
        data = data.reset_index()
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        elif 'Date' in data.columns:
            data.rename(columns={'Date': 'date'}, inplace=True)
            data['date'] = pd.to_datetime(data['date'])

        # Set date as index
        data = data.set_index('date')

        # Validate data
        self._validate_data(data, ticker)

        # Save to cache
        data.to_parquet(cache_file)
        logger.info(f"Saved {ticker} to cache ({len(data)} rows)")

        return data

    def download_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download data for multiple tickers and align dates.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data for all tickers

        Raises:
            ValueError: If no data is downloaded for any ticker
        """
        dfs = []
        for ticker in tickers:
            try:
                df = self.download_ticker(ticker, start_date, end_date, use_cache)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
                continue

        if not dfs:
            raise ValueError("No data downloaded for any ticker")

        # Concatenate all dataframes
        combined = pd.concat(dfs, axis=0)

        logger.info(f"Downloaded {len(tickers)} tickers, total {len(combined)} rows")

        return combined

    def _validate_data(self, data: pd.DataFrame, ticker: str):
        """
        Validate downloaded data for quality issues.

        Args:
            data: DataFrame to validate
            ticker: Ticker symbol for logging
        """
        # Check for missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        if missing_pct.any() > 10:
            logger.warning(f"{ticker}: High missing data percentage")
            for col, pct in missing_pct[missing_pct > 0].items():
                logger.warning(f"  {col}: {pct:.2f}% missing")

        # Check for price anomalies
        if (data['close'] <= 0).any():
            logger.error(f"{ticker}: Found non-positive prices!")
            raise ValueError(f"{ticker} has invalid price data")

        # Check for volume anomalies
        if 'volume' in data.columns and (data['volume'] < 0).any():
            logger.error(f"{ticker}: Found negative volume!")
            raise ValueError(f"{ticker} has invalid volume data")

        # Check data frequency (gaps)
        date_diffs = data.index.to_series().diff()
        max_gap = date_diffs.max().days
        if max_gap > 5:
            logger.warning(f"{ticker}: Found gap of {max_gap} days in data")

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.

        Args:
            ticker: Optional ticker to clear. If None, clears all cache.
        """
        if ticker:
            pattern = f"{ticker}_*.parquet"
        else:
            pattern = "*.parquet"

        files = list(self.cache_dir.glob(pattern))
        for file in files:
            file.unlink()
            logger.info(f"Removed cache file: {file.name}")

        logger.info(f"Cleared {len(files)} cache file(s)")
