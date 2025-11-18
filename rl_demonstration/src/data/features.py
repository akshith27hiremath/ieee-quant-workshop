"""
Feature engineering module.
Creates technical indicators and derived features for RL agents.
All features are deterministic and "as-of" to prevent look-ahead bias.
"""

import pandas as pd
import ta as ta_lib
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical indicators and features for trading."""

    def __init__(self, config: dict):
        """
        Args:
            config: Feature configuration dictionary
        """
        self.config = config

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            data: Preprocessed DataFrame with OHLCV data

        Returns:
            DataFrame with all features added
        """
        logger.info("Starting feature engineering")

        # Work on copy
        df = data.copy()

        # Handle multi-ticker data
        if 'ticker' in df.columns:
            # Apply features per ticker
            dfs = []
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker].copy()
                ticker_df = self._engineer_features_single(ticker_df)
                dfs.append(ticker_df)
            df = pd.concat(dfs, axis=0)
        else:
            # Single ticker
            df = self._engineer_features_single(df)

        # Drop NaN rows created by indicator calculations
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        logger.info(f"Feature engineering complete: {final_len} rows ({initial_len - final_len} dropped due to indicator lag)")
        logger.info(f"Total features: {len(df.columns)}")

        return df

    def _engineer_features_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for a single ticker.

        Args:
            df: Single-ticker DataFrame

        Returns:
            DataFrame with features
        """
        # 1. Price-based features
        df = self._add_returns(df)

        # 2. Momentum indicators
        df = self._add_momentum_indicators(df)

        # 3. Trend indicators
        df = self._add_trend_indicators(df)

        # 4. Volatility indicators
        df = self._add_volatility_indicators(df)

        # 5. Volume indicators
        df = self._add_volume_indicators(df)

        # 6. Portfolio context (will be updated at runtime by environment)
        df = self._add_portfolio_context(df)

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add log returns over multiple windows.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with return features
        """
        windows = self.config['price_features']['log_returns']['windows']

        for window in windows:
            # Log returns: log(P_t / P_(t-n))
            df[f'return_{window}d'] = np.log(df['close'] / df['close'].shift(window))

        logger.debug(f"Added {len(windows)} return features")
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with momentum indicators
        """
        config = self.config['momentum_indicators']

        # RSI
        if 'rsi' in config:
            period = config['rsi']['period']
            rsi_indicator = RSIIndicator(close=df['close'], window=period)
            df['rsi'] = rsi_indicator.rsi()
            # Normalize RSI to [-1, 1]
            df['rsi_norm'] = (df['rsi'] - 50) / 50

        # MACD
        if 'macd' in config:
            macd_config = config['macd']
            macd_indicator = MACD(
                close=df['close'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_diff'] = macd_indicator.macd_diff()
            # Normalize MACD
            df['macd_norm'] = df['macd'] / df['close']

        # Stochastic Oscillator
        if 'stochastic' in config:
            stoch_config = config['stochastic']
            stoch_indicator = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=stoch_config['k_period'],
                smooth_window=stoch_config['d_period']
            )
            df['stoch_k'] = stoch_indicator.stoch()
            df['stoch_d'] = stoch_indicator.stoch_signal()

        logger.debug("Added momentum indicators")
        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-following indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with trend indicators
        """
        config = self.config['trend_indicators']

        # Simple Moving Averages
        if 'sma' in config:
            for window in config['sma']['windows']:
                sma_indicator = SMAIndicator(close=df['close'], window=window)
                df[f'sma_{window}'] = sma_indicator.sma_indicator()
                # Price relative to SMA
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1

        # SMA Crossover signal
        if 'sma' in config and len(config['sma']['windows']) >= 2:
            windows = sorted(config['sma']['windows'])
            fast_sma = f'sma_{windows[0]}'
            slow_sma = f'sma_{windows[1]}'
            if fast_sma in df.columns and slow_sma in df.columns:
                df['sma_crossover'] = (df[fast_sma] > df[slow_sma]).astype(int)

        # Exponential Moving Averages
        if 'ema' in config:
            for window in config['ema']['windows']:
                ema_indicator = EMAIndicator(close=df['close'], window=window)
                df[f'ema_{window}'] = ema_indicator.ema_indicator()
                # Price relative to EMA
                df[f'price_to_ema_{window}'] = df['close'] / df[f'ema_{window}'] - 1

        logger.debug("Added trend indicators")
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with volatility indicators
        """
        config = self.config['volatility_indicators']

        # Bollinger Bands
        if 'bollinger_bands' in config:
            bb_config = config['bollinger_bands']
            bb_indicator = BollingerBands(
                close=df['close'],
                window=bb_config['period'],
                window_dev=bb_config['std']
            )
            df['bb_high'] = bb_indicator.bollinger_hband()
            df['bb_low'] = bb_indicator.bollinger_lband()
            df['bb_mid'] = bb_indicator.bollinger_mavg()

            # %B indicator (position within bands)
            df['bb_percent'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
            # Bandwidth
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

        # Average True Range (ATR)
        if 'atr' in config:
            period = config['atr']['period']
            atr_indicator = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            )
            df['atr'] = atr_indicator.average_true_range()
            # Normalized ATR (as % of price)
            df['atr_pct'] = df['atr'] / df['close']

        logger.debug("Added volatility indicators")
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with volume indicators
        """
        config = self.config['volume_indicators']

        # On-Balance Volume
        if config.get('obv', False):
            obv_indicator = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['obv'] = obv_indicator.on_balance_volume()
            # Normalize OBV using rolling statistics
            df['obv_sma'] = df['obv'].rolling(50).mean()
            df['obv_std'] = df['obv'].rolling(50).std()
            df['obv_norm'] = (df['obv'] - df['obv_sma']) / (df['obv_std'] + 1e-8)

        # Volume SMA
        if 'volume_sma' in config:
            period = config['volume_sma']['period']
            volume_sma_indicator = SMAIndicator(close=df['volume'], window=period)
            df['volume_sma'] = volume_sma_indicator.sma_indicator()
            # Volume relative to SMA
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)

        logger.debug("Added volume indicators")
        return df

    def _add_portfolio_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add placeholder columns for portfolio context.
        These will be updated in real-time by the environment.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with portfolio context columns
        """
        # These will be updated in real-time by the environment
        df['position'] = 0.0          # Current position: -1 (short), 0 (flat), +1 (long)
        df['cash_pct'] = 1.0          # Cash as % of portfolio
        df['unrealized_pnl'] = 0.0    # Unrealized P&L
        df['bars_since_trade'] = 0    # Time since last trade

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (exclude OHLCV and ticker).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude = ['open', 'high', 'low', 'close', 'volume', 'ticker', 'date']
        features = [col for col in df.columns if col not in exclude]

        return features

    def get_feature_importance_candidates(self) -> Dict[str, List[str]]:
        """
        Get feature groups for ablation studies.

        Returns:
            Dictionary mapping feature categories to column patterns
        """
        return {
            'returns': ['return_'],
            'momentum': ['rsi', 'macd', 'stoch', 'STOCHk', 'STOCHd'],
            'trend': ['sma_', 'ema_', 'price_to_', 'crossover'],
            'volatility': ['atr', 'bb_', 'BB'],
            'volume': ['obv', 'volume_'],
            'portfolio': ['position', 'cash_pct', 'unrealized_pnl', 'bars_since']
        }
