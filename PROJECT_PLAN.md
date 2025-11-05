# AlphaAgent: Comprehensive RL Trading System
## Project Implementation Plan

---

## Executive Summary

**Project Name:** AlphaAgent
**Objective:** Design, build, train, and rigorously backtest two distinct Reinforcement Learning trading agents
**Current Phase:** Phase 1 - Foundation (Weeks 1-3)
**Approach:** Research-focused, modular architecture optimized for experimentation

---

## Project Objective

To design, build, train, and rigorously backtest two distinct Reinforcement Learning agents:

1. **"TimingAgent" (Single-Asset)**: A discrete-action agent designed to learn an optimal policy for timing entries and exits in a single, liquid asset, demonstrating core value-based RL methods.

2. **"PortfolioAgent" (Multi-Asset)**: A continuous-action agent designed to learn an optimal policy for dynamic asset allocation across a defined universe of assets, demonstrating advanced policy-gradient and actor-critic methods.

This project will demonstrate key RL concepts: the exploration-exploitation tradeoff, reward engineering for risk-adjusted returns, state representation for complex market dynamics, and policy learning under non-stationarity.

---

## Project Phases Overview

### Phase 1: Foundation - Environment, Data, & Tooling (Weeks 1-3) ⬅️ CURRENT FOCUS
**Objective:** Build the bedrock infrastructure for data, features, environments, and backtesting.

### Phase 2: "TimingAgent" - Single-Asset Trader (Weeks 4-6)
**Objective:** Build and train a discrete-action (Long/Flat/Short) agent using DQN.

### Phase 3: "PortfolioAgent" - Multi-Asset Allocator (Weeks 7-9)
**Objective:** Build and train a continuous-action agent that outputs portfolio weights using PPO/SAC.

### Phase 4: Backtesting & Evaluation (Weeks 10-12)
**Objective:** Rigorously evaluate both agents on unseen test data with comprehensive performance analysis.

---

## Theoretical Foundation: Markov Decision Process (MDP)

We frame trading as a Markov Decision Process (MDP). Unlike supervised learning (which just predicts price), RL makes sequential decisions where each action (trade) affects the future state (portfolio) and cumulative rewards (PnL).

- **State ($S$)**: A snapshot of all relevant information (market data, indicators, current portfolio holdings, cash)
- **Action ($A$)**: The decision the agent makes (Buy, Sell, Hold, or specific portfolio weights)
- **Reward ($R$)**: The feedback the agent receives after an action (e.g., change in portfolio value, risk-adjusted return)
- **Policy ($\pi$)**: The agent's strategy, a function that maps states to actions ($\pi(A|S)$)

---

## Phase 1: Foundation - Detailed Implementation Plan

> **Note:** This is the bedrock. The quality of the data and the realism of the simulation environment will directly determine the success or failure of the agents.

### Project Structure

```
QuantWorkshop/
├── config/
│   ├── data_config.yaml              # Data sources, date ranges, assets
│   ├── feature_config.yaml           # Feature engineering parameters
│   ├── cv_config.yaml                # Walk-forward CV settings
│   └── env_config.yaml               # Environment parameters (fees, slippage)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py            # yfinance data download & caching
│   │   ├── preprocessing.py          # Cleaning, alignment, corporate actions
│   │   ├── features.py               # Technical indicators, feature store
│   │   └── splitter.py               # Walk-forward CV implementation
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base_env.py               # Abstract base trading environment
│   │   ├── timing_env.py             # TimingAgent environment (Phase 2)
│   │   └── portfolio_env.py          # PortfolioAgent environment (Phase 3)
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py                 # Event-driven backtest harness
│   │   ├── metrics.py                # Performance analytics
│   │   └── baselines.py              # Simple baseline strategies
│   ├── agents/
│   │   ├── __init__.py
│   │   └── (future: DQN, PPO, SAC implementations)
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # YAML config loader
│       ├── scaling.py                # Scaler utilities (fit/transform)
│       └── visualizations.py         # Plotting helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data quality checks
│   ├── 02_feature_analysis.ipynb     # Feature correlation, importance
│   ├── 03_cv_validation.ipynb        # Validate walk-forward splits
│   └── 04_environment_testing.ipynb  # Test environment mechanics
├── tests/
│   ├── __init__.py
│   ├── test_data_acquisition.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_splitter.py
│   └── test_environment.py
├── data/
│   ├── raw/                          # Downloaded OHLCV data
│   ├── processed/                    # Cleaned and aligned data
│   └── features/                     # Computed features
├── models/                           # Saved trained models (future)
├── results/                          # Backtest results and reports (future)
├── logs/                             # Training logs (future)
├── requirements.txt
├── setup.py                          # Package installation
├── README.md
└── PROJECT_PLAN.md                   # This document
```

---

## Phase 1: Step-by-Step Implementation

### Step 1.1: Environment Setup

**Objective:** Create a reproducible Python environment with all required dependencies.

**Tasks:**
1. Create a conda environment or venv
2. Install core dependencies
3. Verify installations

**Dependencies:**

```
# Core Data & ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
polars>=0.18.0                # Fast data I/O

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Reinforcement Learning
gymnasium>=0.29.0             # Modern OpenAI Gym
stable-baselines3>=2.1.0      # SOTA RL algorithms (DQN, PPO, SAC)

# Finance & Data
yfinance>=0.2.28              # Market data acquisition
pandas-ta>=0.3.14b            # Technical indicators
quantstats>=0.0.62            # Performance analytics
exchange-calendars>=4.2.0     # Market schedules

# Configuration & Logging
pyyaml>=6.0
python-dotenv>=1.0.0
mlflow>=2.7.0                 # Experiment tracking (optional for Phase 1)

# Development & Testing
pytest>=7.4.0
pytest-cov>=4.1.0
jupyter>=1.0.0
ipykernel>=6.25.0
black>=23.7.0                 # Code formatting
flake8>=6.1.0                 # Linting
```

**Success Criteria:**
- ✓ Environment created and activated
- ✓ All packages install without errors
- ✓ `import gymnasium`, `import stable_baselines3`, `import yfinance` work

---

### Step 1.2: Configuration Framework

**Objective:** Create YAML-based configuration system for reproducibility and experimentation.

**config/data_config.yaml:**
```yaml
# Asset Universe
timing_agent:
  ticker: "SPY"
  description: "S&P 500 ETF - Highly liquid"

portfolio_agent:
  tickers:
    - "SPY"   # S&P 500 (Equities)
    - "QQQ"   # Nasdaq 100 (Tech)
    - "GLD"   # Gold (Commodities)
    - "TLT"   # 20+ Year Treasury (Bonds)

benchmark:
  ticker: "SPY"

# Data Parameters
data_source: "yfinance"
frequency: "1d"               # Daily data
start_date: "2000-01-01"
end_date: "2024-12-31"

# Data Quality
min_trading_days: 252         # Minimum days for valid data
handle_missing: "forward_fill"
max_fill_limit: 5             # Max consecutive days to forward fill

# Corporate Actions
adjust_splits: true
adjust_dividends: true
```

**config/feature_config.yaml:**
```yaml
# Feature Engineering Parameters

price_features:
  log_returns:
    windows: [1, 3, 5, 10, 20]

momentum_indicators:
  rsi:
    period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9
  stochastic:
    k_period: 14
    d_period: 3

trend_indicators:
  sma:
    windows: [50, 200]
  ema:
    windows: [12, 26]

volatility_indicators:
  bollinger_bands:
    period: 20
    std: 2
  atr:
    period: 14

volume_indicators:
  obv: true
  volume_sma:
    period: 20

# Feature Normalization
normalize: true
normalization_method: "standard"  # or "minmax"

# Feature Selection (optional)
feature_selection:
  enabled: false
  method: "mutual_info"
  top_k: 30
```

**config/cv_config.yaml:**
```yaml
# Walk-Forward Cross-Validation Configuration

method: "walk_forward"        # or "simple_split"

# Simple Split Parameters (fallback)
simple_split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

# Walk-Forward Parameters
walk_forward:
  # Training window: 3 years of data
  train_window_days: 1095

  # Validation window: 6 months
  val_window_days: 180

  # Step size: retrain every 3 months
  step_days: 90

  # Purging: remove N days after training set to prevent label leakage
  purge_days: 5

  # Embargo: remove N days before validation set
  embargo_days: 5

# Test Set (Hold-out)
test_set:
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  description: "Unseen data for final evaluation"

# Reproducibility
random_seed: 42
```

**config/env_config.yaml:**
```yaml
# Trading Environment Parameters

# Transaction Costs
transaction_costs:
  commission_bps: 10          # 10 basis points (0.1%)
  slippage_bps: 5             # 5 basis points (0.05%)

# Portfolio Parameters
initial_cash: 100000          # $100,000 starting capital
allow_shorting: true
max_leverage: 1.0             # 1x leverage (no margin)

# Reward Engineering
reward_type: "sharpe"         # Options: "pnl", "sharpe", "sortino", "drawdown_aware"
risk_free_rate: 0.02          # 2% annual
lookback_window: 252          # Days for Sharpe calculation

# Episode Parameters
max_steps: null               # null = full dataset length
done_on_drawdown: false
max_drawdown_threshold: 0.5   # Stop episode at 50% drawdown (if enabled)

# Rebalancing (PortfolioAgent)
rebalancing:
  no_trade_band: 0.005        # 0.5% - no rebalance if weight change < threshold
  turnover_penalty: 0.0001    # Penalty coefficient for high turnover
```

**Implementation:**
```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage YAML configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)

    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a specific config file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files."""
        configs = {}
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            configs[config_name] = self.load(config_name)
        return configs
```

---

### Step 1.3: Data Acquisition Module

**Objective:** Download, cache, and validate OHLCV data for all assets.

**Implementation: src/data/acquisition.py**

```python
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
        """Download OHLCV data for a single ticker."""

        # Check cache
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            logger.info(f"Loading {ticker} from cache")
            return pd.read_parquet(cache_file)

        # Download from yfinance
        logger.info(f"Downloading {ticker} from {start_date} to {end_date}")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,  # Adjust for splits and dividends
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # Standardize column names
        data.columns = [col.lower() for col in data.columns]

        # Add ticker column
        data['ticker'] = ticker

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
        """Download data for multiple tickers and align dates."""

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

        return combined

    def _validate_data(self, data: pd.DataFrame, ticker: str):
        """Validate downloaded data for quality issues."""

        # Check for missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        if missing_pct.any() > 10:
            logger.warning(f"{ticker}: High missing data percentage")
            logger.warning(missing_pct[missing_pct > 0])

        # Check for price anomalies
        if (data['close'] <= 0).any():
            logger.error(f"{ticker}: Found non-positive prices!")

        # Check for volume anomalies
        if (data['volume'] < 0).any():
            logger.error(f"{ticker}: Found negative volume!")

        # Check data frequency
        date_diffs = data.index.to_series().diff()
        max_gap = date_diffs.max().days
        if max_gap > 5:
            logger.warning(f"{ticker}: Found gap of {max_gap} days in data")
```

**Success Criteria:**
- ✓ Downloads data for all specified tickers
- ✓ Handles corporate actions (splits, dividends)
- ✓ Caches data locally for faster reloading
- ✓ Validates data quality (no missing critical values)

---

### Step 1.4: Data Preprocessing Module

**Objective:** Clean, align, and prepare data for feature engineering.

**Implementation: src/data/preprocessing.py**

```python
"""
Data preprocessing module.
Handles cleaning, alignment, and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clean and preprocess market data."""

    def __init__(self, config: dict):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""

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
        """Remove duplicate rows."""

        before = len(data)
        if 'ticker' in data.columns:
            data = data.drop_duplicates(subset=['ticker', data.index.name or 'date'])
        else:
            data = data[~data.index.duplicated(keep='first')]

        after = len(data)
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")

        return data

    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to config."""

        method = self.config.get('handle_missing', 'forward_fill')
        max_fill = self.config.get('max_fill_limit', 5)

        # Check missing before
        missing_before = data.isnull().sum().sum()

        if method == 'forward_fill':
            data = data.fillna(method='ffill', limit=max_fill)

        elif method == 'drop':
            data = data.dropna()

        # Check missing after
        missing_after = data.isnull().sum().sum()

        if missing_after > 0:
            logger.warning(f"Still have {missing_after} missing values after imputation")
            # Drop remaining missing
            data = data.dropna()

        logger.info(f"Handled {missing_before - missing_after} missing values")

        return data

    def _align_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align dates across multiple tickers (inner join)."""

        if 'ticker' not in data.columns:
            return data  # Single ticker, no alignment needed

        # Pivot to wide format
        tickers = data['ticker'].unique()

        # For each OHLCV column, create separate dataframe
        aligned_dfs = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                pivot = data.pivot(columns='ticker', values=col)
                # Inner join - only keep dates present for ALL tickers
                aligned_dfs[col] = pivot.dropna()

        # Stack back to long format
        aligned_data = []
        common_dates = aligned_dfs['close'].index

        for ticker in tickers:
            ticker_data = pd.DataFrame({
                col: aligned_dfs[col][ticker]
                for col in aligned_dfs.keys()
            }, index=common_dates)
            ticker_data['ticker'] = ticker
            aligned_data.append(ticker_data)

        result = pd.concat(aligned_data, axis=0)

        logger.info(f"Aligned to {len(common_dates)} common dates across {len(tickers)} tickers")

        return result

    def _validate_processed(self, data: pd.DataFrame):
        """Final validation of processed data."""

        # Check minimum trading days
        min_days = self.config.get('min_trading_days', 252)
        if len(data) < min_days:
            logger.error(f"Insufficient data: {len(data)} < {min_days} days")

        # Check no missing values
        if data.isnull().sum().sum() > 0:
            logger.error("Processed data still contains missing values!")

        # Check no infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).sum().sum() > 0:
            logger.error("Processed data contains infinite values!")
```

---

### Step 1.5: Feature Engineering Module

**Objective:** Create deterministic, "as-of" timestamped features. State representation must be stationary and informative.

**Implementation: src/data/features.py**

```python
"""
Feature engineering module.
Creates technical indicators and derived features for RL agents.
All features are deterministic and "as-of" to prevent look-ahead bias.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Generate technical indicators and features for trading."""

    def __init__(self, config: dict):
        self.config = config

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline."""

        logger.info("Starting feature engineering")

        # Work on copy
        df = data.copy()

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

        # Drop NaN rows created by indicator calculations
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        logger.info(f"Feature engineering complete: {final_len} rows ({initial_len - final_len} dropped due to indicator lag)")
        logger.info(f"Total features: {len(df.columns)}")

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns over multiple windows."""

        windows = self.config['price_features']['log_returns']['windows']

        for window in windows:
            # Log returns: log(P_t / P_(t-n))
            df[f'return_{window}d'] = np.log(df['close'] / df['close'].shift(window))

        logger.info(f"Added {len(windows)} return features")
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators."""

        config = self.config['momentum_indicators']

        # RSI
        if 'rsi' in config:
            period = config['rsi']['period']
            df['rsi'] = ta.rsi(df['close'], length=period)

        # MACD
        if 'macd' in config:
            macd_config = config['macd']
            macd = ta.macd(
                df['close'],
                fast=macd_config['fast'],
                slow=macd_config['slow'],
                signal=macd_config['signal']
            )
            df = df.join(macd)

        # Stochastic Oscillator
        if 'stochastic' in config:
            stoch_config = config['stochastic']
            stoch = ta.stoch(
                df['high'],
                df['low'],
                df['close'],
                k=stoch_config['k_period'],
                d=stoch_config['d_period']
            )
            df = df.join(stoch)

        logger.info("Added momentum indicators")
        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""

        config = self.config['trend_indicators']

        # Simple Moving Averages
        if 'sma' in config:
            for window in config['sma']['windows']:
                df[f'sma_{window}'] = ta.sma(df['close'], length=window)
                # Price relative to SMA
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1

        # SMA Crossover signal
        if 'sma' in config and len(config['sma']['windows']) >= 2:
            windows = sorted(config['sma']['windows'])
            fast_sma = f'sma_{windows[0]}'
            slow_sma = f'sma_{windows[1]}'
            df['sma_crossover'] = (df[fast_sma] > df[slow_sma]).astype(int)

        # Exponential Moving Averages
        if 'ema' in config:
            for window in config['ema']['windows']:
                df[f'ema_{window}'] = ta.ema(df['close'], length=window)

        logger.info("Added trend indicators")
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""

        config = self.config['volatility_indicators']

        # Bollinger Bands
        if 'bollinger_bands' in config:
            bb_config = config['bollinger_bands']
            bbands = ta.bbands(
                df['close'],
                length=bb_config['period'],
                std=bb_config['std']
            )
            df = df.join(bbands)

            # %B indicator (position within bands)
            df['bb_percent'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

            # Bandwidth
            df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']

        # Average True Range (ATR)
        if 'atr' in config:
            period = config['atr']['period']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=period)
            # Normalized ATR
            df['atr_pct'] = df['atr'] / df['close']

        logger.info("Added volatility indicators")
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""

        config = self.config['volume_indicators']

        # On-Balance Volume
        if config.get('obv', False):
            df['obv'] = ta.obv(df['close'], df['volume'])
            # Normalize OBV
            df['obv_norm'] = (df['obv'] - df['obv'].rolling(50).mean()) / df['obv'].rolling(50).std()

        # Volume SMA
        if 'volume_sma' in config:
            period = config['volume_sma']['period']
            df['volume_sma'] = ta.sma(df['volume'], length=period)
            # Volume relative to SMA
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        logger.info("Added volume indicators")
        return df

    def _add_portfolio_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder columns for portfolio context."""

        # These will be updated in real-time by the environment
        df['position'] = 0.0          # Current position: -1 (short), 0 (flat), +1 (long)
        df['cash_pct'] = 1.0          # Cash as % of portfolio
        df['unrealized_pnl'] = 0.0    # Unrealized P&L
        df['bars_since_trade'] = 0    # Time since last trade

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude OHLCV and ticker)."""

        exclude = ['open', 'high', 'low', 'close', 'volume', 'ticker']
        features = [col for col in df.columns if col not in exclude]

        return features
```

---

### Step 1.6: Walk-Forward Cross-Validation Module

**Objective:** Implement purged and embargoed walk-forward cross-validation to simulate realistic retraining schedules.

**Implementation: src/data/splitter.py**

```python
"""
Data splitting module for walk-forward cross-validation.
Implements purged and embargoed splits to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

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
        self.config = config['walk_forward']
        self.test_config = config['test_set']

    def split(self, data: pd.DataFrame) -> List[Split]:
        """
        Generate walk-forward splits.

        Returns:
            List of Split objects, each containing train/val date ranges
        """

        # Get date range (excluding test set)
        test_start = pd.Timestamp(self.test_config['start_date'])
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
        """Get train and validation data for a specific split."""

        train_data = data[(data.index >= split.train_start) & (data.index <= split.train_end)]
        val_data = data[(data.index >= split.val_start) & (data.index <= split.val_end)]

        return train_data, val_data

    def get_test_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get held-out test set."""

        test_start = pd.Timestamp(self.test_config['start_date'])
        test_end = pd.Timestamp(self.test_config['end_date'])

        test_data = data[(data.index >= test_start) & (data.index <= test_end)]

        logger.info(f"Test set: {len(test_data)} samples from {test_start.date()} to {test_end.date()}")

        return test_data

    def summary(self, splits: List[Split]) -> pd.DataFrame:
        """Generate summary table of all splits."""

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
        self.config = config['simple_split']
        self.test_config = config['test_set']

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Simple train/val/test split."""

        # Exclude test set
        test_start = pd.Timestamp(self.test_config['start_date'])
        available_data = data[data.index < test_start]

        n = len(available_data)
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']

        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))

        train_data = available_data.iloc[:train_end_idx]
        val_data = available_data.iloc[train_end_idx:val_end_idx]
        test_data = data[data.index >= test_start]

        logger.info(f"Simple split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        return train_data, val_data, test_data
```

---

### Step 1.7: Data Scaling Module

**Objective:** Implement proper scaling that prevents information leakage from future data.

**Implementation: src/utils/scaling.py**

```python
"""
Data scaling utilities with proper train/test separation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path
from typing import List, Optional, Tuple
import logging

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
        """Fit scaler on training data ONLY."""

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
        """Transform data using fitted scaler."""

        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        # Create copy
        scaled_data = data.copy()

        # Scale selected columns
        scaled_data[self.feature_columns] = self.scaler.transform(data[self.feature_columns])

        return scaled_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (use only on training data)."""

        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale."""

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
        """Save fitted scaler to disk."""

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
        """Load fitted scaler from disk."""

        loaded = joblib.load(path)

        self.scaler = loaded['scaler']
        self.method = loaded['method']
        self.feature_columns = loaded['feature_columns']
        self.is_fitted = True

        logger.info(f"Loaded scaler from {path}")

        return self
```

---

### Step 1.8: Base Trading Environment

**Objective:** Create abstract base class for trading environments implementing gym.Env interface.

**Implementation: src/environments/base_env.py**

```python
"""
Base trading environment class.
Implements common functionality for both TimingAgent and PortfolioAgent.
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseTradingEnv(gym.Env, ABC):
    """
    Abstract base class for trading environments.

    Implements the OpenAI Gym interface and common trading logic.
    Subclasses must implement: reset(), step(), _calculate_reward()
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict,
        features: list
    ):
        """
        Args:
            data: Preprocessed and feature-engineered DataFrame
            config: Environment configuration dict
            features: List of feature column names to use as state
        """
        super().__init__()

        self.data = data.reset_index(drop=False)
        self.config = config
        self.features = features

        # Extract config parameters
        self.initial_cash = config['initial_cash']
        self.commission_bps = config['transaction_costs']['commission_bps']
        self.slippage_bps = config['transaction_costs']['slippage_bps']
        self.max_steps = config.get('max_steps', len(data))

        # State tracking
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash

        # History tracking
        self.portfolio_history = []
        self.trade_history = []
        self.reward_history = []

        # Define spaces (to be overridden by subclasses)
        self.observation_space = None
        self.action_space = None

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        pass

    @abstractmethod
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        pass

    def _get_current_price(self) -> float:
        """Get current close price."""
        return self.data.loc[self.current_step, 'close']

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""

        # Extract feature values for current step
        obs = self.data.loc[self.current_step, self.features].values

        return obs.astype(np.float32)

    def _apply_transaction_costs(self, trade_value: float) -> float:
        """
        Calculate transaction costs (commission + slippage).

        Args:
            trade_value: Absolute dollar value of trade

        Returns:
            Total transaction cost
        """
        commission = trade_value * (self.commission_bps / 10000)
        slippage = trade_value * (self.slippage_bps / 10000)

        return commission + slippage

    def _is_done(self) -> bool:
        """Check if episode is complete."""

        # End if we've reached the end of data
        if self.current_step >= len(self.data) - 1:
            return True

        # End if max steps reached
        if self.max_steps and self.current_step >= self.max_steps:
            return True

        # End if drawdown threshold breached (if enabled)
        if self.config.get('done_on_drawdown', False):
            max_value = max(self.portfolio_history) if self.portfolio_history else self.initial_cash
            drawdown = (max_value - self.portfolio_value) / max_value
            if drawdown >= self.config.get('max_drawdown_threshold', 0.5):
                logger.warning(f"Episode ended: Max drawdown {drawdown:.1%} reached")
                return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current step."""

        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_trades': len(self.trade_history),
            'current_price': self._get_current_price()
        }

    def render(self, mode='human'):
        """Render environment state (for debugging)."""

        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Price: ${self._get_current_price():.2f}")

    def close(self):
        """Cleanup resources."""
        pass

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for completed episode."""

        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()

        # Calculate metrics
        total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        sharpe = self._calculate_sharpe(returns)
        max_dd = self._calculate_max_drawdown(portfolio_series)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(self.trade_history),
            'final_value': self.portfolio_value
        }

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""

        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        return sharpe

    @staticmethod
    def _calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""

        if len(portfolio_values) < 2:
            return 0.0

        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        return abs(drawdown.min())
```

---

### Step 1.9: Backtesting Infrastructure

**Objective:** Create event-driven backtest engine with performance analytics.

**Implementation: src/backtesting/engine.py**

```python
"""
Backtesting engine for evaluating trained agents.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Uses the same environment logic as training to ensure consistency.
    """

    def __init__(self, env, model=None):
        """
        Args:
            env: Trading environment instance
            model: Trained RL model (from stable-baselines3)
        """
        self.env = env
        self.model = model

    def run(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Run backtest on environment data.

        Args:
            deterministic: If True, agent exploits learned policy (no exploration)

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")

        obs, info = self.env.reset()
        done = False

        # Track results
        portfolio_values = []
        actions_taken = []
        rewards_received = []
        dates = []

        step = 0
        while not done:
            # Get action from model
            if self.model:
                action, _states = self.model.predict(obs, deterministic=deterministic)
            else:
                # Random action (for baseline testing)
                action = self.env.action_space.sample()

            # Execute action
            obs, reward, done, truncated, info = self.env.step(action)

            # Record
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(action)
            rewards_received.append(reward)
            dates.append(self.env.data.loc[self.env.current_step, 'date'])

            step += 1

            if done or truncated:
                break

        logger.info(f"Backtest complete: {step} steps")

        # Compile results
        results = {
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'rewards': rewards_received,
            'dates': dates,
            'env': self.env,
            'final_stats': self.env.get_episode_stats()
        }

        return results

    def run_baseline(self, strategy: str = 'buy_and_hold') -> Dict[str, Any]:
        """
        Run a baseline strategy for comparison.

        Args:
            strategy: 'buy_and_hold' or 'equal_weight'
        """
        logger.info(f"Running baseline: {strategy}")

        # Implementation depends on strategy
        if strategy == 'buy_and_hold':
            return self._buy_and_hold_baseline()
        elif strategy == 'equal_weight':
            return self._equal_weight_baseline()
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

    def _buy_and_hold_baseline(self) -> Dict[str, Any]:
        """Buy on first day, hold until end."""

        prices = self.env.data['close'].values
        initial_price = prices[0]

        # Buy with all capital
        shares = self.env.initial_cash / initial_price

        # Portfolio value over time
        portfolio_values = shares * prices

        dates = self.env.data['date'].values

        return {
            'portfolio_values': portfolio_values.tolist(),
            'strategy': 'Buy and Hold',
            'dates': dates.tolist(),
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - self.env.initial_cash) / self.env.initial_cash
        }

    def _equal_weight_baseline(self) -> Dict[str, Any]:
        """Equal weight rebalancing (for multi-asset)."""

        # This will be implemented in Phase 3 for PortfolioAgent
        raise NotImplementedError("Equal weight baseline for multi-asset not yet implemented")
```

**Implementation: src/backtesting/metrics.py**

```python
"""
Performance metrics and analytics for backtesting.
"""

import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Any
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Calculate and visualize trading performance metrics."""

    def __init__(self, results: Dict[str, Any], benchmark_results: Optional[Dict[str, Any]] = None):
        """
        Args:
            results: Backtest results from BacktestEngine
            benchmark_results: Optional benchmark strategy results
        """
        self.results = results
        self.benchmark_results = benchmark_results

        # Convert to returns
        self.returns = self._calculate_returns()
        if benchmark_results:
            self.benchmark_returns = self._calculate_benchmark_returns()
        else:
            self.benchmark_returns = None

    def _calculate_returns(self) -> pd.Series:
        """Calculate returns from portfolio values."""

        portfolio_values = pd.Series(self.results['portfolio_values'])
        returns = portfolio_values.pct_change().dropna()

        # Set dates as index if available
        if 'dates' in self.results:
            dates = pd.to_datetime(self.results['dates'][1:])  # Skip first (NaN return)
            returns.index = dates

        return returns

    def _calculate_benchmark_returns(self) -> pd.Series:
        """Calculate benchmark returns."""

        portfolio_values = pd.Series(self.benchmark_results['portfolio_values'])
        returns = portfolio_values.pct_change().dropna()

        if 'dates' in self.benchmark_results:
            dates = pd.to_datetime(self.benchmark_results['dates'][1:])
            returns.index = dates

        return returns

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, float]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary of key metrics
        """
        logger.info("Generating performance report")

        metrics = {}

        # Total Return
        initial_value = self.results['portfolio_values'][0]
        final_value = self.results['portfolio_values'][-1]
        metrics['total_return'] = (final_value - initial_value) / initial_value

        # Annualized Return
        n_days = len(self.returns)
        years = n_days / 252
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/years) - 1

        # Sharpe Ratio
        metrics['sharpe_ratio'] = qs.stats.sharpe(self.returns)

        # Sortino Ratio
        metrics['sortino_ratio'] = qs.stats.sortino(self.returns)

        # Calmar Ratio
        metrics['calmar_ratio'] = qs.stats.calmar(self.returns)

        # Max Drawdown
        metrics['max_drawdown'] = qs.stats.max_drawdown(self.returns)

        # Volatility
        metrics['volatility'] = qs.stats.volatility(self.returns)

        # CVaR (Conditional Value at Risk)
        metrics['cvar_95'] = qs.stats.cvar(self.returns)

        # Win Rate
        metrics['win_rate'] = (self.returns > 0).sum() / len(self.returns)

        # Number of trades
        metrics['total_trades'] = self.results['final_stats']['total_trades']

        # Benchmark comparison (if available)
        if self.benchmark_returns is not None:
            bench_total_return = (1 + self.benchmark_returns).prod() - 1
            metrics['alpha'] = metrics['total_return'] - bench_total_return
            metrics['benchmark_return'] = bench_total_return

        # Generate HTML report using quantstats
        if output_path:
            qs.reports.html(
                self.returns,
                self.benchmark_returns,
                output=output_path,
                title="RL Trading Agent Performance"
            )
            logger.info(f"HTML report saved to {output_path}")

        return metrics

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot portfolio value over time."""

        fig, ax = plt.subplots(figsize=(12, 6))

        dates = pd.to_datetime(self.results['dates'])
        portfolio_values = self.results['portfolio_values']

        ax.plot(dates, portfolio_values, label='Agent', linewidth=2)

        if self.benchmark_results:
            bench_dates = pd.to_datetime(self.benchmark_results['dates'])
            bench_values = self.benchmark_results['portfolio_values']
            ax.plot(bench_dates, bench_values, label='Benchmark', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print performance summary to console."""

        metrics = self.generate_report()

        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)

        print(f"\nReturns:")
        print(f"  Total Return:       {metrics['total_return']:>8.2%}")
        print(f"  Annualized Return:  {metrics['annualized_return']:>8.2%}")

        if 'alpha' in metrics:
            print(f"  Alpha:              {metrics['alpha']:>8.2%}")
            print(f"  Benchmark Return:   {metrics['benchmark_return']:>8.2%}")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>8.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
        print(f"  Volatility (Ann.):  {metrics['volatility']:>8.2%}")
        print(f"  CVaR (95%):         {metrics['cvar_95']:>8.2%}")

        print(f"\nTrading Statistics:")
        print(f"  Total Trades:       {metrics['total_trades']:>8.0f}")
        print(f"  Win Rate:           {metrics['win_rate']:>8.2%}")

        print("="*50 + "\n")
```

---

## Phase 1: Success Criteria

Phase 1 is complete when:

✅ **Environment Setup**
- Python environment created with all dependencies installed
- All imports work without errors

✅ **Configuration System**
- All YAML config files created and loadable
- Config loader utility working

✅ **Data Pipeline**
- Data successfully downloaded for all assets (SPY, QQQ, GLD, TLT)
- Corporate actions handled correctly
- Missing values properly imputed
- Data aligned across all tickers

✅ **Feature Engineering**
- All technical indicators calculated correctly
- Features are deterministic (reproducible)
- No look-ahead bias in feature calculation

✅ **Walk-Forward CV**
- Purged and embargoed splits generated correctly
- No overlap between train/val/test sets
- Splits validated in notebook

✅ **Data Scaling**
- Scaler fits only on training data
- Same scaler transforms val/test data
- Can save and load fitted scalers

✅ **Base Environment**
- BaseTradingEnv class implements gym.Env interface
- Transaction costs simulated correctly
- Portfolio tracking works
- Ready for agent implementations

✅ **Backtesting Infrastructure**
- Backtest engine can run episodes
- Baseline strategies implemented
- Performance metrics calculated
- Equity curves can be plotted

✅ **Testing & Validation**
- Unit tests pass for all modules
- No look-ahead bias detected
- Data pipeline reproducible
- All components integrated successfully

---

## Future Phases (Brief Overview)

### Phase 2: TimingAgent (Weeks 4-6)

**Objective:** Build and train discrete-action single-asset agent.

**Key Components:**
- `TimingEnv(BaseTradingEnv)`: Discrete action space (3 actions: Hold, Long, Short)
- DQN implementation using stable-baselines3
- Experience replay and target networks
- Reward engineering experiments (PnL vs Sharpe vs Drawdown-aware)
- Hyperparameter tuning on validation set
- Training loop with mlflow logging

**Deliverables:**
- Trained TimingAgent model
- Training curves and metrics
- Initial validation performance

---

### Phase 3: PortfolioAgent (Weeks 7-9)

**Objective:** Build and train continuous-action multi-asset agent.

**Key Components:**
- `PortfolioEnv(BaseTradingEnv)`: Continuous action space (portfolio weights)
- PPO/SAC implementation
- Action normalization (softmax for long-only)
- Rebalancing logic with no-trade bands
- Turnover penalty in reward
- Advanced reward engineering (Sharpe, risk-adjusted)

**Deliverables:**
- Trained PortfolioAgent model
- Comparison of PPO vs SAC
- Multi-asset allocation analysis

---

### Phase 4: Backtesting & Evaluation (Weeks 10-12)

**Objective:** Rigorous evaluation on unseen test set.

**Key Components:**
- Run both agents on held-out test set
- Generate quantstats tearsheets
- Compare against baselines (Buy&Hold, Equal Weight, HRP)
- Robustness testing:
  - Cost sensitivity analysis (1.5x, 2x costs)
  - Ablation studies (remove feature groups)
  - Factor loading analysis
  - Turnover analysis
- Statistical significance testing
- Final reporting and visualization

**Deliverables:**
- Comprehensive performance reports
- Test set results
- Robustness analysis
- Final presentation deck

---

## Key Design Principles

1. **No Look-Ahead Bias**: All features are "as-of" and deterministic
2. **Realistic Simulation**: Transaction costs and slippage on every trade
3. **Proper Data Splitting**: Walk-forward CV with purging and embargo
4. **Config-Driven**: All parameters in YAML for reproducibility
5. **Modular Architecture**: Clean separation of concerns
6. **Research-Focused**: Easy to experiment with different rewards, features, algorithms
7. **Comprehensive Testing**: Unit tests for all critical components
8. **Production-Ready Practices**: Logging, error handling, documentation

---

## Technology Stack Summary

**Core Python:** 3.9+
**Data:** pandas, numpy, polars
**ML/DL:** PyTorch, scikit-learn
**RL:** gymnasium, stable-baselines3
**Finance:** yfinance, pandas-ta, quantstats
**Config:** pyyaml
**Visualization:** matplotlib, seaborn, plotly (optional)
**Logging:** mlflow (optional for Phase 1)
**Testing:** pytest
**Development:** Jupyter, black, flake8

---

## Next Steps

1. ✅ **Create this PROJECT_PLAN.md** document
2. Set up directory structure
3. Create requirements.txt
4. Implement configuration system
5. Build data acquisition module
6. Implement feature engineering
7. Implement walk-forward CV
8. Create base environment
9. Build backtesting infrastructure
10. Write tests and validation notebooks
11. Validate entire Phase 1 pipeline
12. Move to Phase 2: TimingAgent

---

## Questions & Clarifications

If you have questions during implementation:
- Reward engineering: Start with simple PnL, then experiment with Sharpe
- Hyperparameters: Use stable-baselines3 defaults initially
- Features: Start with subset, expand based on importance
- Training time: DQN ~1-2 hours, PPO/SAC ~3-5 hours on CPU

---

**Document Version:** 1.0
**Last Updated:** 2025-11-05
**Status:** Phase 1 - In Progress
