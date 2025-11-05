"""
Baseline trading strategies for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineStrategies:
    """Collection of simple baseline trading strategies."""

    def __init__(self, initial_cash: float = 100000):
        """
        Args:
            initial_cash: Starting capital
        """
        self.initial_cash = initial_cash

    def buy_and_hold(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        commission_bps: float = 10
    ) -> Dict[str, Any]:
        """
        Buy and hold strategy.

        Args:
            data: DataFrame with price data
            ticker: Optional ticker symbol (for multi-asset data)
            commission_bps: Commission in basis points

        Returns:
            Dictionary with strategy results
        """
        if ticker and 'ticker' in data.columns:
            data = data[data['ticker'] == ticker].copy()

        prices = data['close'].values
        dates = data.index if hasattr(data.index, 'name') else data['date'].values

        # Buy on first day
        initial_price = prices[0]
        commission = self.initial_cash * (commission_bps / 10000)
        available_cash = self.initial_cash - commission

        shares = available_cash / initial_price

        # Portfolio value over time
        portfolio_values = shares * prices

        # Final metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash

        logger.info(f"Buy & Hold: ${final_value:,.2f} ({total_return:.2%})")

        return {
            'strategy': 'Buy and Hold',
            'portfolio_values': portfolio_values.tolist(),
            'dates': dates.tolist(),
            'final_value': final_value,
            'total_return': total_return,
            'trades': 1
        }

    def sma_crossover(
        self,
        data: pd.DataFrame,
        fast_period: int = 50,
        slow_period: int = 200,
        commission_bps: float = 10
    ) -> Dict[str, Any]:
        """
        Simple Moving Average crossover strategy.

        Args:
            data: DataFrame with price data
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            commission_bps: Commission in basis points

        Returns:
            Dictionary with strategy results
        """
        df = data.copy()

        # Calculate SMAs
        df['sma_fast'] = df['close'].rolling(fast_period).mean()
        df['sma_slow'] = df['close'].rolling(slow_period).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1  # Long
        df.loc[df['sma_fast'] <= df['sma_slow'], 'signal'] = 0  # Flat

        # Drop NaN rows
        df = df.dropna()

        # Simulate trading
        cash = self.initial_cash
        shares = 0
        portfolio_values = []
        trades = 0

        for idx, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']

            # Execute trades
            if signal == 1 and shares == 0:
                # Buy
                commission = cash * (commission_bps / 10000)
                shares = (cash - commission) / current_price
                cash = 0
                trades += 1

            elif signal == 0 and shares > 0:
                # Sell
                sale_value = shares * current_price
                commission = sale_value * (commission_bps / 10000)
                cash = sale_value - commission
                shares = 0
                trades += 1

            # Portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append(portfolio_value)

        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash

        logger.info(f"SMA Crossover ({fast_period}/{slow_period}): ${final_value:,.2f} ({total_return:.2%}), Trades={trades}")

        return {
            'strategy': f'SMA Crossover ({fast_period}/{slow_period})',
            'portfolio_values': portfolio_values,
            'dates': df.index.tolist(),
            'final_value': final_value,
            'total_return': total_return,
            'trades': trades
        }

    def equal_weight(
        self,
        data: pd.DataFrame,
        tickers: list,
        rebalance_freq: str = 'Q',
        commission_bps: float = 10
    ) -> Dict[str, Any]:
        """
        Equal weight portfolio with periodic rebalancing.

        Args:
            data: DataFrame with multi-asset price data
            tickers: List of tickers to include
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
            commission_bps: Commission in basis points

        Returns:
            Dictionary with strategy results
        """
        # TODO: Implement in Phase 3 for multi-asset
        raise NotImplementedError("Equal weight strategy not yet implemented")
