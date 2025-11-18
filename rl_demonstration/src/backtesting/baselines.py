"""
Baseline trading strategies for comparison.
Enhanced for Phase 2 with proper transaction cost modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineStrategy:
    """Base class for baseline strategies."""

    def __init__(
        self,
        initial_cash: float = 100000,
        commission_bps: float = 10,
        slippage_bps: float = 5
    ):
        """
        Args:
            initial_cash: Starting capital
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
        """
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

        # State
        self.cash = initial_cash
        self.shares = 0.0
        self.portfolio_history = []
        self.trade_history = []

    def _apply_transaction_costs(self, trade_value: float) -> float:
        """Calculate transaction costs."""
        commission = trade_value * (self.commission_bps / 10000)
        slippage = trade_value * (self.slippage_bps / 10000)
        return commission + slippage

    def reset(self):
        """Reset strategy state."""
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_history = []
        self.trade_history = []

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': len(self.trade_history),
                'final_value': self.cash
            }

        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()

        # Total return
        total_return = (portfolio_series.iloc[-1] - self.initial_cash) / self.initial_cash

        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            risk_free_rate = 0.02 / 252  # Daily
            excess_returns = returns - risk_free_rate
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_history),
            'final_value': portfolio_series.iloc[-1]
        }


class BuyAndHold(BaselineStrategy):
    """Buy and Hold strategy - buy at start, hold until end."""

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute Buy and Hold strategy."""
        self.reset()

        if 'close' not in data.columns:
            raise ValueError("Data must have 'close' column")

        prices = data['close'].values

        # Buy at first price
        first_price = prices[0]
        shares_to_buy = (self.cash * 0.995) / first_price
        trade_value = shares_to_buy * first_price
        cost = self._apply_transaction_costs(trade_value)

        self.shares = shares_to_buy
        self.cash -= (trade_value + cost)

        self.trade_history.append({
            'step': 0,
            'action': 'buy',
            'shares': self.shares,
            'price': first_price,
            'cost': cost
        })

        # Track portfolio value
        for i, price in enumerate(prices):
            portfolio_value = self.cash + (self.shares * price)
            self.portfolio_history.append(portfolio_value)

        logger.info(
            f"Buy & Hold: Initial=${self.initial_cash:,.0f}, "
            f"Final=${self.portfolio_history[-1]:,.0f}"
        )

        return self.get_metrics()


class BaselineStrategies:
    """Legacy wrapper for backward compatibility."""

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


class SMAcrossover(BaselineStrategy):
    """Simple Moving Average Crossover Strategy."""

    def __init__(
        self,
        fast_window: int = 50,
        slow_window: int = 200,
        initial_cash: float = 100000,
        commission_bps: float = 10,
        slippage_bps: float = 5
    ):
        """
        Args:
            fast_window: Fast MA window
            slow_window: Slow MA window
            initial_cash: Starting capital
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
        """
        super().__init__(initial_cash, commission_bps, slippage_bps)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute SMA Crossover strategy."""
        self.reset()

        if 'close' not in data.columns:
            raise ValueError("Data must have 'close' column")

        # Calculate SMAs
        df = data.copy()
        df['sma_fast'] = df['close'].rolling(window=self.fast_window).mean()
        df['sma_slow'] = df['close'].rolling(window=self.slow_window).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1  # Long
        df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1  # Flat

        # Drop NaN rows
        df = df.dropna()

        position = 0  # Current position: 0=flat, 1=long

        for idx, row in df.iterrows():
            price = row['close']
            signal = row['signal']

            # Execute trades based on signal
            if signal == 1 and position == 0:
                # Buy signal
                shares_to_buy = (self.cash * 0.995) / price
                if shares_to_buy > 0:
                    trade_value = shares_to_buy * price
                    cost = self._apply_transaction_costs(trade_value)

                    self.shares = shares_to_buy
                    self.cash -= (trade_value + cost)
                    position = 1

                    self.trade_history.append({
                        'step': len(self.portfolio_history),
                        'action': 'buy',
                        'shares': self.shares,
                        'price': price,
                        'cost': cost
                    })

            elif signal == -1 and position == 1:
                # Sell signal
                trade_value = self.shares * price
                cost = self._apply_transaction_costs(trade_value)

                self.cash += (trade_value - cost)
                self.shares = 0
                position = 0

                self.trade_history.append({
                    'step': len(self.portfolio_history),
                    'action': 'sell',
                    'shares': 0,
                    'price': price,
                    'cost': cost
                })

            # Track portfolio value
            portfolio_value = self.cash + (self.shares * price)
            self.portfolio_history.append(portfolio_value)

        logger.info(
            f"SMA Crossover ({self.fast_window}/{self.slow_window}): "
            f"Initial=${self.initial_cash:,.0f}, "
            f"Final=${self.portfolio_history[-1]:,.0f}, "
            f"Trades={len(self.trade_history)}"
        )

        return self.get_metrics()


def compare_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run all baseline strategies and compare results.

    Args:
        data: DataFrame with 'close' column

    Returns:
        DataFrame with comparison metrics
    """
    strategies = {
        'Buy & Hold': BuyAndHold(),
        'SMA 50/200': SMAcrossover(fast_window=50, slow_window=200),
        'SMA 20/50': SMAcrossover(fast_window=20, slow_window=50),
    }

    results = []

    for name, strategy in strategies.items():
        metrics = strategy.run(data)
        metrics['strategy'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[['strategy', 'total_return', 'sharpe_ratio',
                              'max_drawdown', 'total_trades', 'final_value']]

    return results_df
