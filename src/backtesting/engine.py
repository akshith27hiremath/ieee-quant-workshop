"""
Backtesting engine for evaluating trained agents.
Event-driven design ensures consistency with training environment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
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
            model: Trained RL model (from stable-baselines3), optional
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

            # Get date if available
            if hasattr(self.env, 'data') and 'date' in self.env.data.columns:
                dates.append(self.env.data.loc[self.env.current_step, 'date'])
            else:
                dates.append(step)

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

        Returns:
            Dictionary with baseline results
        """
        logger.info(f"Running baseline: {strategy}")

        if strategy == 'buy_and_hold':
            return self._buy_and_hold_baseline()
        elif strategy == 'equal_weight':
            return self._equal_weight_baseline()
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

    def _buy_and_hold_baseline(self) -> Dict[str, Any]:
        """
        Buy and hold baseline strategy.
        Buy on first day, hold until end.

        Returns:
            Dictionary with baseline results
        """
        prices = self.env.data['close'].values
        initial_price = prices[0]

        # Buy with all capital (minus transaction costs)
        transaction_cost = self.env.initial_cash * (self.env.commission_bps / 10000)
        available_cash = self.env.initial_cash - transaction_cost

        shares = available_cash / initial_price

        # Portfolio value over time
        portfolio_values = shares * prices

        # Get dates
        if 'date' in self.env.data.columns:
            dates = self.env.data['date'].values
        else:
            dates = np.arange(len(prices))

        final_value = portfolio_values[-1]
        total_return = (final_value - self.env.initial_cash) / self.env.initial_cash

        logger.info(f"Buy & Hold: Final Value=${final_value:,.2f}, Return={total_return:.2%}")

        return {
            'portfolio_values': portfolio_values.tolist(),
            'strategy': 'Buy and Hold',
            'dates': dates.tolist(),
            'final_value': final_value,
            'total_return': total_return
        }

    def _equal_weight_baseline(self) -> Dict[str, Any]:
        """
        Equal weight rebalancing baseline (for multi-asset).

        Returns:
            Dictionary with baseline results
        """
        # This will be fully implemented in Phase 3 for PortfolioAgent
        raise NotImplementedError("Equal weight baseline for multi-asset not yet implemented")
