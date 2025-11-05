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

logging.basicConfig(level=logging.INFO)
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
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
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
        return float(self.data.loc[self.current_step, 'close'])

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
            if len(self.portfolio_history) > 0:
                max_value = max(self.portfolio_history)
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
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': len(self.trade_history),
                'final_value': self.portfolio_value
            }

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
        return float(sharpe)

    @staticmethod
    def _calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0

        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return float(abs(drawdown.min()))
