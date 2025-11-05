"""
TimingAgent Environment (Phase 2).
Discrete action space for single-asset trading.
"""

from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional
from .base_env import BaseTradingEnv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingEnv(BaseTradingEnv):
    """
    Trading environment for TimingAgent.

    Action Space: Discrete(3)
        - 0: Hold / Flat
        - 1: Buy / Go Long
        - 2: Sell / Go Short

    This environment will be fully implemented in Phase 2.
    """

    def __init__(self, data, config, features):
        super().__init__(data, config, features)

        # Define action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Define observation space
        n_features = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # Position tracking
        self.position = 0  # -1: short, 0: flat, 1: long
        self.shares_held = 0.0

        logger.info(f"TimingEnv initialized: {n_features} features, 3 actions")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash
        self.position = 0
        self.shares_held = 0.0

        self.portfolio_history = [self.initial_cash]
        self.trade_history = []
        self.reward_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)

        Returns:
            observation, reward, done, truncated, info
        """
        # Store previous portfolio value
        self.previous_portfolio_value = self.portfolio_value

        # Execute action (placeholder - full implementation in Phase 2)
        self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Update portfolio value
        current_price = self._get_current_price()
        self.portfolio_value = self.cash + (self.shares_held * current_price)

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = self._is_done()
        truncated = False

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        # Track history
        self.portfolio_history.append(self.portfolio_value)
        self.reward_history.append(reward)

        return obs, reward, done, truncated, info

    def _execute_action(self, action: int):
        """
        Execute trading action (placeholder for Phase 2).

        Args:
            action: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
        """
        # TODO: Implement in Phase 2
        # This will include:
        # - Position management
        # - Transaction cost calculation
        # - Trade logging
        pass

    def _calculate_reward(self) -> float:
        """
        Calculate reward (placeholder for Phase 2).

        Returns:
            Reward value
        """
        # TODO: Implement in Phase 2
        # Options: PnL, Sharpe, Sortino, Drawdown-aware
        # For now, simple PnL change
        reward = self.portfolio_value - self.previous_portfolio_value
        return float(reward)
