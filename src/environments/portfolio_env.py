"""
PortfolioAgent Environment (Phase 3).
Continuous action space for multi-asset allocation.
"""

from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional
from .base_env import BaseTradingEnv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioEnv(BaseTradingEnv):
    """
    Trading environment for PortfolioAgent.

    Action Space: Box(N_assets + 1)
        - Continuous values representing target portfolio weights
        - [w_asset1, w_asset2, ..., w_assetN, w_cash]
        - Will be normalized to sum to 1

    This environment will be fully implemented in Phase 3.
    """

    def __init__(self, data, config, features, n_assets: int):
        super().__init__(data, config, features)

        self.n_assets = n_assets

        # Define action space: continuous weights for each asset + cash
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets + 1,),
            dtype=np.float32
        )

        # Define observation space
        n_features = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # Portfolio holdings
        self.shares_held = np.zeros(n_assets)  # Shares of each asset
        self.weights = np.zeros(n_assets + 1)  # Current weights (assets + cash)
        self.weights[-1] = 1.0  # Start with 100% cash

        logger.info(f"PortfolioEnv initialized: {n_features} features, {n_assets} assets")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash

        self.shares_held = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[-1] = 1.0  # 100% cash

        self.portfolio_history = [self.initial_cash]
        self.trade_history = []
        self.reward_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: Array of target portfolio weights

        Returns:
            observation, reward, done, truncated, info
        """
        # Store previous portfolio value
        self.previous_portfolio_value = self.portfolio_value

        # Execute action (placeholder - full implementation in Phase 3)
        self._execute_rebalancing(action)

        # Move to next step
        self.current_step += 1

        # Update portfolio value
        self.portfolio_value = self._calculate_portfolio_value()

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

    def _execute_rebalancing(self, target_weights: np.ndarray):
        """
        Execute portfolio rebalancing (placeholder for Phase 3).

        Args:
            target_weights: Target portfolio weights
        """
        # TODO: Implement in Phase 3
        # This will include:
        # - Weight normalization (softmax or budget projection)
        # - Rebalancing logic with no-trade bands
        # - Transaction cost calculation
        # - Turnover penalty
        pass

    def _calculate_portfolio_value(self) -> float:
        """
        Calculate current portfolio value.

        Returns:
            Total portfolio value
        """
        # TODO: Implement in Phase 3
        # For now, just return cash
        return float(self.cash)

    def _calculate_reward(self) -> float:
        """
        Calculate reward (placeholder for Phase 3).

        Returns:
            Reward value
        """
        # TODO: Implement in Phase 3
        # Options: Sharpe, Risk-adjusted, Turnover-penalized
        # For now, simple PnL change
        reward = self.portfolio_value - self.previous_portfolio_value
        return float(reward)
