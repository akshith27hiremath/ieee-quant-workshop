"""
TimingAgent Environment (Phase 2).
Discrete action space for single-asset trading.
"""

from gymnasium import spaces
import numpy as np
import pandas as pd
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
        Execute trading action with position management and transaction costs.

        Args:
            action: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
        """
        # Convert action to int if it's a numpy array (from model.predict())
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        current_price = self._get_current_price()

        # Map actions to target positions
        # 0: Hold (maintain current position)
        # 1: Buy/Long (target position = 1)
        # 2: Sell/Short (target position = -1)
        action_to_position = {0: self.position, 1: 1, 2: -1}
        target_position = action_to_position[action]

        # No action needed if already in target position
        if target_position == self.position:
            return

        # Execute trade to reach target position
        # First, close current position if any
        if self.shares_held != 0:
            # Close existing position
            trade_value = abs(self.shares_held * current_price)
            transaction_cost = self._apply_transaction_costs(trade_value)

            # Update cash (sell shares or cover short)
            if self.shares_held > 0:
                # Selling long position
                self.cash += (self.shares_held * current_price) - transaction_cost
            else:
                # Covering short position
                self.cash -= (abs(self.shares_held) * current_price) + transaction_cost

            # Log the closing trade
            self.trade_history.append({
                'step': self.current_step,
                'action': 'close',
                'position': self.position,
                'shares': self.shares_held,
                'price': current_price,
                'cost': transaction_cost
            })

            self.shares_held = 0

        # Open new position if target is not flat
        if target_position != 0:
            # Calculate shares to trade (use 95% of cash to leave buffer)
            usable_cash = self.cash * 0.95
            shares_to_trade = usable_cash / current_price

            if shares_to_trade > 0:
                trade_value = shares_to_trade * current_price
                transaction_cost = self._apply_transaction_costs(trade_value)

                # Update cash and shares
                if target_position == 1:
                    # Going long
                    self.shares_held = shares_to_trade
                    self.cash -= (trade_value + transaction_cost)
                else:  # target_position == -1
                    # Going short
                    self.shares_held = -shares_to_trade
                    self.cash += (trade_value - transaction_cost)

                # Log the opening trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'open',
                    'position': target_position,
                    'shares': self.shares_held,
                    'price': current_price,
                    'cost': transaction_cost
                })

        # Update position tracker
        self.position = target_position

    def _calculate_reward(self) -> float:
        """
        Calculate reward using configurable strategy.

        Reward types:
        - 'pnl': Simple profit/loss change
        - 'sharpe': Risk-adjusted returns (Sharpe ratio proxy)
        - 'sortino': Downside risk-adjusted
        - 'drawdown_aware': PnL with drawdown penalty

        Returns:
            Reward value
        """
        reward_type = self.config.get('reward_type', 'pnl')

        # Simple PnL change (baseline)
        pnl = self.portfolio_value - self.previous_portfolio_value

        if reward_type == 'pnl':
            # Simple profit/loss
            return float(pnl)

        elif reward_type == 'sharpe':
            # Risk-adjusted reward (Sharpe proxy)
            # Use rolling window of returns to calculate volatility
            if len(self.portfolio_history) < 20:
                # Not enough history for reliable vol estimate
                return float(pnl)

            # Calculate returns over last 20 steps
            returns = pd.Series(self.portfolio_history[-20:]).pct_change().dropna()
            if len(returns) < 2:
                return float(pnl)

            # Sharpe ratio approximation
            mean_return = returns.mean()
            std_return = returns.std()
            risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252  # Daily

            if std_return > 0:
                sharpe = (mean_return - risk_free_rate) / std_return
                reward = sharpe * 100  # Scale for stability
            else:
                reward = mean_return * 100

            return float(reward)

        elif reward_type == 'sortino':
            # Downside risk-adjusted
            if len(self.portfolio_history) < 20:
                return float(pnl)

            returns = pd.Series(self.portfolio_history[-20:]).pct_change().dropna()
            if len(returns) < 2:
                return float(pnl)

            mean_return = returns.mean()
            # Only penalize downside volatility
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
            else:
                downside_std = 0.0001  # Small value to avoid division by zero

            sortino = mean_return / downside_std if downside_std > 0 else mean_return * 100
            reward = sortino * 100  # Scale for stability

            return float(reward)

        elif reward_type == 'drawdown_aware':
            # PnL with explicit drawdown penalty
            if len(self.portfolio_history) < 2:
                return float(pnl)

            # Calculate current drawdown
            max_value = max(self.portfolio_history)
            current_dd = (max_value - self.portfolio_value) / max_value if max_value > 0 else 0.0

            # Penalty coefficient (higher = more penalty)
            dd_penalty_coef = self.config.get('drawdown_penalty', 10.0)

            # Reward = PnL - penalty for increasing drawdown
            if len(self.portfolio_history) >= 2:
                prev_max = max(self.portfolio_history[:-1])
                prev_dd = (prev_max - self.previous_portfolio_value) / prev_max if prev_max > 0 else 0.0
                dd_change = current_dd - prev_dd
                reward = pnl - (dd_penalty_coef * dd_change * self.initial_cash)
            else:
                reward = pnl

            return float(reward)

        else:
            # Default to simple PnL
            return float(pnl)
