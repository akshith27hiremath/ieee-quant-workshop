"""
Aggressive DQN Trading Agent - Tuned for frequent trading and demonstration.

Key differences from conservative agent:
1. Lower transaction costs (2 bps commission, 1 bps slippage) - encourages trading
2. PnL-based reward (instead of Sharpe) - more immediate feedback
3. Higher learning rate - faster adaptation
4. More exploration - tries different actions
5. Uses QQQ (more volatile) instead of SPY
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.environments.timing_env import TimingEnv
import matplotlib.pyplot as plt
from IPython.display import clear_output


class LivePlottingCallback(BaseCallback):
    """Live plotting callback for training progress."""

    def __init__(self, eval_env, plot_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.plot_freq = plot_freq

        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]['r'])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])
            self.timesteps.append(self.num_timesteps)

        if self.num_timesteps % self.plot_freq == 0:
            self.plot_progress()

        return True

    def plot_progress(self):
        if len(self.episode_rewards) < 2:
            return

        clear_output(wait=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Episode rewards
        axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.6, linewidth=0.8)
        if len(self.episode_rewards) > 10:
            window = min(10, len(self.episode_rewards))
            ma = pd.Series(self.episode_rewards).rolling(window).mean()
            axes[0, 0].plot(self.timesteps, ma, 'r-', linewidth=2, label=f'MA({window})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Exploration rate
        exp_rate = self.model.exploration_rate if hasattr(self.model, 'exploration_rate') else 0.05
        axes[0, 1].plot([0, self.num_timesteps], [1.0, exp_rate], 'g-', linewidth=2)
        axes[0, 1].scatter([self.num_timesteps], [exp_rate], c='red', s=100, zorder=5)
        axes[0, 1].set_title(f'Exploration Rate: {exp_rate:.3f}', fontweight='bold')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True, alpha=0.3)

        # Episode lengths
        if len(self.episode_lengths) > 0:
            axes[1, 0].plot(self.timesteps, self.episode_lengths, alpha=0.6)
            axes[1, 0].set_title('Episode Lengths', fontweight='bold')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True, alpha=0.3)

        # Training info
        axes[1, 1].axis('off')
        latest_length = self.episode_lengths[-1] if self.episode_lengths else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        info_text = f"""
TRAINING PROGRESS

Timesteps: {self.num_timesteps:,}
Episodes: {len(self.episode_rewards)}

Latest Episode:
  Reward: {self.episode_rewards[-1]:.2f}
  Length: {latest_length}

Average (last 10):
  Reward: {np.mean(self.episode_rewards[-10:]):.2f}
  Length: {avg_length:.1f}

Exploration: {exp_rate*100:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


def train_aggressive_agent(
    train_data,
    val_data,
    features,
    device='cpu',
    total_timesteps=40000,
    plot_freq=1500
):
    """
    Train aggressive DQN agent for demonstrative trading.
    """
    print("="*70)
    print("TRAINING AGGRESSIVE DQN AGENT")
    print("="*70)

    # AGGRESSIVE CONFIGURATION
    env_config = {
        'initial_cash': 100000,
        'reward_type': 'pnl',  # Direct profit/loss feedback (not Sharpe)
        'transaction_costs': {
            'commission_bps': 2,  # Very low costs (was 10)
            'slippage_bps': 1     # Very low slippage (was 5)
        },
        'position_sizing': 'full',  # Use full capital (more aggressive)
    }

    print("\n🔥 AGGRESSIVE SETTINGS:")
    print(f"  Reward: {env_config['reward_type']} (immediate feedback)")
    print(f"  Commission: {env_config['transaction_costs']['commission_bps']} bps (5x lower!)")
    print(f"  Slippage: {env_config['transaction_costs']['slippage_bps']} bps (5x lower!)")
    print(f"  Position sizing: Full capital")

    # Create environments
    print("\nCreating environments...")
    train_env = TimingEnv(train_data, env_config, features)
    train_env = Monitor(train_env, './demo_logs/train')

    val_env = TimingEnv(val_data, env_config, features)
    val_env = Monitor(val_env, './demo_logs/val')

    print(f"  Train: {len(train_data)} steps")
    print(f"  Val: {len(val_data)} steps")
    print(f"  State dim: {train_env.observation_space.shape[0]}")
    print(f"  Actions: {train_env.action_space.n} (Hold=0, Long=1, Short=2)")

    # Create AGGRESSIVE DQN agent
    print(f"\nCreating aggressive DQN on {device}...")
    model = DQN(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=0.0005,  # Higher LR (was 0.0001)
        buffer_size=50000,
        learning_starts=500,  # Start learning earlier (was 1000)
        batch_size=64,  # Larger batches (was 32)
        tau=0.01,  # Faster target updates (was 0.005)
        gamma=0.98,  # Less future discounting (was 0.99)
        train_freq=2,  # Train more often (was 4)
        gradient_steps=2,  # More gradient steps (was 1)
        exploration_fraction=0.5,  # Explore longer (was 0.3)
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # Higher final exploration (was 0.05)
        target_update_interval=500,  # More frequent updates (was 1000)
        policy_kwargs={'net_arch': [256, 256]},  # Larger network (was [128, 128])
        tensorboard_log='./demo_logs/',
        device=device,
        verbose=0
    )

    print("✓ Aggressive model created")
    print(f"  Learning rate: {model.learning_rate} (5x higher)")
    print(f"  Network: {model.policy_kwargs['net_arch']} (2x larger)")
    print(f"  Exploration: {model.exploration_initial_eps} → {model.exploration_final_eps}")

    # Callbacks
    live_callback = LivePlottingCallback(val_env, plot_freq=plot_freq)

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./demo_models/aggressive_agent/',
        log_path='./demo_logs/',
        eval_freq=3000,  # More frequent eval (was 5000)
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )

    # Train
    print(f"\n🚀 Starting training for {total_timesteps:,} timesteps...")
    print("The agent will be MORE AGGRESSIVE - expect more trades!\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[live_callback, eval_callback],
        log_interval=50,  # More frequent logging
        progress_bar=True
    )

    # Save
    model.save('demo_models/aggressive_agent/final_model')

    print("\n" + "="*70)
    print("✅ AGGRESSIVE TRAINING COMPLETE!")
    print("="*70)
    print(f"  Episodes: {len(live_callback.episode_rewards)}")
    print(f"  Best model: demo_models/aggressive_agent/best_model.zip")
    print(f"  Final model: demo_models/aggressive_agent/final_model.zip")
    print("\nThis agent should trade MORE FREQUENTLY than the conservative one!")

    return model, live_callback


if __name__ == "__main__":
    print("Loading QQQ data...")
    data = pd.read_parquet('demo_data/featured_data.parquet')

    # Split
    test_start = pd.Timestamp('2022-01-01')  # Recent test period
    train_val_data = data[data.index < test_start].copy()
    train_size = int(len(train_val_data) * 0.85)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]

    print(f"Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
    print(f"Val: {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data)} days)")
    print(f"Test: {test_start.date()} to {data.index[-1].date()} ({len(data[data.index >= test_start])} days)")

    features = [
        'return_1d', 'return_5d', 'return_10d',
        'rsi', 'rsi_norm', 'macd', 'macd_signal', 'macd_diff',
        'sma_50', 'sma_200', 'sma_crossover', 'ema_12', 'ema_26',
        'bb_high', 'bb_low', 'bb_width', 'bb_percent', 'atr', 'atr_pct',
        'volume_ratio', 'obv'
    ]

    # Detect device
    if torch.xpu.is_available():
        device = 'xpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Train
    model, callback = train_aggressive_agent(
        train_data, val_data, features,
        device=device,
        total_timesteps=40000,
        plot_freq=1500
    )
