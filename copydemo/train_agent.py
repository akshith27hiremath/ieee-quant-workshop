"""
Training script for DQN agent - used by notebook 03.
Self-contained with embedded configuration.
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
from IPython.display import clear_output, display
import time

class LivePlottingCallback(BaseCallback):
    """
    Custom callback for live plotting during training.
    Shows episode rewards, exploration rate, and other metrics in real-time.
    """
    def __init__(self, eval_env, plot_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.plot_freq = plot_freq

        # Storage for metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.losses = []
        self.exploration_rates = []
        self.eval_rewards = []
        self.eval_timesteps = []

    def _on_step(self) -> bool:
        # Store metrics
        if len(self.model.ep_info_buffer) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]['r'])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])
            self.timesteps.append(self.num_timesteps)

        # Plot periodically
        if self.num_timesteps % self.plot_freq == 0:
            self.plot_progress()

        return True

    def plot_progress(self):
        """Create live training visualization."""
        if len(self.episode_rewards) < 2:
            return

        clear_output(wait=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 1. Episode rewards
        axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.6, linewidth=0.8)
        if len(self.episode_rewards) > 10:
            # Moving average
            window = min(10, len(self.episode_rewards))
            ma = pd.Series(self.episode_rewards).rolling(window).mean()
            axes[0, 0].plot(self.timesteps, ma, 'r-', linewidth=2, label=f'MA({window})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Exploration rate
        if hasattr(self.model, 'exploration_rate'):
            exp_rate = self.model.exploration_rate
        else:
            exp_rate = 0.05
        axes[0, 1].plot([0, self.num_timesteps], [1.0, exp_rate], 'g-', linewidth=2)
        axes[0, 1].scatter([self.num_timesteps], [exp_rate], c='red', s=100, zorder=5)
        axes[0, 1].set_title(f'Exploration Rate: {exp_rate:.3f}', fontweight='bold')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Episode lengths
        if len(self.episode_lengths) > 0:
            axes[1, 0].plot(self.timesteps, self.episode_lengths, alpha=0.6)
            axes[1, 0].set_title('Episode Lengths', fontweight='bold')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Training info
        axes[1, 1].axis('off')
        info_text = f"""
TRAINING PROGRESS

Timesteps: {self.num_timesteps:,}
Episodes: {len(self.episode_rewards)}

Latest Episode:
  Reward: {self.episode_rewards[-1]:.2f}
  Length: {self.episode_lengths[-1] if self.episode_lengths else 0}

Average (last 10):
  Reward: {np.mean(self.episode_rewards[-10:]):.2f}
  Length: {np.mean(self.episode_lengths[-10:]):.1f if self.episode_lengths else 0}

Exploration: {exp_rate*100:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


def train_dqn_agent(
    train_data,
    val_data,
    features,
    device='cpu',
    total_timesteps=50000,
    plot_freq=2000
):
    """
    Train DQN agent with live visualization.

    Args:
        train_data: Training dataset
        val_data: Validation dataset
        features: List of feature column names
        device: 'cpu', 'cuda', or 'xpu'
        total_timesteps: Total training steps
        plot_freq: How often to update plots

    Returns:
        Trained model, callback with metrics
    """
    print("="*70)
    print("STARTING DQN TRAINING")
    print("="*70)

    # Environment configuration
    env_config = {
        'initial_cash': 100000,
        'reward_type': 'sharpe',
        'transaction_costs': {
            'commission_bps': 10,
            'slippage_bps': 5
        }
    }

    # Create environments
    print("\\nCreating environments...")
    train_env = TimingEnv(train_data, env_config, features)
    train_env = Monitor(train_env, './demo_logs/train')

    val_env = TimingEnv(val_data, env_config, features)
    val_env = Monitor(val_env, './demo_logs/val')

    print(f"  Train env: {len(train_data)} timesteps")
    print(f"  Val env: {len(val_data)} timesteps")
    print(f"  State dim: {train_env.observation_space.shape[0]}")
    print(f"  Action dim: {train_env.action_space.n}")

    # Create DQN agent
    print(f"\\nCreating DQN agent on device: {device}...")
    model = DQN(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=0.0001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        policy_kwargs={'net_arch': [128, 128]},
        tensorboard_log='./demo_logs/',
        device=device,
        verbose=0
    )

    print("✓ Model created")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Buffer size: {model.buffer_size:,}")
    print(f"  Network: {model.policy_kwargs['net_arch']}")

    # Create callbacks
    live_callback = LivePlottingCallback(val_env, plot_freq=plot_freq)

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./demo_models/dqn_agent/',
        log_path='./demo_logs/',
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )

    # Train
    print(f"\\nStarting training for {total_timesteps:,} timesteps...")
    print("Watch the plots update in real-time!\\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[live_callback, eval_callback],
        log_interval=100,
        progress_bar=True
    )

    # Save final model
    model.save('demo_models/dqn_agent/final_model')

    print("\\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"  Total episodes: {len(live_callback.episode_rewards)}")
    print(f"  Best model saved to: demo_models/dqn_agent/best_model.zip")
    print(f"  Final model saved to: demo_models/dqn_agent/final_model.zip")

    return model, live_callback


if __name__ == "__main__":
    # This allows running standalone for testing
    print("Loading data...")
    data = pd.read_parquet('../data/features/featured_data.parquet')
    data = data[data['ticker'] == 'SPY'].copy()

    # Split
    test_start = pd.Timestamp('2021-01-01')
    train_val_data = data[data.index < test_start].copy()
    train_size = int(len(train_val_data) * 0.85)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]

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
    model, callback = train_dqn_agent(
        train_data, val_data, features,
        device=device,
        total_timesteps=30000,
        plot_freq=1500
    )
