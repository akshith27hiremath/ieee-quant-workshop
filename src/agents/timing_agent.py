"""
TimingAgent Training Script (Phase 2).
Trains a DQN agent for single-asset market timing.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch

# RL imports
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import ConfigLoader
from src.data.features import FeatureEngineer
from src.data.splitter import WalkForwardSplitter
from src.environments.timing_env import TimingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingAgentTrainer:
    """
    Trainer class for TimingAgent using DQN.
    """

    def __init__(self, config_path: str = "config"):
        """
        Initialize trainer.

        Args:
            config_path: Path to configuration directory
        """
        self.config_loader = ConfigLoader(config_path)
        self.timing_config = self.config_loader.load('timing_config')
        self.feature_config = self.config_loader.load('feature_config')
        self.cv_config = self.config_loader.load('cv_config')

        # Create output directories
        self.model_save_path = Path(self.timing_config['training']['save_path'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        self.best_model_path = Path(self.timing_config['training']['best_model_save_path'])
        self.best_model_path.mkdir(parents=True, exist_ok=True)

        # Set device (use XPU if available, otherwise CPU)
        if torch.xpu.is_available():
            self.device = 'xpu'
            logger.info(f"Using Intel XPU device: {torch.xpu.get_device_name(0)}")
        else:
            self.device = 'cpu'
            logger.info("XPU not available, using CPU")

        logger.info("TimingAgentTrainer initialized")

    def load_data(self) -> pd.DataFrame:
        """
        Load featured data for training.

        Returns:
            Featured DataFrame
        """
        data_path = Path("data/features/featured_data.parquet")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Featured data not found at {data_path}. "
                "Run feature engineering notebook first."
            )

        data = pd.read_parquet(data_path)
        logger.info(f"Loaded featured data: {data.shape}")

        # Filter to SPY only for TimingAgent
        ticker = self.timing_config['data']['ticker']
        if 'ticker' in data.columns:
            data = data[data['ticker'] == ticker].copy()
            logger.info(f"Filtered to {ticker}: {data.shape}")

        return data

    def create_env(self, data: pd.DataFrame, is_eval: bool = False) -> Monitor:
        """
        Create trading environment.

        Args:
            data: Input DataFrame
            is_eval: Whether this is for evaluation

        Returns:
            Monitored environment
        """
        # Get feature list
        features = self.timing_config['data']['features']

        # Validate features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        # Create environment
        env = TimingEnv(
            data=data,
            config=self.timing_config['environment'],
            features=features
        )

        # Wrap in Monitor for logging
        log_dir = Path(self.timing_config['agent']['tensorboard_log'])
        log_dir.mkdir(parents=True, exist_ok=True)

        env_type = "eval" if is_eval else "train"
        monitor_path = log_dir / env_type
        monitor_path.mkdir(exist_ok=True)

        env = Monitor(env, str(monitor_path))

        return env

    def create_agent(self, env: Monitor, seed: Optional[int] = None) -> DQN:
        """
        Create DQN agent.

        Args:
            env: Training environment
            seed: Random seed

        Returns:
            DQN agent
        """
        agent_config = self.timing_config['agent']

        # Activation function mapping
        activation_mapping = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }

        policy_kwargs = agent_config.get('policy_kwargs', {}).copy()
        if 'activation_fn' in policy_kwargs:
            act_fn_name = policy_kwargs['activation_fn']
            policy_kwargs['activation_fn'] = activation_mapping.get(act_fn_name, nn.ReLU)

        # Create DQN agent
        model = DQN(
            policy=agent_config['policy'],
            env=env,
            learning_rate=agent_config['learning_rate'],
            buffer_size=agent_config['buffer_size'],
            learning_starts=agent_config['learning_starts'],
            batch_size=agent_config['batch_size'],
            tau=agent_config['tau'],
            gamma=agent_config['gamma'],
            train_freq=agent_config['train_freq'],
            gradient_steps=agent_config['gradient_steps'],
            exploration_fraction=agent_config['exploration_fraction'],
            exploration_initial_eps=agent_config['exploration_initial_eps'],
            exploration_final_eps=agent_config['exploration_final_eps'],
            target_update_interval=agent_config['target_update_interval'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=agent_config['tensorboard_log'],
            device=self.device,  # Use XPU if available
            seed=seed,
            verbose=1
        )

        logger.info(f"Created DQN agent: {agent_config['policy']} on device: {self.device}")
        return model

    def create_callbacks(self, eval_env: Monitor):
        """
        Create training callbacks.

        Args:
            eval_env: Evaluation environment

        Returns:
            List of callbacks
        """
        train_config = self.timing_config['training']

        callbacks = []

        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.best_model_path),
            log_path=str(self.best_model_path / "logs"),
            eval_freq=train_config['eval_freq'],
            n_eval_episodes=train_config['n_eval_episodes'],
            deterministic=train_config['eval_deterministic'],
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)

        # Early stopping
        if train_config['early_stopping']['enabled']:
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=train_config['early_stopping']['patience'],
                min_evals=3,
                verbose=1
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.best_model_path),
                log_path=str(self.best_model_path / "logs"),
                eval_freq=train_config['eval_freq'],
                n_eval_episodes=train_config['n_eval_episodes'],
                deterministic=train_config['eval_deterministic'],
                callback_after_eval=stop_callback,
                verbose=1
            )
            callbacks = [eval_callback]

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=train_config['save_freq'],
            save_path=str(self.model_save_path),
            name_prefix="dqn_timing",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        total_timesteps: Optional[int] = None
    ) -> DQN:
        """
        Train the agent.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            total_timesteps: Total training steps (overrides config)

        Returns:
            Trained DQN model
        """
        train_config = self.timing_config['training']
        seed = train_config.get('seed', 42)

        # Set random seeds
        np.random.seed(seed)

        # Create environments
        logger.info("Creating training and evaluation environments...")
        train_env = self.create_env(train_data, is_eval=False)
        eval_env = self.create_env(val_data, is_eval=True)

        # Create agent
        logger.info("Creating DQN agent...")
        model = self.create_agent(train_env, seed=seed)

        # Create callbacks
        callbacks = self.create_callbacks(eval_env)

        # Train
        total_steps = total_timesteps or train_config['total_timesteps']
        logger.info(f"Starting training for {total_steps} timesteps...")

        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )

        # Save final model
        final_model_path = self.model_save_path / "final_model.zip"
        model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        return model

    def evaluate(
        self,
        model: DQN,
        eval_data: pd.DataFrame,
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate trained model.

        Args:
            model: Trained DQN model
            eval_data: Evaluation dataset
            n_episodes: Number of episodes to run

        Returns:
            Evaluation metrics
        """
        eval_env = self.create_env(eval_data, is_eval=True)

        episode_rewards = []
        episode_stats = []

        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward

            # Get episode statistics
            stats = eval_env.env.get_episode_stats()
            episode_rewards.append(episode_reward)
            episode_stats.append(stats)

            logger.info(
                f"Episode {ep + 1}/{n_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Return={stats['total_return']:.2%}, "
                f"Sharpe={stats['sharpe_ratio']:.2f}"
            )

        # Aggregate results
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_return': np.mean([s['total_return'] for s in episode_stats]),
            'mean_sharpe': np.mean([s['sharpe_ratio'] for s in episode_stats]),
            'mean_max_drawdown': np.mean([s['max_drawdown'] for s in episode_stats]),
            'mean_trades': np.mean([s['total_trades'] for s in episode_stats])
        }

        logger.info("\nEvaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")

        return results


def main():
    """Main training script."""
    # Initialize trainer
    trainer = TimingAgentTrainer()

    # Load data
    data = trainer.load_data()

    # Get train/val split
    splitter = WalkForwardSplitter(trainer.cv_config)
    test_start = pd.Timestamp(trainer.cv_config['test_set']['start_date'])

    # Use data before test set
    train_val_data = data[data.index < test_start].copy()

    # For initial training, use simple split
    # TODO: Later implement walk-forward training
    train_size = int(len(train_val_data) * 0.85)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]

    logger.info(f"Train set: {len(train_data)} rows")
    logger.info(f"Val set: {len(val_data)} rows")

    # Train agent
    model = trainer.train(train_data, val_data)

    # Evaluate on validation set
    results = trainer.evaluate(model, val_data, n_episodes=5)

    logger.info("\nTraining complete!")
    logger.info(f"Best model saved to: {trainer.best_model_path}")
    logger.info(f"Final model saved to: {trainer.model_save_path}")


if __name__ == "__main__":
    main()
