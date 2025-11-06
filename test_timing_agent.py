"""
Quick test script for TimingAgent setup.
Tests XPU, environment, and DQN agent with minimal training.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("TIMINGAGENT QUICK TEST")
print("=" * 70)

# Test 1: XPU Detection
print("\n[1/5] Testing XPU availability...")
if torch.xpu.is_available():
    device = 'xpu'
    print(f"[OK] XPU Available: {torch.xpu.get_device_name(0)}")
    print(f"  Device count: {torch.xpu.device_count()}")
else:
    device = 'cpu'
    print("[WARN] XPU not available, using CPU")

# Test 2: Load Configuration
print("\n[2/5] Loading configuration...")
from src.utils.config import ConfigLoader

config_loader = ConfigLoader('config')
timing_config = config_loader.load('timing_config')
cv_config = config_loader.load('cv_config')
print(f"[OK] Config loaded")
print(f"  Reward type: {timing_config['environment']['reward_type']}")
print(f"  Learning rate: {timing_config['agent']['learning_rate']}")

# Test 3: Load Data and Create Environment
print("\n[3/5] Creating TimingEnv...")
from src.environments.timing_env import TimingEnv

# Load data
data = pd.read_parquet('data/features/featured_data.parquet')
ticker = timing_config['data']['ticker']
if 'ticker' in data.columns:
    data = data[data['ticker'] == ticker].copy()

# Get train data (use small subset for testing)
test_start = pd.Timestamp(cv_config['test_set']['start_date'])
train_val_data = data[data.index < test_start].copy()
train_size = int(len(train_val_data) * 0.85)
train_data = train_val_data.iloc[:train_size]

print(f"[OK] Loaded {len(train_data)} training rows")

# Create environment
features = timing_config['data']['features']
features = [f for f in features if f in train_data.columns]

env = TimingEnv(
    data=train_data,
    config=timing_config['environment'],
    features=features
)

print(f"[OK] TimingEnv created")
print(f"  Features: {len(features)}")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: Discrete({env.action_space.n})")

# Test 4: Test Environment with Random Actions
print("\n[4/5] Testing environment with 50 random steps...")
obs, info = env.reset()
total_reward = 0

for i in range(50):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f"[OK] Environment works!")
print(f"  Steps: {i + 1}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Final portfolio: ${info['portfolio_value']:,.2f}")

# Test 5: Create and Quick Train DQN Agent
print("\n[5/5] Testing DQN agent with 5,000 timesteps...")
print("This will take ~30 seconds...")

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Wrap env in Monitor
env = Monitor(env, './logs/test_run')

# Create minimal DQN agent
model = DQN(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.0001,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    device=device,
    verbose=0
)

print(f"[OK] DQN agent created on device: {device}")

# Quick training
print("  Training...")
model.learn(total_timesteps=5000, progress_bar=True)

print(f"[OK] Training complete!")

# Quick evaluation
print("\n  Evaluating trained agent...")
obs, info = env.reset()
episode_reward = 0
steps = 0

while steps < 100:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    steps += 1
    if done or truncated:
        break

print(f"[OK] Evaluation complete!")
print(f"  Steps: {steps}")
print(f"  Episode reward: {episode_reward:.2f}")
print(f"  Final portfolio: ${info['portfolio_value']:,.2f}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] XPU Detection:        PASS")
print("[OK] Configuration:        PASS")
print("[OK] TimingEnv:            PASS")
print("[OK] Random Agent:         PASS")
print("[OK] DQN Training:         PASS")
print("\n>>> All systems ready for full training!")
print("=" * 70)
print("\nNext steps:")
print("  1. Open JupyterLab: jupyter lab")
print("  2. Open: 04_timing_agent_training.ipynb")
print("  3. Run all cells for full 100k timestep training")
print("\n  Or run: python src/agents/timing_agent.py")
print("=" * 70)
