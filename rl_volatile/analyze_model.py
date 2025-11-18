"""
Model analysis script - used by notebook 04.
Analyzes trained DQN model to understand its strategy.
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import DQN
from src.environments.timing_env import TimingEnv
from stable_baselines3.common.monitor import Monitor


def analyze_q_values(model, data, features, env_config):
    """
    Analyze Q-values across the entire dataset to understand agent's strategy.
    """
    print("Analyzing Q-values...")

    # Create environment
    env = TimingEnv(data, env_config, features)

    # Run through entire dataset and collect Q-values
    obs, info = env.reset()
    done = False
    truncated = False

    q_values_history = []
    actions_history = []
    timestamps = []
    prices = []
    states_history = []

    step = 0
    while not (done or truncated):
        # Get Q-values
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            q_values = model.q_net(obs_tensor).cpu().numpy()[0]

        # Get action
        action, _ = model.predict(obs, deterministic=True)

        # Store
        q_values_history.append(q_values)
        actions_history.append(int(action))
        timestamps.append(data.index[env.current_step])
        prices.append(env._get_current_price())
        states_history.append(obs.copy())

        # Step
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    # Convert to arrays
    q_values_array = np.array(q_values_history)
    actions_array = np.array(actions_history)
    states_array = np.array(states_history)

    print(f"  Analyzed {len(q_values_array)} timesteps")

    return {
        'q_values': q_values_array,
        'actions': actions_array,
        'timestamps': timestamps,
        'prices': np.array(prices),
        'states': states_array
    }


def visualize_strategy(analysis_results, save_path='demo_outputs/strategy_analysis.png'):
    """
    Create comprehensive strategy visualization.
    """
    q_vals = analysis_results['q_values']
    actions = analysis_results['actions']
    timestamps = analysis_results['timestamps']
    prices = analysis_results['prices']

    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    # 1. Q-values over time
    ax1 = axes[0]
    ax1.plot(timestamps, q_vals[:, 0], label='Q(Hold)', alpha=0.7, linewidth=1)
    ax1.plot(timestamps, q_vals[:, 1], label='Q(Long)', alpha=0.7, linewidth=1)
    ax1.plot(timestamps, q_vals[:, 2], label='Q(Short)', alpha=0.7, linewidth=1)
    ax1.set_title('Q-Values Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Q-Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Q-value spreads (confidence)
    ax2 = axes[1]
    q_max = q_vals.max(axis=1)
    q_min = q_vals.min(axis=1)
    q_spread = q_max - q_min
    ax2.plot(timestamps, q_spread, color='purple', linewidth=1)
    ax2.fill_between(timestamps, 0, q_spread, alpha=0.3, color='purple')
    ax2.set_title('Q-Value Spread (Agent Confidence)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Max Q - Min Q')
    ax2.grid(True, alpha=0.3)

    # 3. Actions taken
    ax3 = axes[2]
    action_colors = {0: 'gray', 1: 'green', 2: 'red'}
    for action in [0, 1, 2]:
        mask = actions == action
        if mask.any():
            ax3.scatter(np.array(timestamps)[mask], [action]*mask.sum(),
                       c=action_colors[action], alpha=0.5, s=10)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Hold', 'Long', 'Short'])
    ax3.set_title('Actions Taken', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Action')
    ax3.grid(True, alpha=0.3)

    # 4. Price with preferred action overlay
    ax4 = axes[3]
    ax4.plot(timestamps, prices, color='black', linewidth=1, alpha=0.5)

    # Color background by preferred action
    for i in range(len(actions)-1):
        if actions[i] == 1:  # Long
            ax4.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='green')
        elif actions[i] == 2:  # Short
            ax4.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='red')

    ax4.set_title('Price with Agent Preference', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Price ($)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved strategy visualization: {save_path}")


def get_feature_importance(analysis_results, feature_names):
    """
    Analyze which features the agent pays most attention to.
    Uses variance in Q-values when features change.
    """
    states = analysis_results['states']
    q_values = analysis_results['q_values']

    # Calculate correlation between each feature and Q-value changes
    feature_importance = {}

    for idx, feature in enumerate(feature_names):
        feature_vals = states[:, idx]
        q_max = q_values.max(axis=1)

        # Correlation between feature and max Q-value
        corr = np.corrcoef(feature_vals, q_max)[0, 1]
        feature_importance[feature] = abs(corr) if not np.isnan(corr) else 0

    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)

    return importance_df


def print_strategy_summary(analysis_results):
    """
    Print a summary of the learned strategy.
    """
    q_vals = analysis_results['q_values']
    actions = analysis_results['actions']

    print("\\n" + "="*70)
    print("LEARNED STRATEGY SUMMARY")
    print("="*70)

    # Action distribution
    action_counts = pd.Series(actions).value_counts().sort_index()
    action_names = {0: 'Hold', 1: 'Long', 2: 'Short'}

    print("\\nAction Distribution:")
    for action, count in action_counts.items():
        pct = count / len(actions) * 100
        print(f"  {action_names[action]:<8} {count:>6} ({pct:>5.1f}%)")

    # Preferred action
    most_common = action_counts.idxmax()
    print(f"\\nPreferred Action: {action_names[most_common]}")

    # Q-value statistics
    print("\\nAverage Q-Values:")
    print(f"  Q(Hold):  {q_vals[:, 0].mean():>8.2f}")
    print(f"  Q(Long):  {q_vals[:, 1].mean():>8.2f}")
    print(f"  Q(Short): {q_vals[:, 2].mean():>8.2f}")

    # Confidence
    q_spread = q_vals.max(axis=1) - q_vals.min(axis=1)
    print(f"\\nDecision Confidence:")
    print(f"  Avg Q-spread: {q_spread.mean():.2f}")
    print(f"  Max Q-spread: {q_spread.max():.2f}")
    print(f"  Min Q-spread: {q_spread.min():.2f}")

    # Strategy interpretation
    print("\\nStrategy Interpretation:")
    if most_common == 0:
        print("  → Agent prefers to HOLD (stay in cash)")
        print("  → Risk-averse strategy, avoiding transaction costs")
    elif most_common == 1:
        print("  → Agent prefers to be LONG (buy and hold)")
        print("  → Bullish strategy, expects positive returns")
    else:
        print("  → Agent prefers to be SHORT")
        print("  → Bearish strategy, expects negative returns")

    if action_counts.std() < 10:
        print("  → Very consistent strategy (low variance in actions)")
    else:
        print("  → Adaptive strategy (varies actions based on market state)")

    print("="*70)


if __name__ == "__main__":
    # Test standalone
    print("="*70)
    print("ANALYZING AGGRESSIVE QQQ AGENT STRATEGY")
    print("="*70)

    print("\nLoading model...")
    model_path = Path('demo_models/aggressive_agent/best_model.zip')

    if not model_path.exists():
        # Try without .zip extension (stable-baselines3 adds it automatically)
        model_path = Path('demo_models/aggressive_agent/best_model')

        if not model_path.exists():
            print(f"ERROR: Model not found!")
            print(f"Tried: demo_models/aggressive_agent/best_model.zip")
            print(f"Tried: demo_models/aggressive_agent/best_model")
            print("\nAvailable files in demo_models/aggressive_agent/:")
            model_dir = Path('demo_models/aggressive_agent/')
            if model_dir.exists():
                for f in model_dir.iterdir():
                    print(f"  - {f.name}")
            else:
                print(f"  Directory {model_dir} does not exist!")
                print("\nRun training first: python train_agent_aggressive.py")
            sys.exit(1)

    print(f"✓ Loading model from: {model_path}")
    # Load without .zip extension, let stable-baselines3 handle it
    model = DQN.load(str(model_path).replace('.zip', ''))

    print("Loading QQQ data...")
    data = pd.read_parquet('demo_data/featured_data.parquet')

    # Use 2022+ as test period (aggressive agent trained on 2010-2022)
    test_start = pd.Timestamp('2022-01-01')
    test_data = data[data.index >= test_start].copy()

    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"Test data: {len(test_data)} days\n")

    features = [
        'return_1d', 'return_5d', 'return_10d',
        'rsi', 'rsi_norm', 'macd', 'macd_signal', 'macd_diff',
        'sma_50', 'sma_200', 'sma_crossover', 'ema_12', 'ema_26',
        'bb_high', 'bb_low', 'bb_width', 'bb_percent', 'atr', 'atr_pct',
        'volume_ratio', 'obv'
    ]

    # AGGRESSIVE AGENT CONFIG (same as training)
    env_config = {
        'initial_cash': 100000,
        'reward_type': 'pnl',  # PnL reward (not Sharpe)
        'transaction_costs': {'commission_bps': 2, 'slippage_bps': 1}  # Low costs
    }

    # Analyze
    results = analyze_q_values(model, test_data, features, env_config)
    visualize_strategy(results)
    print_strategy_summary(results)

    # Feature importance
    importance = get_feature_importance(results, features)
    print("\\nTop 10 Most Important Features:")
    print(importance.head(10))
