"""
Visualize RL agent's trading behavior and learning process.
Shows actions, rewards, portfolio value, and Q-values over time.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('.')

from stable_baselines3 import DQN
from src.utils.config import ConfigLoader
from src.environments.timing_env import TimingEnv
from stable_baselines3.common.monitor import Monitor

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)

print("="*70)
print("RL AGENT TRADING VISUALIZATION")
print("="*70)

# Load configuration
config_loader = ConfigLoader('config')
timing_config = config_loader.load('timing_config')
cv_config = config_loader.load('cv_config')

# Load data
print("\n[1/5] Loading data...")
data = pd.read_parquet('data/features/featured_data.parquet')
data = data[data['ticker'] == 'SPY'].copy()

# Get validation set
test_start = pd.Timestamp(cv_config['test_set']['start_date'])
train_val_data = data[data.index < test_start].copy()
train_size = int(len(train_val_data) * 0.85)
val_data = train_val_data.iloc[train_size:]

print(f"Validation period: {val_data.index[0]} to {val_data.index[-1]}")
print(f"Data points: {len(val_data)}")

# Load trained model
print("\n[2/5] Loading trained DQN model...")
model_path = Path("models/timing_agent/best/best_model.zip")
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    print("Please run training first: python src/agents/timing_agent.py")
    sys.exit(1)

model = DQN.load(model_path)
print(f"Model loaded from {model_path}")

# Create environment
print("\n[3/5] Running agent on validation set...")
features = timing_config['data']['features']
features = [f for f in features if f in val_data.columns]

env = TimingEnv(
    data=val_data,
    config=timing_config['environment'],
    features=features
)

# Run episode and collect detailed info
obs, info = env.reset()
done = False
truncated = False

# Storage for visualization
timestamps = []
portfolio_values = []
cash_values = []
position_values = []
actions_taken = []
rewards_received = []
positions = []
prices = []
q_values_list = []

step = 0
while not (done or truncated):
    # Get Q-values for current state
    q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]

    # Get action from policy
    action, _ = model.predict(obs, deterministic=True)

    # Store current state
    timestamps.append(val_data.index[env.current_step])
    portfolio_values.append(env.portfolio_value)
    cash_values.append(env.cash)
    position_values.append(env.shares_held * env._get_current_price())
    positions.append(env.position)
    prices.append(env._get_current_price())
    actions_taken.append(int(action))
    q_values_list.append(q_values)

    # Take action
    obs, reward, done, truncated, info = env.step(action)
    rewards_received.append(reward)

    step += 1

# Get final episode stats
episode_stats = env.get_episode_stats()

print(f"Episode complete: {step} steps")
print(f"Final portfolio: ${info['portfolio_value']:,.2f}")
print(f"Total return: {episode_stats['total_return']:.2%}")
print(f"Sharpe ratio: {episode_stats['sharpe_ratio']:.2f}")

# Create comprehensive visualization
print("\n[4/5] Creating visualizations...")

fig, axes = plt.subplots(5, 1, figsize=(16, 16))
fig.suptitle('DQN Agent Trading Behavior Analysis', fontsize=16, fontweight='bold')

# Convert to arrays for plotting
timestamps = pd.to_datetime(timestamps)
portfolio_values = np.array(portfolio_values)
cash_values = np.array(cash_values)
position_values = np.array(position_values)
actions_taken = np.array(actions_taken)
rewards_received = np.array(rewards_received)
positions = np.array(positions)
prices = np.array(prices)
q_values_array = np.array(q_values_list)

# 1. Portfolio Value & Price
ax1 = axes[0]
ax1_twin = ax1.twinx()

line1 = ax1.plot(timestamps, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_ylabel('Portfolio Value ($)', color='b', fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

line2 = ax1_twin.plot(timestamps, prices, 'g-', linewidth=1, alpha=0.7, label='SPY Price')
ax1_twin.set_ylabel('SPY Price ($)', color='g', fontsize=11, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='g')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=9)
ax1.set_title('Portfolio Value vs SPY Price', fontsize=12, fontweight='bold')

# 2. Actions and Positions
ax2 = axes[1]
action_colors = {0: 'gray', 1: 'green', 2: 'red'}
action_labels = {0: 'Hold', 1: 'Long', 2: 'Short'}

# Plot position as background
for i in range(len(positions)-1):
    if positions[i] == 1:  # Long
        ax2.axvspan(timestamps[i], timestamps[i+1], alpha=0.2, color='green')
    elif positions[i] == -1:  # Short
        ax2.axvspan(timestamps[i], timestamps[i+1], alpha=0.2, color='red')

# Mark action changes
action_changes = np.where(np.diff(actions_taken) != 0)[0] + 1
for idx in action_changes:
    action = actions_taken[idx]
    color = action_colors[action]
    marker = '^' if action == 1 else ('v' if action == 2 else 'o')
    ax2.scatter(timestamps[idx], 0, c=color, marker=marker, s=150,
                label=action_labels[action] if action not in [a for a in actions_taken[:idx]] else "",
                edgecolors='black', linewidths=1.5, zorder=5)

ax2.set_ylim(-0.5, 0.5)
ax2.set_ylabel('Actions', fontsize=11, fontweight='bold')
ax2.set_yticks([])
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title('Trading Actions (Green=Long, Red=Short, Gray=Hold)', fontsize=12, fontweight='bold')

# 3. Rewards Over Time
ax3 = axes[2]
cumulative_rewards = np.cumsum(rewards_received)
ax3.plot(timestamps, rewards_received, 'purple', alpha=0.3, linewidth=0.5, label='Step Reward')
ax3.plot(timestamps, cumulative_rewards, 'purple', linewidth=2, label='Cumulative Reward')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.fill_between(timestamps, rewards_received, 0, where=(np.array(rewards_received)>=0),
                  color='green', alpha=0.2, label='Positive Reward')
ax3.fill_between(timestamps, rewards_received, 0, where=(np.array(rewards_received)<0),
                  color='red', alpha=0.2, label='Negative Reward')
ax3.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title('Rewards and Penalties Over Time', fontsize=12, fontweight='bold')

# 4. Q-Values Evolution
ax4 = axes[3]
ax4.plot(timestamps, q_values_array[:, 0], 'gray', linewidth=1.5, label='Q(Hold)', alpha=0.7)
ax4.plot(timestamps, q_values_array[:, 1], 'green', linewidth=1.5, label='Q(Long)', alpha=0.7)
ax4.plot(timestamps, q_values_array[:, 2], 'red', linewidth=1.5, label='Q(Short)', alpha=0.7)
ax4.set_ylabel('Q-Value', fontsize=11, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_title('Q-Values for Each Action (Agent\'s Expected Future Returns)', fontsize=12, fontweight='bold')

# 5. Cash vs Position Value
ax5 = axes[4]
ax5.fill_between(timestamps, 0, cash_values, color='blue', alpha=0.3, label='Cash')
ax5.fill_between(timestamps, cash_values, cash_values + position_values,
                 color='orange', alpha=0.3, label='Position Value')
ax5.plot(timestamps, portfolio_values, 'black', linewidth=2, label='Total Portfolio')
ax5.set_ylabel('Value ($)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Date', fontsize=11, fontweight='bold')
ax5.legend(loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_title('Portfolio Composition (Cash vs Invested)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('agent_trading_visualization.png', dpi=150, bbox_inches='tight')
print("Saved: agent_trading_visualization.png")

# Create summary statistics
print("\n[5/5] Generating summary statistics...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Agent Learning & Performance Statistics', fontsize=16, fontweight='bold')

# Action distribution
ax_actions = axes2[0, 0]
action_counts = pd.Series(actions_taken).value_counts().sort_index()
action_names = ['Hold', 'Long', 'Short']
colors = ['gray', 'green', 'red']
ax_actions.bar(range(len(action_counts)), action_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax_actions.set_xticks(range(len(action_counts)))
ax_actions.set_xticklabels([action_names[i] for i in action_counts.index])
ax_actions.set_ylabel('Count', fontsize=11, fontweight='bold')
ax_actions.set_title('Action Distribution', fontsize=12, fontweight='bold')
ax_actions.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(action_counts.values):
    ax_actions.text(i, v + 5, str(v), ha='center', fontweight='bold')

# Reward distribution
ax_rewards = axes2[0, 1]
ax_rewards.hist(rewards_received, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax_rewards.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Reward')
ax_rewards.axvline(x=np.mean(rewards_received), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards_received):.2f}')
ax_rewards.set_xlabel('Reward Value', fontsize=11, fontweight='bold')
ax_rewards.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax_rewards.set_title('Reward Distribution', fontsize=12, fontweight='bold')
ax_rewards.legend(fontsize=9)
ax_rewards.grid(True, alpha=0.3)

# Q-value statistics
ax_qvals = axes2[1, 0]
q_means = q_values_array.mean(axis=0)
q_stds = q_values_array.std(axis=0)
x_pos = np.arange(3)
ax_qvals.bar(x_pos, q_means, yerr=q_stds, color=colors, alpha=0.7,
             edgecolor='black', capsize=5)
ax_qvals.set_xticks(x_pos)
ax_qvals.set_xticklabels(action_names)
ax_qvals.set_ylabel('Average Q-Value', fontsize=11, fontweight='bold')
ax_qvals.set_title('Average Q-Values by Action', fontsize=12, fontweight='bold')
ax_qvals.grid(True, alpha=0.3, axis='y')
for i, (m, s) in enumerate(zip(q_means, q_stds)):
    ax_qvals.text(i, m + s + 100, f'{m:.0f}', ha='center', fontweight='bold')

# Episode statistics summary
ax_stats = axes2[1, 1]
ax_stats.axis('off')

# Calculate additional stats
positive_rewards = sum(1 for r in rewards_received if r > 0)
negative_rewards = sum(1 for r in rewards_received if r < 0)
win_rate = positive_rewards / len(rewards_received) if len(rewards_received) > 0 else 0

stats_text = f"""
EPISODE PERFORMANCE SUMMARY

Portfolio Metrics:
  Initial Capital:     ${100000:,.2f}
  Final Value:         ${portfolio_values[-1]:,.2f}
  Total Return:        {episode_stats['total_return']:.2%}

Risk Metrics:
  Sharpe Ratio:        {episode_stats['sharpe_ratio']:.2f}
  Max Drawdown:        {episode_stats['max_drawdown']:.2%}

Trading Activity:
  Total Trades:        {episode_stats['total_trades']}
  Steps in Market:     {sum(1 for p in positions if p != 0)}
  Steps in Cash:       {sum(1 for p in positions if p == 0)}

Reward Statistics:
  Total Reward:        {sum(rewards_received):.2f}
  Mean Reward:         {np.mean(rewards_received):.2f}
  Positive Steps:      {positive_rewards} ({win_rate:.1%})
  Negative Steps:      {negative_rewards}

Q-Value Insights:
  Preferred Action:    {action_names[np.argmax(q_means)]}
  Q-Value Spread:      {q_means.max() - q_means.min():.2f}
"""
ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('agent_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: agent_statistics.png")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. agent_trading_visualization.png - Detailed trading behavior")
print("  2. agent_statistics.png - Performance statistics")
print("\nKey Insights:")
print(f"  - Agent took {episode_stats['total_trades']} trades")
print(f"  - Preferred action: {action_names[np.argmax(q_means)]} (highest Q-value)")
print(f"  - Total return: {episode_stats['total_return']:.2%}")
print(f"  - Sharpe ratio: {episode_stats['sharpe_ratio']:.2f}")
print("\nTo view training progress in TensorBoard:")
print("  tensorboard --logdir=./logs/timing_agent")
print("  Open: http://localhost:6006")
print("="*70)
