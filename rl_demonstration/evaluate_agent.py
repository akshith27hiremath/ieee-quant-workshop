"""
Comprehensive evaluation script - used by notebook 05.
Compares DQN agent against baseline strategies.
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import DQN
from src.environments.timing_env import TimingEnv
from src.backtesting.baselines import BuyAndHold, SMAcrossover


def evaluate_dqn_agent(model, data, features, env_config, n_episodes=1):
    """
    Evaluate DQN agent on test data.
    """
    print("Evaluating DQN Agent...")

    env = TimingEnv(data, env_config, features)

    episode_results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        portfolio_history = []
        actions_history = []
        rewards_history = []
        timestamps = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            portfolio_history.append(env.portfolio_value)
            actions_history.append(int(action))
            timestamps.append(data.index[env.current_step])

            obs, reward, done, truncated, info = env.step(action)
            rewards_history.append(reward)

        stats = env.get_episode_stats()

        episode_results.append({
            'portfolio_history': portfolio_history,
            'actions_history': actions_history,
            'rewards_history': rewards_history,
            'timestamps': timestamps,
            'final_value': info['portfolio_value'],
            'total_return': stats['total_return'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown'],
            'total_trades': stats['total_trades']
        })

    print(f"  Completed {n_episodes} episode(s)")
    return episode_results[0] if n_episodes == 1 else episode_results


def compare_strategies(dqn_result, data):
    """
    Compare DQN against Buy & Hold and SMA Crossover.
    """
    print("\\nRunning baseline strategies...")

    # Buy and Hold
    print("  [1/2] Buy & Hold...")
    bh = BuyAndHold(initial_cash=100000)
    bh_result = bh.run(data)

    # SMA Crossover
    print("  [2/2] SMA Crossover...")
    sma = SMAcrossover(initial_cash=100000, fast_window=50, slow_window=200)
    sma_result = sma.run(data)

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Strategy': ['DQN Agent', 'Buy & Hold', 'SMA(50/200)'],
        'Final Value': [
            dqn_result['final_value'],
            bh_result['final_value'],
            sma_result['final_value']
        ],
        'Total Return': [
            dqn_result['total_return'],
            bh_result['total_return'],
            sma_result['total_return']
        ],
        'Sharpe Ratio': [
            dqn_result['sharpe_ratio'],
            bh_result['sharpe_ratio'],
            sma_result['sharpe_ratio']
        ],
        'Max Drawdown': [
            dqn_result['max_drawdown'],
            bh_result['max_drawdown'],
            sma_result['max_drawdown']
        ],
        'Total Trades': [
            dqn_result['total_trades'],
            bh_result.get('total_trades', 1),
            sma_result['total_trades']
        ]
    })

    return comparison, bh_result, sma_result


def create_comprehensive_visualization(dqn_result, bh_result, sma_result, data, comparison_df):
    """
    Create comprehensive evaluation visualization.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    timestamps = dqn_result['timestamps']
    dqn_portfolio = dqn_result['portfolio_history']

    # 1. Portfolio value comparison (large, top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, dqn_portfolio, label='DQN Agent', linewidth=2, color='blue')
    ax1.plot(bh_result['portfolio_history'].index, bh_result['portfolio_history'].values,
             label='Buy & Hold', linewidth=2, color='green', alpha=0.7)
    ax1.plot(sma_result['portfolio_history'].index, sma_result['portfolio_history'].values,
             label='SMA Crossover', linewidth=2, color='orange', alpha=0.7)
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Returns comparison (bar chart)
    ax2 = fig.add_subplot(gs[1, 0])
    returns = comparison_df['Total Return'].values * 100
    colors = ['blue', 'green', 'orange']
    bars = ax2.bar(comparison_df['Strategy'], returns, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Total Returns', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

    # 3. Sharpe Ratio comparison
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = comparison_df['Sharpe Ratio'].values
    bars = ax3.bar(comparison_df['Strategy'], sharpes, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sharpes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

    # 4. Max Drawdown comparison
    ax4 = fig.add_subplot(gs[1, 2])
    drawdowns = comparison_df['Max Drawdown'].values * 100
    bars = ax4.bar(comparison_df['Strategy'], drawdowns, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.invert_yaxis()
    for bar, val in zip(bars, drawdowns):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='top',
                fontweight='bold')

    # 5. Cumulative rewards (DQN only)
    ax5 = fig.add_subplot(gs[2, 0])
    cumulative_rewards = np.cumsum(dqn_result['rewards_history'])
    ax5.plot(timestamps, cumulative_rewards, color='purple', linewidth=2)
    ax5.fill_between(timestamps, cumulative_rewards, 0, alpha=0.3, color='purple')
    ax5.set_title('DQN Cumulative Rewards', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Reward', fontsize=10)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Action distribution (DQN only)
    ax6 = fig.add_subplot(gs[2, 1])
    action_counts = pd.Series(dqn_result['actions_history']).value_counts().sort_index()
    action_names = ['Hold', 'Long', 'Short']
    action_colors_bar = ['gray', 'green', 'red']
    bars = ax6.bar(range(len(action_counts)), action_counts.values,
                   color=action_colors_bar, alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(action_counts)))
    ax6.set_xticklabels([action_names[i] for i in action_counts.index])
    ax6.set_title('DQN Action Distribution', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, action_counts.values):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(val), ha='center', va='bottom', fontweight='bold')

    # 7. Performance metrics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    table_data = []
    for _, row in comparison_df.iterrows():
        table_data.append([
            row['Strategy'],
            f"${row['Final Value']:,.0f}",
            f"{row['Total Return']:.1%}",
            f"{row['Sharpe Ratio']:.2f}",
            f"{row['Max Drawdown']:.1%}",
            f"{int(row['Total Trades'])}"
        ])

    table = ax7.table(cellText=table_data,
                     colLabels=['Strategy', 'Final $', 'Return', 'Sharpe', 'DD', 'Trades'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style table header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows
    row_colors = ['#DAEEF3', '#C5E0B4', '#FFE699']
    for i in range(1, 4):
        for j in range(6):
            table[(i, j)].set_facecolor(row_colors[i-1])

    ax7.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Comprehensive Strategy Evaluation', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('demo_outputs/comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\\n✓ Saved comprehensive evaluation: demo_outputs/comprehensive_evaluation.png")


def print_evaluation_summary(comparison_df):
    """
    Print detailed evaluation summary.
    """
    print("\\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)

    print("\\n" + comparison_df.to_string(index=False))

    print("\\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Best strategy by different metrics
    best_return_idx = comparison_df['Total Return'].idxmax()
    best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
    best_dd_idx = comparison_df['Max Drawdown'].idxmax()  # Least negative

    print(f"\\nBest Total Return: {comparison_df.loc[best_return_idx, 'Strategy']}")
    print(f"  → {comparison_df.loc[best_return_idx, 'Total Return']:.2%}")

    print(f"\\nBest Sharpe Ratio: {comparison_df.loc[best_sharpe_idx, 'Strategy']}")
    print(f"  → {comparison_df.loc[best_sharpe_idx, 'Sharpe Ratio']:.2f}")

    print(f"\\nLowest Drawdown: {comparison_df.loc[best_dd_idx, 'Strategy']}")
    print(f"  → {comparison_df.loc[best_dd_idx, 'Max Drawdown']:.2%}")

    # DQN performance
    dqn_row = comparison_df[comparison_df['Strategy'] == 'DQN Agent'].iloc[0]
    bh_row = comparison_df[comparison_df['Strategy'] == 'Buy & Hold'].iloc[0]

    print(f"\\nDQN vs Buy & Hold:")
    return_diff = dqn_row['Total Return'] - bh_row['Total Return']
    print(f"  Return difference: {return_diff:+.2%}")

    if dqn_row['Total Return'] > bh_row['Total Return']:
        print(f"  ✓ DQN outperformed by {return_diff:.2%}")
    else:
        print(f"  ✗ DQN underperformed by {abs(return_diff):.2%}")

    sharpe_diff = dqn_row['Sharpe Ratio'] - bh_row['Sharpe Ratio']
    if dqn_row['Sharpe Ratio'] > bh_row['Sharpe Ratio']:
        print(f"  ✓ DQN has better risk-adjusted returns (+{sharpe_diff:.2f} Sharpe)")
    else:
        print(f"  ✗ DQN has worse risk-adjusted returns ({sharpe_diff:.2f} Sharpe)")

    print("="*80)


if __name__ == "__main__":
    # Test standalone
    print("Loading model...")
    model = DQN.load('demo_models/dqn_agent/best_model.zip')

    print("Loading data...")
    data = pd.read_parquet('../data/features/featured_data.parquet')
    data = data[data['ticker'] == 'SPY'].copy()

    test_start = pd.Timestamp('2021-01-01')
    test_data = data[data.index >= test_start].copy()

    features = [
        'return_1d', 'return_5d', 'return_10d',
        'rsi', 'rsi_norm', 'macd', 'macd_signal', 'macd_diff',
        'sma_50', 'sma_200', 'sma_crossover', 'ema_12', 'ema_26',
        'bb_high', 'bb_low', 'bb_width', 'bb_percent', 'atr', 'atr_pct',
        'volume_ratio', 'obv'
    ]

    env_config = {
        'initial_cash': 100000,
        'reward_type': 'sharpe',
        'transaction_costs': {'commission_bps': 10, 'slippage_bps': 5}
    }

    # Evaluate
    dqn_result = evaluate_dqn_agent(model, test_data, features, env_config)
    comparison_df, bh_result, sma_result = compare_strategies(dqn_result, test_data)

    # Visualize and print
    create_comprehensive_visualization(dqn_result, bh_result, sma_result, test_data, comparison_df)
    print_evaluation_summary(comparison_df)
