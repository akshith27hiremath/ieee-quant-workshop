"""
Live Prediction Script - Apply trained model to fresh/unseen data.

Shows what the agent would do RIGHT NOW on new market data.
Perfect for demonstrations!
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
import yfinance as yf
import ta
from datetime import datetime, timedelta

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)


def download_and_prepare_fresh_data(ticker='QQQ', days_back=90):
    """Download fresh data from Yahoo Finance and prepare features."""
    print(f"📥 Downloading fresh {ticker} data (last {days_back} days)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 250)  # Extra for indicators

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Fix MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = data.columns.str.lower()
    data['ticker'] = ticker

    print(f"✓ Downloaded {len(data)} days: {data.index[0].date()} to {data.index[-1].date()}")

    # Calculate all features
    print("🔧 Calculating technical indicators...")

    # Returns
    for period in [1, 3, 5, 10, 20]:
        data[f'return_{period}d'] = data['close'].pct_change(period)

    # RSI
    data['rsi'] = ta.momentum.rsi(data['close'], window=14)
    data['rsi_norm'] = (data['rsi'] - 50) / 50

    # MACD
    macd = ta.trend.MACD(data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    data['macd_norm'] = data['macd'] / data['close']

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()

    # Moving Averages
    data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
    data['price_to_sma_50'] = (data['close'] / data['sma_50']) - 1
    data['sma_200'] = ta.trend.sma_indicator(data['close'], window=200)
    data['price_to_sma_200'] = (data['close'] / data['sma_200']) - 1
    data['sma_crossover'] = (data['sma_50'] > data['sma_200']).astype(int)

    data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
    data['price_to_ema_12'] = (data['close'] / data['ema_12']) - 1
    data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
    data['price_to_ema_26'] = (data['close'] / data['ema_26']) - 1

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['close'])
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_mid'] = bb.bollinger_mavg()
    data['bb_percent'] = bb.bollinger_pband()
    data['bb_width'] = bb.bollinger_wband()

    # ATR
    data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    data['atr_pct'] = data['atr'] / data['close']

    # Volume
    data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
    data['obv_sma'] = ta.trend.sma_indicator(data['obv'], window=20)
    data['obv_std'] = data['obv'].rolling(window=20).std()
    data['obv_norm'] = (data['obv'] - data['obv_sma']) / data['obv_std']

    data['volume_sma'] = ta.trend.sma_indicator(data['volume'], window=20)
    data['volume_ratio'] = data['volume'] / data['volume_sma']

    # Clean
    data_clean = data.dropna()
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()

    # Keep only last N days
    data_clean = data_clean.tail(days_back)

    print(f"✓ Prepared {len(data_clean)} days of fresh data with features")

    return data_clean


def predict_on_fresh_data(model, data, features, env_config):
    """Run model on fresh data and collect predictions."""
    print("\n🤖 Running agent on fresh data...")

    # Create environment
    env = TimingEnv(data, env_config, features)

    # Collect predictions
    obs, info = env.reset()
    done = False
    truncated = False

    timestamps = []
    actions_predicted = []
    q_values_history = []
    prices = []
    positions = []
    portfolio_values = []

    step = 0
    while not (done or truncated):
        # Get Q-values
        q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]

        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Store
        timestamps.append(data.index[env.current_step])
        actions_predicted.append(int(action))
        q_values_history.append(q_values)
        prices.append(env._get_current_price())
        positions.append(env.position)
        portfolio_values.append(env.portfolio_value)

        # Step
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    print(f"✓ Generated {step} predictions")

    return {
        'timestamps': timestamps,
        'actions': np.array(actions_predicted),
        'q_values': np.array(q_values_history),
        'prices': np.array(prices),
        'positions': np.array(positions),
        'portfolio_values': np.array(portfolio_values),
        'final_value': info['portfolio_value'],
        'final_position': env.position
    }


def visualize_live_predictions(result, data, ticker='QQQ'):
    """Create comprehensive visualization of predictions."""
    print("\n📊 Creating visualization...")

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 1, hspace=0.4)

    action_labels = {0: 'Hold', 1: 'Long', 2: 'Short'}
    action_colors = {0: 'gray', 1: 'green', 2: 'red'}

    timestamps = result['timestamps']

    # 1. Price with positions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(timestamps, result['prices'], 'k-', linewidth=2, label=f'{ticker} Price')

    # Color background by position
    for i in range(len(timestamps)-1):
        if result['positions'][i] == 1:  # Long
            ax1.axvspan(timestamps[i], timestamps[i+1], alpha=0.2, color='green')
        elif result['positions'][i] == -1:  # Short
            ax1.axvspan(timestamps[i], timestamps[i+1], alpha=0.2, color='red')

    # Mark action changes
    action_changes = np.where(np.diff(result['actions']) != 0)[0] + 1
    for idx in action_changes:
        action = result['actions'][idx]
        color = action_colors[action]
        marker = '^' if action == 1 else ('v' if action == 2 else 'o')
        ax1.scatter(timestamps[idx], result['prices'][idx], c=color, marker=marker,
                   s=200, edgecolors='black', linewidths=2, zorder=5)

    ax1.set_title(f'{ticker} Price with Agent Positions (Fresh Data)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Actions timeline
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(len(timestamps)-1):
        action = result['actions'][i]
        ax2.axvspan(timestamps[i], timestamps[i+1], alpha=0.6,
                   color=action_colors[action], label=action_labels[action] if i == 0 or action != result['actions'][i-1] else "")
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel('Action', fontsize=11)
    ax2.set_yticks([])
    ax2.set_title('Agent Actions Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Remove duplicate labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left')

    # 3. Q-values
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(timestamps, result['q_values'][:, 0], 'gray', label='Q(Hold)', linewidth=1.5, alpha=0.7)
    ax3.plot(timestamps, result['q_values'][:, 1], 'green', label='Q(Long)', linewidth=1.5, alpha=0.7)
    ax3.plot(timestamps, result['q_values'][:, 2], 'red', label='Q(Short)', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Q-Value', fontsize=11)
    ax3.set_title('Q-Values (Agent\'s Expected Future Returns)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. Portfolio value
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(timestamps, result['portfolio_values'], 'b-', linewidth=2, label='Portfolio Value')
    ax4.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    final_return = (result['final_value'] - 100000) / 100000 * 100
    ax4.set_ylabel('Value ($)', fontsize=11)
    ax4.set_title(f'Portfolio Value (Return: {final_return:+.2f}%)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. Action distribution
    ax5 = fig.add_subplot(gs[4, 0])
    action_counts = pd.Series(result['actions']).value_counts().sort_index()
    colors = [action_colors[i] for i in action_counts.index]
    bars = ax5.bar([action_labels[i] for i in action_counts.index],
                   action_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add counts on bars
    for bar, count in zip(bars, action_counts.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}\n({count/len(result["actions"])*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax5.set_xlabel('Date', fontsize=11)

    plt.suptitle(f'🤖 Live Agent Predictions on Fresh {ticker} Data',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('demo_outputs/live_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: demo_outputs/live_predictions.png")
    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("📈 LIVE PREDICTION SUMMARY")
    print("="*70)
    print(f"Period: {timestamps[0].date()} to {timestamps[-1].date()}")
    print(f"Total steps: {len(timestamps)}")
    print(f"\nFinal Position: {['Hold/Cash', 'Long', 'Short'][result['final_position']+1]}")
    print(f"Final Portfolio Value: ${result['final_value']:,.2f}")
    print(f"Total Return: {final_return:+.2f}%")
    print(f"\nAction Distribution:")
    for action_idx in sorted(action_counts.index):
        count = action_counts[action_idx]
        pct = count / len(result['actions']) * 100
        print(f"  {action_labels[action_idx]}: {count} times ({pct:.1f}%)")
    print(f"\nTotal Action Changes: {len(action_changes)}")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("🔮 LIVE PREDICTION MODE - Testing on Fresh Data")
    print("="*70)

    # Load trained model
    print("\n📦 Loading trained agent...")
    model_path = Path('demo_models/aggressive_agent/best_model')
    if not model_path.with_suffix('.zip').exists():
        print("❌ ERROR: No trained model found!")
        print("   Run training first: python train_agent_aggressive.py")
        sys.exit(1)

    model = DQN.load(str(model_path))
    print(f"✓ Loaded model from: {model_path}.zip")

    # Download fresh data
    ticker = 'QQQ'
    fresh_data = download_and_prepare_fresh_data(ticker=ticker, days_back=90)

    # Features
    features = [
        'return_1d', 'return_5d', 'return_10d',
        'rsi', 'rsi_norm', 'macd', 'macd_signal', 'macd_diff',
        'sma_50', 'sma_200', 'sma_crossover', 'ema_12', 'ema_26',
        'bb_high', 'bb_low', 'bb_width', 'bb_percent', 'atr', 'atr_pct',
        'volume_ratio', 'obv'
    ]

    # Environment config (same as training)
    env_config = {
        'initial_cash': 100000,
        'reward_type': 'pnl',
        'transaction_costs': {'commission_bps': 2, 'slippage_bps': 1},
        'position_sizing': 'full',
    }

    # Predict
    result = predict_on_fresh_data(model, fresh_data, features, env_config)

    # Visualize
    visualize_live_predictions(result, fresh_data, ticker=ticker)

    print("\n✨ Done! Check demo_outputs/live_predictions.png")
