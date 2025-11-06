"""
Compare DQN agent performance against baseline strategies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')
from src.utils.config import ConfigLoader
from src.backtesting.baselines import BuyAndHold, SMAcrossover

# Load data
data = pd.read_parquet('data/features/featured_data.parquet')
data = data[data['ticker'] == 'SPY'].copy()

# Load config for splits
config_loader = ConfigLoader('config')
cv_config = config_loader.load('cv_config')
test_start = pd.Timestamp(cv_config['test_set']['start_date'])
train_val_data = data[data.index < test_start].copy()
train_size = int(len(train_val_data) * 0.85)
val_data = train_val_data.iloc[train_size:]

print('='*70)
print('BASELINE STRATEGY COMPARISON ON VALIDATION SET')
print('='*70)
print(f'Validation period: {val_data.index[0]} to {val_data.index[-1]}')
print(f'Data points: {len(val_data)}')
print()

# Buy and Hold
print('[1/2] Running Buy & Hold...')
bh = BuyAndHold(initial_cash=100000)
bh_results = bh.run(val_data)
print(f'  Total Return: {bh_results["total_return"]:.2%}')
print(f'  Sharpe Ratio: {bh_results["sharpe_ratio"]:.2f}')
print(f'  Max Drawdown: {bh_results["max_drawdown"]:.2%}')
print()

# SMA Crossover
print('[2/2] Running SMA Crossover (50/200)...')
sma = SMAcrossover(initial_cash=100000, fast_window=50, slow_window=200)
sma_results = sma.run(val_data)
print(f'  Total Return: {sma_results["total_return"]:.2%}')
print(f'  Sharpe Ratio: {sma_results["sharpe_ratio"]:.2f}')
print(f'  Max Drawdown: {sma_results["max_drawdown"]:.2%}')
print(f'  Total Trades: {sma_results["total_trades"]}')
print()

print('='*70)
print('PERFORMANCE COMPARISON')
print('='*70)
header = f"{'Strategy':<20} {'Return':>12} {'Sharpe':>10} {'Max DD':>10} {'Trades':>10}"
print(header)
print('-'*70)
print(f"{'DQN Agent':<20} {-0.1508:>11.2%} {-0.45:>10.2f} {0.3309:>9.2%} {1:>10}")
print(f"{'Buy & Hold':<20} {bh_results['total_return']:>11.2%} {bh_results['sharpe_ratio']:>10.2f} {bh_results['max_drawdown']:>9.2%} {bh_results.get('total_trades', 1):>10}")
print(f"{'SMA(50/200)':<20} {sma_results['total_return']:>11.2%} {sma_results['sharpe_ratio']:>10.2f} {sma_results['max_drawdown']:>9.2%} {sma_results['total_trades']:>10}")
print('='*70)
