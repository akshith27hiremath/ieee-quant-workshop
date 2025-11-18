"""
Quick comparison between conservative (SPY) and aggressive (QQQ) agents.
Shows why the aggressive agent is better for demos!
"""
import sys
sys.path.append('..')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

print("="*70)
print("📊 AGENT COMPARISON: Conservative vs Aggressive")
print("="*70)

# Configuration comparison
comparison_data = {
    'Setting': [
        'Asset',
        'Commission',
        'Slippage',
        'Reward Type',
        'Learning Rate',
        'Network Size',
        'Final Exploration',
        'Training Steps',
        'Expected Trades',
        'Demo Value'
    ],
    'Conservative (SPY)': [
        'SPY',
        '10 bps',
        '5 bps',
        'Sharpe Ratio',
        '0.0001',
        '128x128',
        '0.05',
        '30,000',
        '1-3 trades',
        'Low 😴'
    ],
    'Aggressive (QQQ)': [
        'QQQ',
        '2 bps (5x lower!)',
        '1 bps (5x lower!)',
        'PnL (immediate)',
        '0.0005 (5x higher)',
        '256x256 (4x params)',
        '0.1 (2x higher)',
        '40,000',
        '10-30+ trades',
        'High 🚀'
    ]
}

df = pd.DataFrame(comparison_data)

print("\n" + df.to_string(index=False))

print("\n" + "="*70)
print("💡 KEY DIFFERENCES")
print("="*70)
print("""
1. **Asset Volatility**
   Conservative: SPY (S&P 500) - stable, less volatile
   Aggressive: QQQ (Nasdaq) - tech-heavy, more volatile
   → More volatility = More trading opportunities

2. **Transaction Costs**
   Conservative: 15 bps total (10 commission + 5 slippage)
   Aggressive: 3 bps total (2 commission + 1 slippage)
   → Lower costs = Trading is more profitable = Agent trades more

3. **Reward Function**
   Conservative: Sharpe Ratio (risk-adjusted returns over time)
   Aggressive: PnL (direct profit/loss)
   → PnL gives immediate feedback, encourages active trading

4. **Learning Dynamics**
   Conservative: Slow learning, conservative exploration
   Aggressive: Fast learning, more exploration
   → Agent tries more strategies and adapts quicker

5. **Network Capacity**
   Conservative: 128x128 = 16,384 + 128 = 16,512 parameters per layer
   Aggressive: 256x256 = 65,536 + 256 = 65,792 parameters per layer
   → Larger network can learn more complex trading patterns
""")

print("="*70)
print("🎯 DEMONSTRATION VALUE")
print("="*70)
print("""
Conservative Agent:
  ✗ Makes 1-3 trades total
  ✗ Usually just buys and holds
  ✗ Boring to watch
  ✗ Doesn't showcase RL capabilities well

Aggressive Agent:
  ✓ Makes 10-30+ trades
  ✓ Actively switches positions
  ✓ Responds to market conditions
  ✓ Shows Q-value evolution
  ✓ Perfect for live demonstrations!
""")

print("="*70)
print("📈 RECOMMENDATION")
print("="*70)
print("""
For IEEE Workshop Demonstrations:
  → Use the AGGRESSIVE agent in rl_demo_aggressive/
  → Shows actual RL decision-making
  → More engaging for audience
  → Better visualization of learned strategy

For Research/Real Trading:
  → Use the CONSERVATIVE agent in rl_demonstration/
  → More realistic transaction costs
  → Risk-adjusted performance
  → Better for actual deployment
""")

print("\n" + "="*70)
print("✨ To run aggressive demo:")
print("="*70)
print("""
cd rl_demo_aggressive
jupyter notebook 01_data_collection.ipynb  # Download QQQ data
jupyter notebook 02_train_agent.ipynb      # Train aggressive agent
python predict_live.py                     # Test on fresh data!
""")

print("="*70)
