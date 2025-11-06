# RL Agent Visualization Guide

**Date**: 2025-11-05

Complete guide to visualizing your RL agent's learning process and trading behavior.

---

## Quick Start

### 1. View Agent Trading Behavior (Static Images)
```bash
python visualize_agent.py
```

This generates two comprehensive visualization files:
- `agent_trading_visualization.png` - 5-panel detailed trading analysis
- `agent_statistics.png` - Summary statistics and distributions

### 2. View Training Progress (Interactive TensorBoard)
```bash
tensorboard --logdir=./logs/timing_agent
```
Then open: http://localhost:6006

---

## What Each Visualization Shows

### A. Agent Trading Visualization (`agent_trading_visualization.png`)

#### Panel 1: Portfolio Value vs SPY Price
- **Blue line**: Your agent's portfolio value over time
- **Green line**: SPY price (underlying asset)
- **Gray dashed line**: Initial capital ($100,000)

**What to look for:**
- Is portfolio value tracking SPY price? (buy-and-hold behavior)
- Is it outperforming SPY? (good timing)
- Is it underperforming? (poor timing or excessive costs)

#### Panel 2: Trading Actions
- **Green background**: Agent is LONG (holding SPY)
- **Red background**: Agent is SHORT (betting against SPY)
- **White background**: Agent is in CASH (no position)
- **Markers**:
  - ▲ Green triangle = Entering long position
  - ▼ Red triangle = Entering short position

**What to look for:**
- How often does the agent trade? (too frequent = overtrading)
- Does it stay in positions long enough?
- Are actions changing randomly or following patterns?

#### Panel 3: Rewards and Penalties
- **Purple line (thick)**: Cumulative reward over time
- **Purple line (thin)**: Step-by-step reward
- **Green shading**: Positive rewards (good actions)
- **Red shading**: Negative rewards (bad actions/penalties)

**What to look for:**
- Is cumulative reward trending up? (agent learning)
- Are there long periods of negative rewards? (agent struggling)
- Big spikes = major wins or losses

#### Panel 4: Q-Values Evolution
- **Gray line**: Q(Hold) - Expected value of holding cash
- **Green line**: Q(Long) - Expected value of buying SPY
- **Red line**: Q(Short) - Expected value of shorting SPY

**What to look for:**
- Which Q-value is highest most of the time? (agent's preference)
- Do Q-values diverge (spread out)? Means agent is confident
- Do they stay close together? Means agent is uncertain
- Are they stable or jumping around? (learning stability)

#### Panel 5: Portfolio Composition
- **Blue area**: Cash on hand
- **Orange area**: Value of current position (shares held)
- **Black line**: Total portfolio value

**What to look for:**
- Is agent mostly in cash or mostly invested?
- Does it use full capital or keep reserves?

---

### B. Agent Statistics (`agent_statistics.png`)

#### Panel 1: Action Distribution (Bar Chart)
Shows how many times each action was taken:
- **Hold**: Maintain current position
- **Long**: Buy/enter long position
- **Short**: Sell/enter short position

**What to look for:**
- Balanced distribution? Or dominated by one action?
- If "Hold" dominates, agent may have learned to avoid trading

#### Panel 2: Reward Distribution (Histogram)
Distribution of rewards across all timesteps.

**What to look for:**
- Centered around zero? (neutral learning)
- Skewed positive? (mostly good actions)
- Skewed negative? (mostly penalties)
- Wide spread? (high variance in outcomes)

#### Panel 3: Average Q-Values by Action (Bar Chart)
Average Q-value for each action type with error bars.

**What to look for:**
- Which action has highest Q-value? (agent's belief about best action)
- Large error bars? (high uncertainty)
- Negative Q-values? (agent expects losses)

#### Panel 4: Performance Summary (Text Box)
Complete episode statistics.

**Key metrics:**
- **Total Return**: Overall portfolio gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >0.5 is good)
- **Max Drawdown**: Worst peak-to-trough decline (lower is better)
- **Total Trades**: Number of position changes
- **Preferred Action**: Action with highest average Q-value

---

## TensorBoard Visualizations

### Starting TensorBoard
```bash
# Basic usage
tensorboard --logdir=./logs/timing_agent

# Specific port
tensorboard --logdir=./logs/timing_agent --port=6007

# Load faster (update every 30 seconds)
tensorboard --logdir=./logs/timing_agent --reload_interval=30
```

### What You'll See in TensorBoard

#### 1. SCALARS Tab (Most Important)

**Training Metrics:**
- `train/learning_rate` - Learning rate over time (should be constant)
- `train/loss` - DQN loss (should decrease over time)
- `train/n_updates` - Number of gradient updates

**Evaluation Metrics:**
- `eval/mean_reward` - Average reward on validation set
- `eval/mean_ep_length` - Average episode length

**Rollout Metrics:**
- `rollout/exploration_rate` - Epsilon (random action probability)
  - Starts at 1.0 (100% random)
  - Decreases to 0.05 (5% random)
- `rollout/ep_rew_mean` - Mean episode reward during training

**What to Look For:**
- **Loss decreasing**: Agent is learning
- **Loss stable**: Learning has converged
- **Loss increasing**: Training instability (bad!)
- **Eval reward increasing**: Agent improving on validation
- **Exploration rate decreasing**: Agent exploring less over time

#### 2. IMAGES Tab
If you enabled image logging, shows:
- Environment observations
- Policy visualizations

#### 3. GRAPHS Tab
Shows the neural network architecture:
- Input layer (state features)
- Hidden layers
- Output layer (Q-values for each action)

#### 4. DISTRIBUTIONS Tab
Shows weight/bias distributions in the network:
- Should be relatively stable during training
- Major shifts indicate learning dynamics

---

## Interpreting Agent Behavior

### Good Signs ✓
1. **Cumulative reward trending upward**
2. **Eval mean reward increasing over training**
3. **Loss decreasing then stabilizing**
4. **Q-values converging (not all the same)**
5. **Actions aligned with Q-values** (takes action with highest Q)
6. **Multiple trades** (not just buy-and-hold)
7. **Sharpe ratio > 0.5**
8. **Outperforming baselines**

### Warning Signs ⚠
1. **Only 1-2 trades total** → Agent avoiding risk
2. **Q-values all similar** → Agent uncertain/confused
3. **Loss exploding** (>1e7) → Training instability
4. **Negative Sharpe ratio** → Poor risk-adjusted returns
5. **All actions are "Hold"** → Agent learned to do nothing
6. **Cumulative reward flat/declining** → Not learning
7. **Max drawdown > 30%** → Too risky

### Bad Signs ✗
1. **Loss increasing continuously** → Diverging Q-values
2. **Random-looking actions** → No learned policy
3. **Portfolio value crashing** → Catastrophic strategy
4. **Sharpe ratio < -0.5** → Worse than random
5. **Q-values jumping wildly** → Unstable estimates

---

## Advanced Visualizations

### Compare Multiple Training Runs
```bash
# If you have multiple runs in different folders
tensorboard --logdir_spec=run1:./logs/timing_agent_v1,run2:./logs/timing_agent_v2
```

### Export TensorBoard Data
```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('./logs/timing_agent/DQN_1')
ea.Reload()

# Get scalar data
loss = ea.Scalars('train/loss')
rewards = ea.Scalars('eval/mean_reward')
```

---

## Creating Custom Visualizations

### Example: Plot Learning Curve
```python
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('./logs/timing_agent/DQN_1')
ea.Reload()

# Extract evaluation rewards
rewards = ea.Scalars('eval/mean_reward')
steps = [r.step for r in rewards]
values = [r.value for r in rewards]

plt.figure(figsize=(10, 6))
plt.plot(steps, values, linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Mean Evaluation Reward')
plt.title('Learning Curve')
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve.png', dpi=150)
```

---

## Troubleshooting

### TensorBoard shows "No dashboards are active"
**Cause**: No events logged yet
**Solution**: Run training first, then start TensorBoard

### Old runs showing in TensorBoard
**Cause**: Previous logs in directory
**Solution**:
```bash
# Clear old logs
rm -rf ./logs/timing_agent/*

# Or specify exact run
tensorboard --logdir=./logs/timing_agent/DQN_1
```

### Can't access TensorBoard UI
**Cause**: Port already in use
**Solution**:
```bash
# Use different port
tensorboard --logdir=./logs/timing_agent --port=6007

# Kill existing process (Windows)
taskkill /F /IM tensorboard.exe
```

### Visualizations not updating
**Cause**: TensorBoard caching
**Solution**:
- Refresh browser (Ctrl+F5)
- Restart TensorBoard
- Clear browser cache

---

## Key Questions to Answer

When analyzing your agent, ask:

### About Learning:
1. **Is the agent learning?**
   - Check: Loss decreasing, eval reward increasing

2. **Is learning stable?**
   - Check: Smooth curves, no wild jumps in loss

3. **Has it converged?**
   - Check: Metrics plateaued, no improvement in eval

### About Strategy:
4. **What is the agent doing?**
   - Check: Action distribution, Q-value preferences

5. **Is it timing the market or buy-and-hold?**
   - Check: Number of trades, position changes

6. **Is it risk-averse or aggressive?**
   - Check: Max drawdown, time in market vs cash

### About Performance:
7. **Is it profitable?**
   - Check: Total return, Sharpe ratio

8. **Does it beat baselines?**
   - Compare to Buy & Hold and SMA strategies

9. **Is the risk worth the return?**
   - Check: Sharpe ratio, max drawdown

---

## Quick Diagnostic Checklist

Run this after every training session:

```bash
# 1. Generate visualizations
python visualize_agent.py

# 2. Start TensorBoard
tensorboard --logdir=./logs/timing_agent

# 3. Check these metrics:
#    - Final eval reward > 0?
#    - Training loss < 1e6?
#    - Sharpe ratio > 0?
#    - More than 2 trades?
#    - Q-values differentiated?

# 4. Compare to baselines
python compare_baselines.py
```

---

## Next Steps

After visualizing:

### If Performance is Good ✓
- Save the model for production
- Test on out-of-sample data
- Try different market conditions
- Experiment with position sizing

### If Performance is Poor ✗
- Adjust hyperparameters (see PHASE2_RESULTS.md)
- Change reward function
- Reduce transaction costs for training
- Increase exploration
- Try different RL algorithm

### If Agent Isn't Learning
- Check feature quality
- Verify reward signal is meaningful
- Increase network capacity
- Add more training data
- Reduce state space complexity

---

## Resources

**TensorBoard Documentation**: https://www.tensorflow.org/tensorboard
**Stable-Baselines3 Logging**: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

**Generated Files Location:**
```
./agent_trading_visualization.png   - Trading behavior analysis
./agent_statistics.png              - Summary statistics
./logs/timing_agent/                - TensorBoard logs
./models/timing_agent/best/         - Best model checkpoint
```

---

**Happy visualizing!** 📊
