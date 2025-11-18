# 🔥 Aggressive DQN Trading Agent - Demo

**An RL agent tuned for FREQUENT trading and impressive demonstrations!**

---

## 🎯 Why This Demo?

The standard conservative agent only makes **1-2 trades** because:
- High transaction costs discourage trading
- Sharpe reward promotes holding
- SPY is less volatile

This **aggressive demo** solves that with:
- ✅ **Lower transaction costs** (2 bps vs 10 bps)
- ✅ **PnL reward** (immediate profit/loss feedback)
- ✅ **QQQ** (Nasdaq - way more volatile!)
- ✅ **More exploration** (tries different strategies)
- ✅ **Larger network** (256x256 vs 128x128)
- ✅ **Live prediction on fresh data** (see what it does RIGHT NOW)

**Result: An agent that actually trades and is fun to watch!** 🚀

---

## 📁 Files

```
rl_demo_aggressive/
├── 01_data_collection.ipynb     # Download QQQ + calculate features
├── 02_train_agent.ipynb         # Train aggressive agent (with live plots)
├── train_agent_aggressive.py    # Training script (standalone)
├── predict_live.py              # 🔮 Test on fresh data from Yahoo Finance
├── demo_data/                   # Processed data
├── demo_models/                 # Trained models
├── demo_logs/                   # TensorBoard logs
└── demo_outputs/                # Visualizations
```

---

## 🚀 Quick Start

### **Step 1: Data Collection**

Run notebook `01_data_collection.ipynb` or:

```bash
cd rl_demo_aggressive
jupyter notebook 01_data_collection.ipynb
```

This downloads QQQ data (2010-2024) and calculates all technical indicators.

### **Step 2: Train Aggressive Agent**

**Option A: Notebook (recommended)**
```bash
jupyter notebook 02_train_agent.ipynb
```

**Option B: Python script**
```bash
python train_agent_aggressive.py
```

**What happens:**
- Trains for 40,000 timesteps (~20-40 minutes)
- Shows **live plots** updating every 1,500 steps
- Logs to TensorBoard: `tensorboard --logdir=demo_logs`
- Saves best model to `demo_models/aggressive_agent/`

### **Step 3: Test on Fresh Data! 🔮**

This is the **coolest part** - test your trained agent on brand new data:

```bash
python predict_live.py
```

**What it does:**
1. Downloads the **latest 90 days** of QQQ from Yahoo Finance
2. Calculates all features
3. Runs your trained agent on it
4. Shows you **exactly what it would do RIGHT NOW**!

**Output:**
- `demo_outputs/live_predictions.png` - Comprehensive visualization
- Shows: positions, Q-values, actions, portfolio value
- Perfect for demonstrations!

---

## 🔥 Aggressive Settings Explained

### Transaction Costs
```python
'commission_bps': 2  # Was 10 (5x lower!)
'slippage_bps': 1    # Was 5 (5x lower!)
```
**Why:** Lower costs = trading is more profitable = agent trades more

### Reward Function
```python
'reward_type': 'pnl'  # Was 'sharpe'
```
**Why:** Direct profit/loss gives immediate feedback vs Sharpe which requires stable returns over time

### Learning Parameters
```python
learning_rate=0.0005    # Was 0.0001 (5x higher)
exploration_final=0.1   # Was 0.05 (2x higher)
net_arch=[256, 256]     # Was [128, 128] (4x parameters)
```
**Why:** Faster learning, more exploration, bigger network = adapts quicker to trading opportunities

### Asset Choice
- **QQQ** (Nasdaq-100) instead of SPY (S&P 500)
- More volatile = more trading opportunities
- Tech-heavy = stronger trends

---

## 📊 Expected Results

### Conservative Agent (SPY + High Costs)
- Makes **1-3 trades** total
- Usually just buys and holds
- Boring for demos 😴

### Aggressive Agent (QQQ + Low Costs)
- Makes **10-30+ trades**
- Actively switches between Long/Short/Hold
- Responds to market conditions
- Much better for demonstrations! 🎉

---

## 🎨 Visualizations

### During Training (`02_train_agent.ipynb`)
- Episode rewards (live)
- Exploration rate decay
- Episode lengths
- Training statistics

### TensorBoard
```bash
tensorboard --logdir=demo_logs
```
Open: http://localhost:6006
- Detailed training metrics
- Loss curves
- Evaluation rewards

### Live Predictions (`predict_live.py`)
Creates `demo_outputs/live_predictions.png` with:
1. **Price with positions** - See when agent goes Long/Short
2. **Actions timeline** - Color-coded actions
3. **Q-values** - Agent's "confidence" for each action
4. **Portfolio value** - Track performance
5. **Action distribution** - How often it trades

---

## 🎓 Comparison: Conservative vs Aggressive

| Setting | Conservative | Aggressive |
|---------|-------------|-----------|
| **Asset** | SPY | QQQ |
| **Commission** | 10 bps | 2 bps |
| **Slippage** | 5 bps | 1 bps |
| **Reward** | Sharpe | PnL |
| **Learning Rate** | 0.0001 | 0.0005 |
| **Network** | 128x128 | 256x256 |
| **Exploration** | 0.05 final | 0.1 final |
| **Result** | 1-3 trades | 10-30+ trades |
| **Demo Value** | Low 😴 | High 🚀 |

---

## 🛠️ Customization

### Try Different Assets
Edit `predict_live.py`:
```python
ticker = 'QQQ'  # Try: 'TQQQ', 'SQQQ', 'ARKK', 'TSLA'
```

### Adjust Aggressiveness
Edit `train_agent_aggressive.py`:
```python
'commission_bps': 1,  # Even lower!
'slippage_bps': 0.5,  # Even lower!
```

### Change Training Duration
```python
total_timesteps=60000  # More training
```

---

## 📈 TensorBoard Metrics

```bash
tensorboard --logdir=demo_logs
```

**What to watch:**
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `train/loss` - Training loss (should decrease)
- `eval/mean_reward` - Evaluation reward (should increase)
- `train/exploration_rate` - Epsilon (should decay to 0.1)

---

## 🐛 Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're running from rl_demo_aggressive/ directory
cd rl_demo_aggressive
python predict_live.py
```

### "Model not found"
```bash
# Train first!
python train_agent_aggressive.py
```

### "Out of memory"
Reduce batch size in `train_agent_aggressive.py`:
```python
batch_size=32  # Was 64
```

### Training is slow
- Use GPU/XPU if available
- Reduce `total_timesteps`
- Reduce network size: `net_arch=[128, 128]`

---

## 🎯 Demo Workflow

**Perfect 5-minute demo:**

1. **Show the problem** (1 min)
   - Conservative agent only makes 1 trade
   - Not impressive for demos

2. **Show the solution** (1 min)
   - Open `train_agent_aggressive.py`
   - Highlight aggressive settings

3. **Show training** (1 min)
   - Open TensorBoard
   - Show live plots updating
   - Point out increasing rewards

4. **Show live predictions** (2 min)
   - Run `python predict_live.py`
   - Show fresh data being downloaded
   - Display visualization
   - Explain Q-values, positions, actions

**Boom! Impressive demo complete.** 🎉

---

## 📝 Notes

- **Transaction costs:** In reality, costs are higher. We lower them to encourage trading for demo purposes.
- **Overfitting:** Agent is trained on limited data. Real trading would need more validation.
- **Risk:** This is for education/demonstration only. NOT financial advice!
- **Market conditions:** Agent learns from past data. Future performance may differ.

---

## 🚀 Next Steps

1. Run all 3 steps above
2. Experiment with different assets (TQQQ for 3x leverage!)
3. Try different hyperparameters
4. Compare aggressive vs conservative agents side-by-side
5. Test on your own custom indicators

---

## 💡 Tips for Best Results

1. **Use volatile assets** - QQQ, TQQQ, individual stocks
2. **Lower transaction costs** - Encourages trading
3. **PnL reward** - More immediate than Sharpe
4. **Train longer** - 50k+ timesteps
5. **Test on fresh data** - Shows it's not just memorizing

---

**Happy Trading! 📈🤖**
