# Quick Start Guide - RL Demonstration

**Complete DQN Trading Agent Demonstration Package**

---

## 🚀 Run The Complete Workflow

### Prerequisites
```bash
# Make sure you're in the rlquant conda environment
conda activate rlquant

# Navigate to the demonstration folder
cd rl_demonstration
```

### Option 1: Run Notebooks in Order (Recommended)
```bash
# Start Jupyter Lab
jupyter lab

# Then open and run in order:
# 1. 01_data_collection.ipynb
# 2. 02_cross_validation.ipynb
# 3. Use the Python scripts directly (see below)
```

### Option 2: Run Python Scripts Directly
```bash
# Step 1: Prepare data (if not done)
cd ..
python -c "
import pandas as pd
data = pd.read_parquet('data/features/featured_data.parquet')
data[data['ticker'] == 'SPY'].to_parquet('rl_demonstration/demo_data/featured_data.parquet')
print('Data ready!')
"

# Step 2: Train the agent
cd rl_demonstration
python train_agent.py

# Step 3: Analyze the trained model
python analyze_model.py

# Step 4: Comprehensive evaluation
python evaluate_agent.py
```

---

## 📁 What's Included

### Notebooks
- `01_data_collection.ipynb` - Download and prepare SPY data with technical indicators
- `02_cross_validation.ipynb` - Visualize train/val/test splits

### Python Scripts (Can run standalone or in notebooks)
- `train_agent.py` - Train DQN with live visualization
- `analyze_model.py` - Analyze learned strategy and Q-values
- `evaluate_agent.py` - Compare vs baselines with comprehensive visualizations

### Output Directories
- `demo_data/` - Processed data files
- `demo_models/` - Trained model checkpoints
- `demo_logs/` - TensorBoard logs and training metrics
- `demo_outputs/` - Generated visualizations

---

## 🎯 What You'll Learn

### 1. Data Preparation (`01_data_collection.ipynb`)
- Download market data from Yahoo Finance
- Calculate 21 technical indicators
- Validate data quality
- **Output**: `demo_data/featured_data.parquet`

### 2. Cross-Validation (`02_cross_validation.ipynb`)
- Time-series data splitting
- Visualize train/val/test periods
- Verify no data leakage
- **Output**: Visualization of splits

### 3. Agent Training (`train_agent.py`)
- Build DQN agent
- Train with real-time visualization
- See rewards, Q-values, actions update live
- TensorBoard integration
- **Output**: `demo_models/dqn_agent/best_model.zip`

### 4. Strategy Analysis (`analyze_model.py`)
- Discover what strategy the agent learned
- Visualize Q-values over time
- Understand decision-making process
- Feature importance analysis
- **Output**: `demo_outputs/strategy_analysis.png`

### 5. Comprehensive Evaluation (`evaluate_agent.py`)
- Test on unseen data (2021-2024)
- Compare vs Buy & Hold and SMA Crossover
- Performance metrics and visualizations
- **Output**: `demo_outputs/comprehensive_evaluation.png`

---

## 📊 Expected Results

After running the complete workflow, you should see:

### Training
- **Duration**: ~10-15 minutes on Intel XPU/GPU, ~30 minutes on CPU
- **Best model**: Typically found around 5,000-15,000 timesteps
- **Training loss**: Should decrease and stabilize

### Performance (approximate, varies by market period)
| Strategy | Return | Sharpe | Max DD |
|----------|--------|--------|--------|
| DQN Agent | Varies | Varies | Varies |
| Buy & Hold | ~15-20% | ~0.5 | ~20% |
| SMA Cross | ~5-10% | ~0.3 | ~10% |

### Key Insights
- Agent learns market-timing strategy
- Usually learns buy-and-hold or momentum following
- Transaction costs heavily influence strategy
- Different market periods yield different results

---

## 🛠️ Customization

### Change the asset
Edit in `train_agent.py`:
```python
# Instead of SPY, try:
ticker = "AAPL"  # Apple
ticker = "QQQ"   # Nasdaq 100
ticker = "GLD"   # Gold
```

### Adjust training duration
```python
total_timesteps = 100000  # More training
total_timesteps = 20000   # Quick test
```

### Try different reward functions
```python
env_config = {
    'reward_type': 'pnl',  # Simple profit/loss
    # or 'sharpe', 'sortino', 'drawdown_aware'
}
```

### Modify transaction costs
```python
'transaction_costs': {
    'commission_bps': 5,  # Lower costs
    'slippage_bps': 2
}
```

---

## 📈 Viewing Training Progress

### TensorBoard (Real-time)
```bash
# In a separate terminal
tensorboard --logdir=demo_logs

# Open browser to: http://localhost:6006
```

You'll see:
- Training loss over time
- Evaluation rewards
- Exploration rate decay
- Episode statistics

---

## 🐛 Troubleshooting

### "demo_data/featured_data.parquet not found"
```bash
# Run notebook 01 first, or:
cd ..
python -c "
import pandas as pd
data = pd.read_parquet('data/features/featured_data.parquet')
data[data['ticker'] == 'SPY'].to_parquet('rl_demonstration/demo_data/featured_data.parquet')
"
```

### "Model not found"
```bash
# Run training first:
python train_agent.py
```

### Training is slow
```python
# In train_agent.py, reduce timesteps:
total_timesteps = 20000  # Instead of 50000
```

### Out of memory
```python
# In train_agent.py, reduce buffer:
buffer_size = 10000  # Instead of 50000
batch_size = 16      # Instead of 32
```

---

## 📝 File Structure

```
rl_demonstration/
├── README.md                      # Full documentation
├── QUICK_START.md                 # This file
├── 01_data_collection.ipynb       # Data preparation
├── 02_cross_validation.ipynb      # CV visualization
├── train_agent.py                 # Training script
├── analyze_model.py               # Strategy analysis
├── evaluate_agent.py              # Comprehensive evaluation
├── demo_data/                     # Data files
│   ├── featured_data.parquet
│   └── split_info.json
├── demo_models/dqn_agent/         # Saved models
│   ├── best_model.zip
│   └── final_model.zip
├── demo_logs/                     # TensorBoard logs
└── demo_outputs/                  # Visualizations
    ├── data_splits.png
    ├── strategy_analysis.png
    └── comprehensive_evaluation.png
```

---

## ⏱️ Time Estimates

- **Data preparation** (notebook 01): 3-5 minutes
- **CV visualization** (notebook 02): 2-3 minutes
- **Training** (train_agent.py): 10-30 minutes (depending on hardware)
- **Analysis** (analyze_model.py): 2-3 minutes
- **Evaluation** (evaluate_agent.py): 3-5 minutes

**Total**: 20-50 minutes for complete workflow

---

## 🎓 Learning Path

###  Beginner
1. Run all notebooks/scripts in order
2. Observe the visualizations
3. Understand what each metric means
4. Compare DQN vs baselines

### Intermediate
5. Modify hyperparameters and retrain
6. Try different technical indicators
7. Test on different assets
8. Experiment with reward functions

### Advanced
9. Implement new RL algorithms (PPO, SAC)
10. Add continuous actions (position sizing)
11. Multi-asset portfolio agent
12. Online learning and adaptation

---

## 📚 Additional Resources

- **Main project**: `../` (parent directory)
- **Full config**: `../config/timing_config.yaml`
- **Environment code**: `../src/environments/timing_env.py`
- **Agent code**: `../src/agents/timing_agent.py`
- **Visualization guide**: `../remarks/VISUALIZATION_GUIDE.md`

---

## 🙋 FAQ

**Q: Can I run this without GPU?**
A: Yes! It will use CPU automatically, just slower.

**Q: How do I know if training is working?**
A: Watch the live plots. Episode rewards should show some improvement over time.

**Q: Why does my agent just buy and hold?**
A: Common! Transaction costs make frequent trading expensive. Try reducing costs for training.

**Q: How do I save my results?**
A: All outputs are automatically saved to `demo_outputs/` and `demo_models/`.

**Q: Can I train on my own data?**
A: Yes! Replace `demo_data/featured_data.parquet` with your own preprocessed data.

---

**Ready to start? Run:**
```bash
jupyter lab
# Open 01_data_collection.ipynb
```

**Or jump straight to training:**
```bash
python train_agent.py
```

**Happy learning!** 🚀📈
