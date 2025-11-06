# Intel XPU Configuration and Fixes

**Date**: 2025-11-05
**Status**: ✅ Complete

---

## Issues Fixed

### 1. Missing TensorBoard ✅

**Error**:
```
ImportError: Trying to log data to tensorboard but tensorboard is not installed.
```

**Solution**:
```bash
pip install tensorboard
```

**Files Updated**:
- `requirements.txt` - Added `tensorboard>=2.14.0`

---

### 2. XPU Device Configuration ✅

**Requirement**: Ensure PyTorch uses Intel XPU (Arc Graphics) for hardware acceleration

**Verification**:
```python
import torch
print(torch.__version__)           # 2.8.0+xpu
print(torch.xpu.is_available())    # True
print(torch.xpu.device_count())    # 1
print(torch.xpu.get_device_name(0))  # Intel(R) Arc(TM) Graphics
```

**Files Updated**:
1. `src/agents/timing_agent.py`
   - Added device detection in `__init__`
   - Automatically uses XPU if available, falls back to CPU
   - Passes device to DQN agent

2. `notebooks/04_timing_agent_training.ipynb`
   - Added XPU detection in setup cell
   - Displays device being used
   - Passes device to DQN agent

---

## Device Selection Logic

```python
# In TimingAgentTrainer.__init__()
if torch.xpu.is_available():
    self.device = 'xpu'
    logger.info(f"Using Intel XPU device: {torch.xpu.get_device_name(0)}")
else:
    self.device = 'cpu'
    logger.info("XPU not available, using CPU")

# In DQN agent creation
model = DQN(
    policy='MlpPolicy',
    env=train_env,
    device=self.device,  # 'xpu' or 'cpu'
    ...
)
```

---

## Verification Commands

### Check PyTorch Installation
```bash
python -c "import torch; print('Version:', torch.__version__); print('XPU:', torch.xpu.is_available())"
```

**Expected Output**:
```
Version: 2.8.0+xpu
XPU: True
```

### Check TensorBoard Installation
```bash
python -c "import tensorboard; print('TensorBoard installed')"
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir=./logs/timing_agent
# Open browser to http://localhost:6006
```

---

## Performance Expectations

### CPU Training
- **Speed**: ~10-20 steps/sec
- **Time**: 15-30 minutes for 100k timesteps
- **Memory**: 1-2 GB RAM

### XPU Training (Intel Arc Graphics)
- **Speed**: ~50-100 steps/sec (3-5x faster)
- **Time**: 5-10 minutes for 100k timesteps
- **Memory**: 1-2 GB VRAM

---

## System Info

```
Environment: rlquant (conda)
Python: 3.10.18
PyTorch: 2.8.0+xpu
Device: Intel(R) Arc(TM) Graphics
stable-baselines3: 2.3.2
tensorboard: 2.20.0
```

---

## Updated Files Summary

```
✅ requirements.txt
   - Added tensorboard>=2.14.0

✅ src/agents/timing_agent.py
   - Import torch
   - Device detection in __init__()
   - Pass device to DQN agent
   - Log device being used

✅ notebooks/04_timing_agent_training.ipynb
   - Import torch in setup cell
   - Detect and display XPU availability
   - Pass device to DQN agent
   - Show device in output
```

---

## Testing the Setup

Run this in a notebook cell:

```python
import torch
from stable_baselines3 import DQN
from gymnasium import spaces
import numpy as np

# Check XPU
print(f"XPU Available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU Device: {torch.xpu.get_device_name(0)}")

# Test simple environment
env = gym.make('CartPole-v1')

# Create DQN on XPU
model = DQN('MlpPolicy', env, device='xpu', verbose=1)

# Quick training test (10k steps)
model.learn(total_timesteps=10000)

print("✓ XPU training works!")
```

---

## Troubleshooting

### If XPU is not detected:

1. **Check drivers**:
   ```bash
   # Windows: Intel Arc Graphics driver should be installed
   # Check Device Manager > Display adapters
   ```

2. **Reinstall PyTorch XPU**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
   ```

3. **Check environment**:
   ```bash
   conda activate rlquant
   python -c "import torch; print(torch.__version__)"
   # Should show: 2.8.0+xpu
   ```

### If TensorBoard doesn't start:

```bash
# Kill any existing TensorBoard processes
taskkill /F /IM tensorboard.exe  # Windows

# Start with specific port
tensorboard --logdir=./logs/timing_agent --port=6007
```

---

## Next Steps

✅ **All fixes applied - ready to train!**

Run the training notebook:
```bash
jupyter lab
# Open: 04_timing_agent_training.ipynb
# Select kernel: Python (rlquant)
# Run all cells
```

Expected output in first cell:
```
✓ Using Intel XPU: Intel(R) Arc(TM) Graphics
✓ Imports successful
```

---

**Status**: Ready for training with XPU acceleration! 🚀
