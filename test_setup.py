"""
Quick setup verification script.
Tests that all core components are working correctly.
"""

import sys
from pathlib import Path

print("="*60)
print("AlphaAgent - Setup Verification")
print("="*60)
print()

# Test 1: Imports
print("1. Testing imports...")
try:
    import torch
    import gymnasium
    import stable_baselines3
    import pandas as pd
    import numpy as np
    from src.utils.config import ConfigLoader
    from src.data.acquisition import DataAcquisition
    from src.data.preprocessing import DataPreprocessor
    from src.data.features import FeatureEngineer
    from src.data.splitter import WalkForwardSplitter
    print("   [OK] All imports successful")
except Exception as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: XPU Check
print()
print("2. Checking hardware acceleration...")
try:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"   [OK] Intel XPU available: {torch.xpu.get_device_name(0)}")
    else:
        print("   [INFO] CPU only (no XPU detected)")
except:
    print("   [INFO] CPU only")

# Test 3: Configuration
print()
print("3. Testing configuration system...")
try:
    config_loader = ConfigLoader()
    data_config = config_loader.load('data_config')
    feature_config = config_loader.load('feature_config')
    cv_config = config_loader.load('cv_config')
    env_config = config_loader.load('env_config')
    print(f"   [OK] Loaded 4 configuration files")
    print(f"   [OK] Asset universe: {data_config['portfolio_agent']['tickers']}")
except Exception as e:
    print(f"   [FAIL] Configuration error: {e}")
    sys.exit(1)

# Test 4: Data Directory
print()
print("4. Checking project structure...")
required_dirs = ['config', 'src', 'data', 'notebooks', 'tests']
missing_dirs = []
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   [OK] {dir_name}/ exists")
    else:
        print(f"   [MISSING] {dir_name}/ missing")
        missing_dirs.append(dir_name)

if missing_dirs:
    print(f"   Warning: Missing directories: {missing_dirs}")

# Test 5: Python modules
print()
print("5. Checking Python modules...")
src_modules = list(Path('src').rglob('*.py'))
print(f"   [OK] Found {len(src_modules)} Python files in src/")

# Test 6: Feature Engineering Test
print()
print("6. Testing feature engineering (minimal)...")
try:
    # Create small test dataset
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    test_data = pd.DataFrame({
        'close': 100 + pd.Series(range(300)) * 0.1,
        'open': 100 + pd.Series(range(300)) * 0.1,
        'high': 101 + pd.Series(range(300)) * 0.1,
        'low': 99 + pd.Series(range(300)) * 0.1,
        'volume': 1000000
    }, index=dates)

    feature_eng = FeatureEngineer(feature_config)
    featured_data = feature_eng.engineer_features(test_data)

    feature_names = feature_eng.get_feature_names(featured_data)
    print(f"   [OK] Generated {len(feature_names)} features")
    print(f"   [OK] Sample features: {feature_names[:5]}")
except Exception as e:
    print(f"   [FAIL] Feature engineering error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print()
print("="*60)
print("Setup Verification Complete!")
print("="*60)
print()
print("Next steps:")
print("1. Run full tests: pytest tests/ -v")
print("2. Explore notebooks: jupyter notebook notebooks/")
print("3. Download data: See notebooks/01_data_exploration.ipynb")
print()
print("Environment is ready for Phase 2 implementation!")
