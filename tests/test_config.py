"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigLoader, get_config


def test_config_loader_initialization():
    """Test ConfigLoader initialization."""
    loader = ConfigLoader("config")
    assert loader.config_dir.exists()


def test_load_data_config():
    """Test loading data configuration."""
    config = get_config("data_config")

    assert 'timing_agent' in config
    assert 'portfolio_agent' in config
    assert 'benchmark' in config

    # Check timing agent config
    assert config['timing_agent']['ticker'] == 'SPY'

    # Check portfolio agent config
    assert 'tickers' in config['portfolio_agent']
    assert len(config['portfolio_agent']['tickers']) == 4


def test_load_feature_config():
    """Test loading feature configuration."""
    config = get_config("feature_config")

    assert 'price_features' in config
    assert 'momentum_indicators' in config
    assert 'trend_indicators' in config
    assert 'volatility_indicators' in config


def test_load_cv_config():
    """Test loading CV configuration."""
    config = get_config("cv_config")

    assert 'method' in config
    assert 'walk_forward' in config
    assert 'test_set' in config

    # Check walk-forward params
    assert config['walk_forward']['train_window_days'] == 1095
    assert config['walk_forward']['val_window_days'] == 180


def test_load_env_config():
    """Test loading environment configuration."""
    config = get_config("env_config")

    assert 'transaction_costs' in config
    assert 'initial_cash' in config
    assert config['initial_cash'] == 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
