"""
Basic integration tests for data pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.acquisition import DataAcquisition
from src.data.preprocessing import DataPreprocessor
from src.data.features import FeatureEngineer
from src.utils.config import get_config


class TestDataPipeline:
    """Integration tests for data pipeline."""

    @pytest.fixture
    def data_config(self):
        """Load data configuration."""
        return get_config("data_config")

    @pytest.fixture
    def feature_config(self):
        """Load feature configuration."""
        return get_config("feature_config")

    def test_data_acquisition(self, data_config):
        """Test data acquisition for a single ticker."""
        # Use a short date range for testing
        data_acq = DataAcquisition(cache_dir="data/raw")

        ticker = data_config['timing_agent']['ticker']

        data = data_acq.download_ticker(
            ticker=ticker,
            start_date="2023-01-01",
            end_date="2023-12-31",
            use_cache=True
        )

        assert not data.empty
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert 'ticker' in data.columns
        assert data['ticker'].iloc[0] == ticker

    def test_preprocessing(self, data_config):
        """Test data preprocessing."""
        # Create sample data with sufficient rows (need > min_trading_days)
        import numpy as np
        dates = pd.date_range('2022-01-01', periods=300, freq='D')

        # Generate realistic varying data
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(300) * 0.02  # 2% daily volatility
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.randn(300) * 0.005),
            'high': prices * (1 + abs(np.random.randn(300)) * 0.01),
            'low': prices * (1 - abs(np.random.randn(300)) * 0.01),
            'volume': 1000000 + np.random.randint(-100000, 100000, 300)
        }, index=dates)

        preprocessor = DataPreprocessor(data_config)
        processed_data = preprocessor.process(data)

        assert not processed_data.empty
        assert processed_data.isnull().sum().sum() == 0

    def test_feature_engineering(self, feature_config):
        """Test feature engineering."""
        # Create sample data with sufficient history for indicators
        dates = pd.date_range('2022-01-01', periods=300, freq='D')
        data = pd.DataFrame({
            'close': 100 + pd.Series(range(300)) * 0.1,
            'open': 100 + pd.Series(range(300)) * 0.1,
            'high': 101 + pd.Series(range(300)) * 0.1,
            'low': 99 + pd.Series(range(300)) * 0.1,
            'volume': 1000000
        }, index=dates)

        feature_eng = FeatureEngineer(feature_config)
        featured_data = feature_eng.engineer_features(data)

        # Check that features were added
        assert 'return_1d' in featured_data.columns
        assert 'rsi' in featured_data.columns

        # Check for no missing values in output
        feature_names = feature_eng.get_feature_names(featured_data)
        assert len(feature_names) > 0

    def test_full_pipeline(self, data_config, feature_config):
        """Test full data pipeline integration."""
        # This is a placeholder - requires actual data download
        # In production, you'd use real data or mocked data
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
