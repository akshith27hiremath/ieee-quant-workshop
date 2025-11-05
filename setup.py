"""Setup script for AlphaAgent package."""

from setuptools import setup, find_packages

setup(
    name="alphaagent",
    version="0.1.0",
    description="Reinforcement Learning Trading Agents",
    author="IEEE ML Workshop",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "polars>=0.18.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "yfinance>=0.2.28",
        "pandas-ta>=0.3.14b",
        "quantstats>=0.0.62",
        "exchange-calendars>=4.2.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ]
    },
)
