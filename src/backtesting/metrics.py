"""
Performance metrics and analytics for backtesting.
"""

import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Calculate and visualize trading performance metrics."""

    def __init__(
        self,
        results: Dict[str, Any],
        benchmark_results: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            results: Backtest results from BacktestEngine
            benchmark_results: Optional benchmark strategy results
        """
        self.results = results
        self.benchmark_results = benchmark_results

        # Convert to returns
        self.returns = self._calculate_returns()
        if benchmark_results:
            self.benchmark_returns = self._calculate_benchmark_returns()
        else:
            self.benchmark_returns = None

    def _calculate_returns(self) -> pd.Series:
        """
        Calculate returns from portfolio values.

        Returns:
            Series of returns
        """
        portfolio_values = pd.Series(self.results['portfolio_values'])
        returns = portfolio_values.pct_change().dropna()

        # Set dates as index if available
        if 'dates' in self.results and len(self.results['dates']) > 1:
            try:
                dates = pd.to_datetime(self.results['dates'][1:])  # Skip first (NaN return)
                if len(dates) == len(returns):
                    returns.index = dates
            except:
                pass

        return returns

    def _calculate_benchmark_returns(self) -> pd.Series:
        """
        Calculate benchmark returns.

        Returns:
            Series of benchmark returns
        """
        portfolio_values = pd.Series(self.benchmark_results['portfolio_values'])
        returns = portfolio_values.pct_change().dropna()

        if 'dates' in self.benchmark_results and len(self.benchmark_results['dates']) > 1:
            try:
                dates = pd.to_datetime(self.benchmark_results['dates'][1:])
                if len(dates) == len(returns):
                    returns.index = dates
            except:
                pass

        return returns

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, float]:
        """
        Generate comprehensive performance report.

        Args:
            output_path: Optional path to save HTML report

        Returns:
            Dictionary of key metrics
        """
        logger.info("Generating performance report")

        metrics = {}

        # Total Return
        initial_value = self.results['portfolio_values'][0]
        final_value = self.results['portfolio_values'][-1]
        metrics['total_return'] = (final_value - initial_value) / initial_value

        # Annualized Return
        n_days = len(self.returns)
        years = n_days / 252
        if years > 0:
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/years) - 1
        else:
            metrics['annualized_return'] = 0.0

        # Sharpe Ratio
        try:
            metrics['sharpe_ratio'] = qs.stats.sharpe(self.returns)
        except:
            metrics['sharpe_ratio'] = 0.0

        # Sortino Ratio
        try:
            metrics['sortino_ratio'] = qs.stats.sortino(self.returns)
        except:
            metrics['sortino_ratio'] = 0.0

        # Calmar Ratio
        try:
            metrics['calmar_ratio'] = qs.stats.calmar(self.returns)
        except:
            metrics['calmar_ratio'] = 0.0

        # Max Drawdown
        try:
            metrics['max_drawdown'] = qs.stats.max_drawdown(self.returns)
        except:
            metrics['max_drawdown'] = 0.0

        # Volatility
        try:
            metrics['volatility'] = qs.stats.volatility(self.returns)
        except:
            metrics['volatility'] = self.returns.std() * np.sqrt(252)

        # CVaR (Conditional Value at Risk)
        try:
            metrics['cvar_95'] = qs.stats.cvar(self.returns)
        except:
            metrics['cvar_95'] = self.returns.quantile(0.05)

        # Win Rate
        metrics['win_rate'] = (self.returns > 0).sum() / len(self.returns)

        # Number of trades
        metrics['total_trades'] = self.results['final_stats']['total_trades']

        # Benchmark comparison (if available)
        if self.benchmark_returns is not None:
            bench_total_return = (1 + self.benchmark_returns).prod() - 1
            metrics['alpha'] = metrics['total_return'] - bench_total_return
            metrics['benchmark_return'] = bench_total_return

        # Generate HTML report using quantstats
        if output_path:
            try:
                qs.reports.html(
                    self.returns,
                    self.benchmark_returns,
                    output=output_path,
                    title="RL Trading Agent Performance"
                )
                logger.info(f"HTML report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")

        return metrics

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Plot portfolio value over time.

        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        dates = self.results['dates']
        portfolio_values = self.results['portfolio_values']

        if isinstance(dates[0], pd.Timestamp):
            ax.plot(dates, portfolio_values, label='Agent', linewidth=2)
        else:
            ax.plot(portfolio_values, label='Agent', linewidth=2)

        if self.benchmark_results:
            bench_dates = self.benchmark_results['dates']
            bench_values = self.benchmark_results['portfolio_values']

            if isinstance(bench_dates[0], pd.Timestamp):
                ax.plot(bench_dates, bench_values, label='Benchmark', linewidth=2, alpha=0.7)
            else:
                ax.plot(bench_values, label='Benchmark', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print performance summary to console."""
        metrics = self.generate_report()

        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)

        print(f"\nReturns:")
        print(f"  Total Return:       {metrics['total_return']:>8.2%}")
        print(f"  Annualized Return:  {metrics['annualized_return']:>8.2%}")

        if 'alpha' in metrics:
            print(f"  Alpha:              {metrics['alpha']:>8.2%}")
            print(f"  Benchmark Return:   {metrics['benchmark_return']:>8.2%}")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>8.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
        print(f"  Volatility (Ann.):  {metrics['volatility']:>8.2%}")
        print(f"  CVaR (95%):         {metrics['cvar_95']:>8.2%}")

        print(f"\nTrading Statistics:")
        print(f"  Total Trades:       {metrics['total_trades']:>8.0f}")
        print(f"  Win Rate:           {metrics['win_rate']:>8.2%}")

        print("="*50 + "\n")
