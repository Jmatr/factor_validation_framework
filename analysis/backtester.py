import pandas as pd
import numpy as np
from analysis.performance import PerformanceAnalyzer
from config.settings import *


class FactorBacktester:
    def __init__(self, initial_capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.performance_analyzer = PerformanceAnalyzer()

    def run_single_factor_backtest(self, factor_quantiles, forward_returns, factor_name, rebalance_freq=21):
        """Run single factor backtest"""
        # Long top quantile, short bottom quantile
        top_quantile = factor_quantiles.columns[-1]
        bottom_quantile = factor_quantiles.columns[0]

        portfolio_returns = pd.Series(index=forward_returns.index, dtype=float)
        positions = {}

        rebalance_dates = forward_returns.index[::rebalance_freq]

        for i, date in enumerate(rebalance_dates):
            if i == len(rebalance_dates) - 1:
                break

            current_date = date
            next_rebalance = rebalance_dates[i + 1]

            # Get current positions
            top_stocks = factor_quantiles.loc[current_date][
                factor_quantiles.loc[current_date] == top_quantile].dropna().index
            bottom_stocks = factor_quantiles.loc[current_date][
                factor_quantiles.loc[current_date] == bottom_quantile].dropna().index

            # Calculate returns for the period
            period_returns = forward_returns.loc[current_date:next_rebalance]
            if len(period_returns) == 0:
                continue

            # Long top, short bottom (equal weighted)
            if len(top_stocks) > 0 and len(bottom_stocks) > 0:
                top_portfolio_returns = period_returns[top_stocks].mean(axis=1)
                bottom_portfolio_returns = period_returns[bottom_stocks].mean(axis=1)
                strategy_returns = top_portfolio_returns - bottom_portfolio_returns

                # Apply transaction costs (simplified)
                if i > 0:
                    strategy_returns.iloc[0] -= self.transaction_cost * 2  # Both long and short

                portfolio_returns.loc[strategy_returns.index] = strategy_returns

        # Fill NaN with 0
        portfolio_returns = portfolio_returns.fillna(0)

        # Calculate cumulative returns and portfolio value
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = self.initial_capital * cumulative_returns

        # Calculate performance metrics
        metrics = self.performance_analyzer.calculate_performance_metrics(portfolio_returns)

        backtest_results = {
            'factor_name': factor_name,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value,
            'metrics': metrics
        }

        return backtest_results

    def run_multiple_factor_backtest(self, factors_results, forward_returns, rebalance_freq=21):
        """Run multiple factor backtests"""
        backtest_results = {}

        for result in factors_results:
            factor_name = result['name']
            factor_quantiles = result.get('quantile_returns', None)

            if factor_quantiles is not None and not factor_quantiles.empty:
                try:
                    backtest = self.run_single_factor_backtest(
                        factor_quantiles, forward_returns, factor_name, rebalance_freq
                    )
                    backtest_results[factor_name] = backtest
                except Exception as e:
                    print(f"Backtest failed for {factor_name}: {str(e)}")

        return backtest_results

    def run_equal_weight_composite(self, factors_results, forward_returns, top_n=3):
        """Run equal-weighted composite of top N factors"""
        # Select top N factors by Sharpe ratio
        factor_sharpe_ratios = {}
        for result in factors_results:
            if 'results' in result and 'tmb_sharpe' in result['results']:
                factor_sharpe_ratios[result['name']] = result['results']['tmb_sharpe']

        top_factors = sorted(factor_sharpe_ratios.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_factor_names = [factor[0] for factor in top_factors]

        print(f"Selected top {top_n} factors for composite: {top_factor_names}")

        # Combine factor signals
        composite_returns = pd.Series(0, index=forward_returns.index)
        valid_factors = 0

        for result in factors_results:
            if result['name'] in top_factor_names:
                factor_quantiles = result.get('quantile_returns', None)
                if factor_quantiles is not None:
                    # Run individual backtest
                    backtest = self.run_single_factor_backtest(factor_quantiles, forward_returns, result['name'])
                    composite_returns += backtest['returns']
                    valid_factors += 1

        if valid_factors > 0:
            composite_returns /= valid_factors

            # Calculate composite performance
            cumulative_returns = (1 + composite_returns).cumprod()
            portfolio_value = self.initial_capital * cumulative_returns
            metrics = self.performance_analyzer.calculate_performance_metrics(composite_returns)

            composite_results = {
                'factor_name': f'Composite_Top{top_n}',
                'returns': composite_returns,
                'cumulative_returns': cumulative_returns,
                'portfolio_value': portfolio_value,
                'metrics': metrics,
                'constituents': top_factor_names
            }

            return composite_results
        else:
            return None