import pandas as pd
import numpy as np
from scipy import stats
from config.settings import *


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate

    def calculate_performance_metrics(self, returns_series):
        """Calculate comprehensive performance metrics"""
        if len(returns_series.dropna()) == 0:
            return {}

        returns = returns_series.dropna()

        metrics = {}
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['volatility'] if metrics[
                                                                                                                  'volatility'] != 0 else 0
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics[
                                                                                                 'max_drawdown'] != 0 else 0
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)

        # Win rate and profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        metrics['profit_factor'] = abs(positive_returns.sum() / negative_returns.sum()) if len(
            negative_returns) > 0 else float('inf')

        # VaR and CVaR
        metrics['var_95'] = returns.quantile(0.05)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()

        return metrics

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_rolling_performance(self, returns_series, window=252):
        """Calculate rolling performance metrics"""
        rolling_returns = returns_series.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns - self.risk_free_rate / 252) / rolling_vol

        return rolling_returns, rolling_vol, rolling_sharpe

    def calculate_alpha_beta(self, portfolio_returns, benchmark_returns):
        """Calculate alpha and beta relative to benchmark"""
        common_index = portfolio_returns.dropna().index.intersection(benchmark_returns.dropna().index)
        if len(common_index) < 2:
            return 0, 0

        port_ret = portfolio_returns[common_index]
        bench_ret = benchmark_returns[common_index]

        # Calculate beta (covariance / variance)
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        variance = np.var(bench_ret)
        beta = covariance / variance if variance != 0 else 0

        # Calculate alpha (intercept of regression)
        alpha = np.mean(port_ret) - beta * np.mean(bench_ret)

        # Annualize alpha
        alpha_annualized = (1 + alpha) ** 252 - 1

        return alpha_annualized, beta

    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """Calculate information ratio"""
        common_index = portfolio_returns.dropna().index.intersection(benchmark_returns.dropna().index)
        if len(common_index) < 2:
            return 0

        active_returns = portfolio_returns[common_index] - benchmark_returns[common_index]
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0

        return information_ratio