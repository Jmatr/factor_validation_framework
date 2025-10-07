### 7. 性能分析模块 (analysis/performance.py)

import pandas as pd
import numpy as np
from scipy import stats


class PerformanceAnalyzer:
    def __init__(self):
        pass

    def calculate_performance_metrics(self, returns_series):
        """Calculate comprehensive performance metrics"""
        if len(returns_series.dropna()) == 0:
            return {}

        returns = returns_series.dropna()

        metrics = {}
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] != 0 else 0
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics[
                                                                                                 'max_drawdown'] != 0 else 0
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)

        # Win rate
        metrics['win_rate'] = (returns > 0).sum() / len(returns)

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
        rolling_sharpe = rolling_returns / rolling_vol

        return rolling_returns, rolling_vol, rolling_sharpe