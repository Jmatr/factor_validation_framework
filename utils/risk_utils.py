import pandas as pd
import numpy as np
from scipy import stats


class RiskUtils:
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(confidence_level)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskUtils.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_drawdowns(returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        drawdowns = RiskUtils.calculate_drawdowns(returns)
        return drawdowns.min()

    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        drawdowns = RiskUtils.calculate_drawdowns(returns)
        return np.sqrt(np.mean(drawdowns ** 2))

    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
        """Calculate tail ratio (right tail / left tail)"""
        left_tail = returns.quantile(quantile)
        right_tail = returns.quantile(1 - quantile)
        return abs(right_tail / left_tail) if left_tail != 0 else float('inf')

    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()
        return gains / losses if losses != 0 else float('inf')