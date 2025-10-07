import sys

import pandas as pd
import numpy as np
import pickle
import os
import socket
from typing import Any, Dict


def safe_socket_operation(func):
    """Decorator for safe socket operations"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (socket.error, OSError) as e:
            print(f"Socket operation warning: {e}")
            return None

    return wrapper


def save_results(results: Any, filename: str):
    """Save analysis results to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")


def load_results(filename: str) -> Any:
    """Load analysis results from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def ensure_directory(directory: str):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def print_factor_results(results: Dict, factor_name: str):
    """Print formatted factor test results"""
    print(f"\n{'=' * 60}")
    print(f"FACTOR RESULTS: {factor_name}")
    print(f"{'=' * 60}")

    print("\nIC Analysis:")
    print(f"  Mean IC: {results['ic_mean']:.4f}")
    print(f"  IC Std: {results['ic_std']:.4f}")
    print(f"  IC IR: {results['ic_ir']:.4f}")
    print(f"  IC t-stat: {results['ic_tstat']:.4f}")
    print(f"  IC Positive Ratio: {results['ic_positive_ratio']:.4f}")
    print(f"  Hit Ratio: {results.get('hit_ratio', 0):.4f}")

    print("\nQuantile Analysis (Top Minus Bottom):")
    print(f"  Mean Return: {results['tmb_mean_return']:.6f}")
    print(f"  Annualized Return: {results['tmb_mean_return'] * 252:.4f}%")
    print(f"  Std: {results['tmb_std']:.6f}")
    print(f"  Sharpe: {results['tmb_sharpe']:.4f}")
    print(f"  t-stat: {results['tmb_tstat']:.4f}")

    print("\nFactor Returns:")
    print(f"  Mean Factor Return: {results['factor_return_mean']:.6f}")
    print(f"  Factor Return Sharpe: {results['factor_return_sharpe']:.4f}")

    if 'avg_turnover' in results:
        print(f"  Average Turnover: {results['avg_turnover']:.4f}")


def interpret_factor_performance(results: Dict, factor_name: str):
    """Provide interpretation of factor performance"""
    ic_mean = results['ic_mean']
    tmb_return = results['tmb_mean_return'] * 252
    tmb_tstat = abs(results['tmb_tstat'])

    print(f"\nPerformance Interpretation for {factor_name}:")

    # IC interpretation
    if abs(ic_mean) < 0.02:
        ic_interpretation = "Very weak predictive power"
    elif abs(ic_mean) < 0.05:
        ic_interpretation = "Weak predictive power"
    elif abs(ic_mean) < 0.1:
        ic_interpretation = "Moderate predictive power"
    else:
        ic_interpretation = "Strong predictive power"

    print(f"  IC Significance: {ic_interpretation}")

    # Statistical significance
    if tmb_tstat > 2.58:
        significance = "*** (Highly significant at 1% level)"
    elif tmb_tstat > 1.96:
        significance = "** (Significant at 5% level)"
    elif tmb_tstat > 1.65:
        significance = "* (Significant at 10% level)"
    else:
        significance = "Not statistically significant"

    print(f"  TMB Significance: {significance}")

    # Factor direction and strength
    direction = "Positive" if tmb_return > 0 else "Negative"
    if abs(tmb_return) < 2:
        strength = "Weak"
    elif abs(tmb_return) < 5:
        strength = "Moderate"
    elif abs(tmb_return) < 10:
        strength = "Strong"
    else:
        strength = "Very strong"

    print(f"  Factor Direction: {direction}")
    print(f"  Factor Strength: {strength}")

    # Investment recommendation
    if tmb_tstat > 1.96 and tmb_return > 2:
        recommendation = "STRONG BUY - Consider for portfolio inclusion"
    elif tmb_tstat > 1.65 and tmb_return > 1:
        recommendation = "BUY - Potential for positive returns"
    elif tmb_tstat > 1.65 and tmb_return < -1:
        recommendation = "SELL - Consider for short positions"
    elif tmb_tstat > 1.96 and tmb_return < -2:
        recommendation = "STRONG SELL - Good candidate for shorting"
    else:
        recommendation = "HOLD - Insufficient evidence for action"

    print(f"  Recommendation: {recommendation}")


def calculate_performance_rankings(factors_results: list) -> pd.DataFrame:
    """Calculate performance rankings for all factors"""
    ranking_data = []

    for result in factors_results:
        results = result['results']
        ranking_data.append({
            'Factor': result['name'],
            'IC_Rank': results['ic_mean'],
            'Sharpe_Rank': results['tmb_sharpe'],
            'Return_Rank': results['tmb_mean_return'] * 252,
            'TStat_Rank': abs(results['tmb_tstat']),
            'Hit_Ratio_Rank': results.get('hit_ratio', 0)
        })

    df = pd.DataFrame(ranking_data)

    # Calculate composite rank
    for col in ['IC_Rank', 'Sharpe_Rank', 'Return_Rank', 'TStat_Rank', 'Hit_Ratio_Rank']:
        df[f'{col}_Rank'] = df[col].rank(ascending=False if col != 'IC_Rank' else True)

    df['Composite_Rank'] = df[
        [f'{col}_Rank' for col in ['IC_Rank', 'Sharpe_Rank', 'Return_Rank', 'TStat_Rank', 'Hit_Ratio_Rank']]].mean(
        axis=1)
    df['Overall_Rank'] = df['Composite_Rank'].rank()

    return df.sort_values('Overall_Rank')


def check_memory_usage(data_dict: Dict) -> None:
    """Check memory usage of data objects"""
    total_memory = 0
    print("\nMemory Usage Analysis:")
    print("-" * 30)

    for name, data in data_dict.items():
        if hasattr(data, 'memory_usage'):
            if isinstance(data, pd.DataFrame):
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            else:
                memory_mb = sys.getsizeof(data) / 1024 / 1024

            total_memory += memory_mb
            print(f"{name}: {memory_mb:.2f} MB")

    print(f"Total Memory: {total_memory:.2f} MB")