import pandas as pd
import numpy as np
import pickle
import os
import socket


def safe_socket_operation(func):
    """装饰器用于安全处理socket操作"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (socket.error, OSError) as e:
            print(f"Socket operation warning: {e}")
            return None

    return wrapper


def save_results(results, filename):
    """Save analysis results to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")


def load_results(filename):
    """Load analysis results from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def ensure_directory(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def print_factor_results(results, factor_name):
    """Print formatted factor test results"""
    print(f"\n{'=' * 50}")
    print(f"FACTOR RESULTS: {factor_name}")
    print(f"{'=' * 50}")

    print("\nIC Analysis:")
    print(f"  Mean IC: {results['ic_mean']:.4f}")
    print(f"  IC Std: {results['ic_std']:.4f}")
    print(f"  IC IR: {results['ic_ir']:.4f}")
    print(f"  IC t-stat: {results['ic_tstat']:.4f}")
    print(f"  IC Positive Ratio: {results['ic_positive_ratio']:.4f}")

    print("\nQuantile Analysis (Top Minus Bottom):")
    print(f"  Mean Return: {results['tmb_mean_return']:.6f}")
    print(f"  Annualized Return: {results['tmb_mean_return'] * 252:.4f}")
    print(f"  Std: {results['tmb_std']:.6f}")
    print(f"  Sharpe: {results['tmb_sharpe']:.4f}")
    print(f"  t-stat: {results['tmb_tstat']:.4f}")

    print("\nFactor Returns:")
    print(f"  Mean Factor Return: {results['factor_return_mean']:.6f}")
    print(f"  Factor Return Sharpe: {results['factor_return_sharpe']:.4f}")


def interpret_factor_performance(results, factor_name):
    """提供因子表现的解读"""
    ic_mean = results['ic_mean']
    tmb_return = results['tmb_mean_return'] * 252
    tmb_tstat = abs(results['tmb_tstat'])

    print("\nPerformance Interpretation:")

    # IC解读
    if abs(ic_mean) < 0.02:
        ic_interpretation = "Very weak predictive power"
    elif abs(ic_mean) < 0.05:
        ic_interpretation = "Weak predictive power"
    elif abs(ic_mean) < 0.1:
        ic_interpretation = "Moderate predictive power"
    else:
        ic_interpretation = "Strong predictive power"

    print(f"  IC Significance: {ic_interpretation}")

    # 收益显著性解读
    if tmb_tstat > 2.58:
        significance = "*** (1% level)"
    elif tmb_tstat > 1.96:
        significance = "** (5% level)"
    elif tmb_tstat > 1.65:
        significance = "* (10% level)"
    else:
        significance = "Not significant"

    print(f"  TMB Significance: {significance}")

    # 因子方向
    direction = "Positive" if tmb_return > 0 else "Negative"
    print(f"  Factor Direction: {direction}")