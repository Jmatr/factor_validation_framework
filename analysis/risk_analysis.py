import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class RiskAnalyzer:
    def __init__(self):
        pass

    def calculate_factor_exposures(self, factor_returns, market_returns=None):
        """Calculate factor exposures and correlations"""
        exposures = {}

        if market_returns is not None:
            # Calculate beta to market
            for factor_name, factor_ret in factor_returns.items():
                common_index = factor_ret.dropna().index.intersection(market_returns.dropna().index)
                if len(common_index) > 10:
                    factor_aligned = factor_ret[common_index]
                    market_aligned = market_returns[common_index]

                    # Calculate beta
                    covariance = np.cov(factor_aligned, market_aligned)[0, 1]
                    variance = np.var(market_aligned)
                    beta = covariance / variance if variance != 0 else 0

                    exposures[factor_name] = {
                        'market_beta': beta,
                        'correlation_with_market': factor_aligned.corr(market_aligned)
                    }

        return exposures

    def calculate_factor_correlation_matrix(self, factors_data):
        """Calculate correlation matrix between factors"""
        factor_names = list(factors_data.keys())
        n_factors = len(factor_names)

        if n_factors < 2:
            return pd.DataFrame()

        # Flatten factor data for correlation calculation
        flat_data = {}
        for factor_name, factor_data in factors_data.items():
            flat_series = factor_data.stack().reset_index()
            flat_series.columns = ['date', 'stock', 'value']
            flat_data[factor_name] = flat_series

        # Calculate pairwise correlations
        corr_matrix = pd.DataFrame(index=factor_names, columns=factor_names)

        for i, name1 in enumerate(factor_names):
            for j, name2 in enumerate(factor_names):
                if i == j:
                    corr_matrix.loc[name1, name2] = 1.0
                else:
                    try:
                        # Merge factor data
                        merged = pd.merge(flat_data[name1], flat_data[name2],
                                          on=['date', 'stock'], suffixes=('_1', '_2'))
                        if len(merged) > 10:
                            corr = merged['value_1'].corr(merged['value_2'])
                            corr_matrix.loc[name1, name2] = corr
                        else:
                            corr_matrix.loc[name1, name2] = np.nan
                    except:
                        corr_matrix.loc[name1, name2] = np.nan

        return corr_matrix.astype(float)

    def calculate_factor_stability(self, ic_series, window=252):
        """Calculate factor stability metrics"""
        if len(ic_series) < window:
            return {}

        rolling_ic = ic_series.rolling(window=window).mean()
        rolling_std = ic_series.rolling(window=window).std()

        stability_metrics = {
            'ic_stability': 1 - (rolling_std.std() / abs(rolling_ic.mean())) if rolling_ic.mean() != 0 else 0,
            'ic_decay_rate': self._calculate_ic_decay(ic_series),
            'rolling_correlation': rolling_ic.autocorr(lag=1)  # First-order autocorrelation
        }

        return stability_metrics

    def _calculate_ic_decay(self, ic_series, max_lag=12):
        """Calculate IC decay over different lags"""
        if len(ic_series) < max_lag:
            return 0

        decays = []
        for lag in range(1, min(max_lag, len(ic_series) // 2)):
            if lag < len(ic_series):
                corr = ic_series.autocorr(lag=lag)
                if not np.isnan(corr):
                    decays.append(corr)

        return np.mean(decays) if decays else 0

    def calculate_risk_adjusted_metrics(self, factor_returns, risk_free_rate=0.03):
        """Calculate risk-adjusted performance metrics"""
        risk_metrics = {}

        for factor_name, returns in factor_returns.items():
            returns_clean = returns.dropna()
            if len(returns_clean) == 0:
                continue

            # Basic risk metrics
            annual_return = returns_clean.mean() * 252
            annual_volatility = returns_clean.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0

            # Downside risk metrics
            downside_returns = returns_clean[returns_clean < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0

            # Maximum drawdown
            cumulative_returns = (1 + returns_clean).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # VaR and CVaR
            var_95 = returns_clean.quantile(0.05)
            cvar_95 = returns_clean[returns_clean <= var_95].mean()

            risk_metrics[factor_name] = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'downside_volatility': downside_volatility
            }

        return risk_metrics