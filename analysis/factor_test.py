import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class FactorTester:
    def __init__(self, quantiles=5):
        self.quantiles = quantiles

    def create_factor_quantiles(self, factor_data):
        """Create quantile portfolios based on factor values"""
        quantile_data = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)

        for date in factor_data.index:
            daily_factors = factor_data.loc[date].dropna()
            if len(daily_factors) < self.quantiles:
                continue

            try:
                # 使用rank而不是qcut来处理重复值
                ranks = daily_factors.rank(method='first')
                quantile_labels = pd.cut(ranks, bins=self.quantiles, labels=False, duplicates='drop')
                quantile_data.loc[date] = quantile_labels
            except Exception as e:
                continue

        return quantile_data

    def calculate_quantile_returns(self, factor_quantiles, forward_returns):
        """Calculate returns for each quantile portfolio"""
        quantile_returns = {}

        for quantile in range(self.quantiles):
            quantile_mask = factor_quantiles == quantile
            # 等权重组合收益
            quantile_ret = forward_returns[quantile_mask].mean(axis=1)
            quantile_returns[quantile] = quantile_ret

        return pd.DataFrame(quantile_returns)

    def ic_analysis(self, factor_data, forward_returns):
        """Information Coefficient analysis - 使用Rank IC"""
        ic_series = pd.Series(index=factor_data.index, dtype=float)

        for date in factor_data.index:
            factor_values = factor_data.loc[date].dropna()
            returns = forward_returns.loc[date].dropna()

            common_stocks = factor_values.index.intersection(returns.index)
            if len(common_stocks) > 5:
                factor_values = factor_values[common_stocks]
                returns = returns[common_stocks]

                # 使用Spearman秩相关 (Rank IC)
                try:
                    ic = stats.spearmanr(factor_values, returns)[0]
                    if not np.isnan(ic):
                        ic_series[date] = ic
                except:
                    continue

        return ic_series

    def factor_returns_analysis(self, factor_data, forward_returns):
        """Calculate factor returns using Fama-MacBeth regression"""
        factor_returns = pd.Series(index=factor_data.index, dtype=float)

        for date in factor_data.index:
            factor_values = factor_data.loc[date].dropna()
            returns = forward_returns.loc[date].dropna()

            common_stocks = factor_values.index.intersection(returns.index)
            if len(common_stocks) > 5:
                factor_values = factor_values[common_stocks]
                returns = returns[common_stocks]

                # 标准化因子值
                factor_values = (factor_values - factor_values.mean()) / factor_values.std()

                # 简单横截面回归
                try:
                    X = np.column_stack([np.ones(len(factor_values)), factor_values])
                    beta = np.linalg.lstsq(X, returns, rcond=None)[0]
                    factor_returns[date] = beta[1]  # Factor return
                except:
                    continue

        return factor_returns

    def calculate_turnover_analysis(self, factor_quantiles):
        """计算因子组合换手率"""
        turnover = pd.Series(index=factor_quantiles.index[1:], dtype=float)

        for i in range(1, len(factor_quantiles)):
            current = factor_quantiles.iloc[i]
            previous = factor_quantiles.iloc[i - 1]

            common_dates = current.dropna().index.intersection(previous.dropna().index)
            if len(common_dates) > 0:
                # 计算组合变化率
                changes = (current[common_dates] != previous[common_dates]).mean()
                turnover.iloc[i - 1] = changes

        return turnover

    def run_comprehensive_test(self, factor_data, forward_returns):
        """Run comprehensive factor tests"""
        results = {}

        # IC Analysis
        ic_series = self.ic_analysis(factor_data, forward_returns)
        valid_ic = ic_series.dropna()

        if len(valid_ic) == 0:
            return None, None, None, None

        results['ic_mean'] = valid_ic.mean()
        results['ic_std'] = valid_ic.std()
        results['ic_ir'] = results['ic_mean'] / results['ic_std'] if results['ic_std'] != 0 else 0
        results['ic_tstat'] = stats.ttest_1samp(valid_ic, 0)[0] if len(valid_ic) > 1 else 0
        results['ic_positive_ratio'] = (valid_ic > 0).sum() / len(valid_ic)

        # Quantile Analysis
        factor_quantiles = self.create_factor_quantiles(factor_data)
        quantile_returns = self.calculate_quantile_returns(factor_quantiles, forward_returns)

        # Top minus bottom portfolio
        if len(quantile_returns.columns) >= 2:
            top_minus_bottom = quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
            results['tmb_mean_return'] = top_minus_bottom.mean()
            results['tmb_std'] = top_minus_bottom.std()
            results['tmb_sharpe'] = results['tmb_mean_return'] / results['tmb_std'] * np.sqrt(252) if results[
                                                                                                          'tmb_std'] != 0 else 0
            results['tmb_tstat'] = stats.ttest_1samp(top_minus_bottom.dropna(), 0)[0] if len(
                top_minus_bottom.dropna()) > 1 else 0
        else:
            results['tmb_mean_return'] = 0
            results['tmb_std'] = 0
            results['tmb_sharpe'] = 0
            results['tmb_tstat'] = 0

        # Factor returns
        factor_returns = self.factor_returns_analysis(factor_data, forward_returns)
        valid_factor_returns = factor_returns.dropna()
        if len(valid_factor_returns) > 0:
            results['factor_return_mean'] = valid_factor_returns.mean()
            results['factor_return_std'] = valid_factor_returns.std()
            results['factor_return_sharpe'] = results['factor_return_mean'] / results['factor_return_std'] * np.sqrt(
                252) if results['factor_return_std'] != 0 else 0
        else:
            results['factor_return_mean'] = 0
            results['factor_return_std'] = 0
            results['factor_return_sharpe'] = 0

        # Turnover analysis
        turnover = self.calculate_turnover_analysis(factor_quantiles)
        results['avg_turnover'] = turnover.mean() if len(turnover) > 0 else 0

        return results, ic_series, quantile_returns, top_minus_bottom