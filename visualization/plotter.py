import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.settings import *


class FactorPlotter:
    def __init__(self):
        try:
            plt.style.use(PLOT_STYLE)
        except:
            plt.style.use('default')
        self.fig_size = FIG_SIZE

        # 设置中文字体（如果需要显示中文）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_ic_analysis(self, ic_series, factor_name):
        """Plot IC analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'IC Analysis - {factor_name}', fontsize=16)

        # Time series of IC
        axes[0, 0].plot(ic_series.index, ic_series.values, linewidth=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[0, 0].set_title('IC Time Series')
        axes[0, 0].set_ylabel('Information Coefficient')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of IC
        axes[0, 1].hist(ic_series.dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=ic_series.mean(), color='r', linestyle='--',
                           label=f'Mean: {ic_series.mean():.4f}')
        axes[0, 1].set_title('IC Distribution')
        axes[0, 1].set_xlabel('Information Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling mean IC
        if len(ic_series) > 63:
            rolling_ic = ic_series.rolling(window=63).mean()
            axes[1, 0].plot(rolling_ic.index, rolling_ic.values, linewidth=1)
            axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axes[1, 0].set_title('Rolling 3-Month IC Mean')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for rolling analysis',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Rolling IC Mean')
        axes[1, 0].set_ylabel('Information Coefficient')
        axes[1, 0].grid(True, alpha=0.3)

        # IC statistics
        axes[1, 1].text(0.1, 0.8, f'Mean IC: {ic_series.mean():.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'IC Std: {ic_series.std():.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'IC IR: {ic_series.mean() / ic_series.std():.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Positive Ratio: {(ic_series > 0).mean():.4f}', fontsize=12)
        axes[1, 1].set_title('IC Statistics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        plt.tight_layout()
        return fig

    def plot_quantile_returns(self, quantile_returns, factor_name):
        """Plot quantile portfolio returns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Quantile Portfolio Analysis - {factor_name}', fontsize=16)

        # Cumulative returns by quantile
        cumulative_returns = (1 + quantile_returns).cumprod()
        colors = plt.cm.viridis(np.linspace(0, 1, len(quantile_returns.columns)))

        for i in range(len(quantile_returns.columns)):
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.iloc[:, i],
                            label=f'Q{i + 1}', color=colors[i], linewidth=1.5)
        axes[0, 0].set_title('Cumulative Returns by Quantile')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Top minus bottom cumulative return
        if len(quantile_returns.columns) >= 2:
            tmb = quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
            cumulative_tmb = (1 + tmb).cumprod()
            axes[0, 1].plot(cumulative_tmb.index, cumulative_tmb.values,
                            linewidth=2, color='red')
            axes[0, 1].set_title('Top Minus Bottom Portfolio (Long Top, Short Bottom)')
            axes[0, 1].set_ylabel('Cumulative Return')
            axes[0, 1].grid(True, alpha=0.3)

        # Annualized returns by quantile
        annual_returns = quantile_returns.mean() * 252
        axes[1, 0].bar(range(len(annual_returns)), annual_returns.values,
                       color=colors, alpha=0.7)
        axes[1, 0].set_title('Annualized Returns by Quantile')
        axes[1, 0].set_xlabel('Quantile')
        axes[1, 0].set_ylabel('Annualized Return')
        axes[1, 0].grid(True, alpha=0.3)

        # Sharpe ratios by quantile
        sharpe_ratios = quantile_returns.mean() / quantile_returns.std() * np.sqrt(252)
        sharpe_ratios = sharpe_ratios.replace([np.inf, -np.inf], np.nan).fillna(0)

        axes[1, 1].bar(range(len(sharpe_ratios)), sharpe_ratios.values,
                       color=colors, alpha=0.7)
        axes[1, 1].set_title('Sharpe Ratios by Quantile')
        axes[1, 1].set_xlabel('Quantile')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_factor_comparison(self, factors_results):
        """Compare multiple factors"""
        if len(factors_results) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Need at least 2 factors for comparison',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Factor Comparison')
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Factor Comparison', fontsize=16)

        # IC comparison
        ic_means = [result['results']['ic_mean'] for result in factors_results]
        factor_names = [result['name'] for result in factors_results]

        bars = axes[0, 0].bar(factor_names, ic_means,
                              color=plt.cm.Set3(np.linspace(0, 1, len(factor_names))))
        axes[0, 0].set_title('Mean Information Coefficient')
        axes[0, 0].set_ylabel('Mean IC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.4f}', ha='center', va='bottom')

        # IC IR comparison
        ic_irs = [result['results']['ic_ir'] for result in factors_results]
        bars = axes[0, 1].bar(factor_names, ic_irs,
                              color=plt.cm.Set3(np.linspace(0, 1, len(factor_names))))
        axes[0, 1].set_title('IC Information Ratio')
        axes[0, 1].set_ylabel('IC IR')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.4f}', ha='center', va='bottom')

        # Top minus bottom returns
        tmb_returns = [result['results']['tmb_mean_return'] * 252 for result in factors_results]
        bars = axes[1, 0].bar(factor_names, tmb_returns,
                              color=plt.cm.Set3(np.linspace(0, 1, len(factor_names))))
        axes[1, 0].set_title('Annualized Top Minus Bottom Return')
        axes[1, 0].set_ylabel('Annual Return')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.4f}', ha='center', va='bottom')

        # TMB t-statistics
        tmb_tstats = [result['results']['tmb_tstat'] for result in factors_results]
        bars = axes[1, 1].bar(factor_names, tmb_tstats,
                              color=plt.cm.Set3(np.linspace(0, 1, len(factor_names))))
        axes[1, 1].set_title('Top Minus Bottom t-Statistics')
        axes[1, 1].set_ylabel('t-stat')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, factors_data, factor_names):
        """Plot correlation heatmap between factors"""
        if len(factor_names) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Need at least 2 factors for correlation matrix',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Factor Correlation Matrix')
            return fig

        # Calculate correlations
        corr_matrix = pd.DataFrame(index=factor_names, columns=factor_names)

        for i, name1 in enumerate(factor_names):
            for j, name2 in enumerate(factor_names):
                if i == j:
                    corr_matrix.loc[name1, name2] = 1.0
                else:
                    try:
                        # Align data and calculate correlation
                        factor1 = factors_data[name1].stack().reset_index()
                        factor2 = factors_data[name2].stack().reset_index()
                        factor1.columns = ['date', 'code', 'value1']
                        factor2.columns = ['date', 'code', 'value2']

                        merged = pd.merge(factor1, factor2, on=['date', 'code'])
                        if len(merged) > 10:  # Minimum observations
                            corr = merged['value1'].corr(merged['value2'])
                            corr_matrix.loc[name1, name2] = corr
                        else:
                            corr_matrix.loc[name1, name2] = np.nan
                    except:
                        corr_matrix.loc[name1, name2] = np.nan

        corr_matrix = corr_matrix.astype(float)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)

        # Add text annotations
        for i in range(len(factor_names)):
            for j in range(len(factor_names)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=12)

        ax.set_xticks(range(len(factor_names)))
        ax.set_yticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=45)
        ax.set_yticklabels(factor_names)
        ax.set_title('Factor Correlation Matrix')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        return fig