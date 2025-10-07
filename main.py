import pandas as pd
import numpy as np
import warnings
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from factors.factor_library import FactorFactory
from analysis.factor_test import FactorTester
from analysis.performance import PerformanceAnalyzer
from visualization.plotter import FactorPlotter
from utils.helpers import print_factor_results, ensure_directory
from config.settings import *
from report.report_generator import ReportGenerator

def test_small_universe():
    """使用小型测试股票池"""
    return [
        'sh.600000',  # 浦发银行
        'sh.600036',  # 招商银行
        'sh.600519',  # 贵州茅台
        'sz.000001',  # 平安银行
        'sz.000858',  # 五粮液
        'sz.002415',  # 海康威视
        'sh.601318',  # 中国平安
        'sh.601888',  # 中国国旅
        'sz.000002',  # 万科A
        'sh.600030',  # 中信证券
    ]


def main():
    # Initialize components
    print("Initializing Factor Analysis System...")
    data_loader = DataLoader()
    data_processor = DataProcessor()
    factor_tester = FactorTester(quantiles=QUANTILES)
    performance_analyzer = PerformanceAnalyzer()
    plotter = FactorPlotter()

    # Ensure output directory exists
    ensure_directory('output')

    try:
        # Step 1: Get stock universe and data
        print("\nStep 1: Fetching stock data...")

        # 使用小型测试股票池
        sample_stocks = test_small_universe()
        print(f"Using {len(sample_stocks)} stocks for analysis: {sample_stocks}")

        stock_data = data_loader.get_batch_stock_data(
            sample_stocks, START_DATE, END_DATE
        )

        if len(stock_data) == 0:
            print("No data fetched. Exiting.")
            return

        # Step 2: Process data into panel format
        print("\nStep 2: Processing data...")
        panel_data, common_dates = data_processor.create_panel_data(stock_data)

        # 检查是否有足够的数据
        if not panel_data or 'close' not in panel_data:
            print("Insufficient data for analysis. Exiting.")
            return

        print(f"Available data fields: {list(panel_data.keys())}")

        # Clean data
        panel_data = data_processor.clean_data(panel_data)

        # Calculate forward returns
        forward_returns = data_processor.calculate_returns(
            panel_data['close'], periods=RETURN_PERIOD
        )

        if forward_returns is None or forward_returns.empty:
            print("Cannot calculate forward returns. Exiting.")
            return

        # Step 3: Define factors to test - 从简单因子开始
        print("\nStep 3: Calculating factors...")
        factors_to_test = [
            ("momentum", {"lookback_period": 21}),
            ("size", {}),
            ("value", {}),
        ]

        factors_data = {}
        factors_results = []

        for factor_type, params in factors_to_test:
            try:
                # Create factor
                factor = FactorFactory.create_factor(factor_type, **params)
                print(f"\nCalculating {factor.name}...")

                # Calculate factor values
                factor_values = factor.calculate(panel_data)
                if factor_values is None or factor_values.empty:
                    print(f"  No data for {factor.name}")
                    continue

                factors_data[factor.name] = factor_values

                # Align factor data with forward returns
                common_index = factor_values.index.intersection(forward_returns.index)
                aligned_factor = factor_values.loc[common_index]
                aligned_returns = forward_returns.loc[common_index]

                # Remove stocks with insufficient data
                valid_stocks = aligned_factor.columns[
                    aligned_factor.notna().sum() > MIN_PERIODS
                    ]
                aligned_factor = aligned_factor[valid_stocks]
                aligned_returns = aligned_returns[valid_stocks]

                if len(valid_stocks) < 5:  # 降低最小股票数量要求
                    print(f"  Skipping {factor.name}: insufficient stocks ({len(valid_stocks)})")
                    continue

                print(f"  Testing {factor.name} with {len(valid_stocks)} stocks...")

                # Run factor tests
                results, ic_series, quantile_returns, tmb = factor_tester.run_comprehensive_test(
                    aligned_factor, aligned_returns
                )

                # Store results
                factor_result = {
                    'name': factor.name,
                    'results': results,
                    'ic_series': ic_series,
                    'quantile_returns': quantile_returns,
                    'top_minus_bottom': tmb
                }
                factors_results.append(factor_result)

                # Print results
                print_factor_results(results, factor.name)

                # Generate plots
                print(f"  Generating plots for {factor.name}...")
                try:
                    ic_plot = plotter.plot_ic_analysis(ic_series, factor.name)
                    ic_plot.savefig(f'output/{factor.name}_ic_analysis.png', dpi=150, bbox_inches='tight')
                    plt.close(ic_plot)

                    quantile_plot = plotter.plot_quantile_returns(quantile_returns, factor.name)
                    quantile_plot.savefig(f'output/{factor.name}_quantile_analysis.png', dpi=150, bbox_inches='tight')
                    plt.close(quantile_plot)

                    print(f"  ✓ {factor.name} analysis completed")
                except Exception as e:
                    print(f"  ✗ Plot generation failed for {factor.name}: {str(e)}")

            except Exception as e:
                print(f"  ✗ Error processing {factor_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Step 4: Comparative analysis
        print("\nStep 4: Running comparative analysis...")
        if len(factors_results) > 1:
            try:
                # Factor comparison plot
                comparison_plot = plotter.plot_factor_comparison(factors_results)
                comparison_plot.savefig('output/factor_comparison.png', dpi=150, bbox_inches='tight')
                plt.close(comparison_plot)

                # Correlation heatmap
                correlation_plot = plotter.plot_correlation_heatmap(factors_data, list(factors_data.keys()))
                correlation_plot.savefig('output/factor_correlations.png', dpi=150, bbox_inches='tight')
                plt.close(correlation_plot)

                print("✓ Comparative analysis completed")
            except Exception as e:
                print(f"✗ Comparative analysis failed: {str(e)}")
        else:
            print("  Not enough factors for comparative analysis")

        # Step 5: Summary report
        print("\nStep 5: Generating summary report...")
        if factors_results:
            summary_data = []
            for result in factors_results:
                summary_data.append({
                    'Factor': result['name'],
                    'Mean_IC': result['results']['ic_mean'],
                    'IC_IR': result['results']['ic_ir'],
                    'IC_tstat': result['results']['ic_tstat'],
                    'TMB_Return_ann': result['results']['tmb_mean_return'] * 252,
                    'TMB_Sharpe': result['results']['tmb_sharpe'],
                    'TMB_tstat': result['results']['tmb_tstat']
                })

            summary_df = pd.DataFrame(summary_data)
            print("\nFACTOR PERFORMANCE SUMMARY:")
            print("=" * 80)
            print(summary_df.round(4))

            # Save summary to CSV
            summary_df.to_csv('output/factor_performance_summary.csv', index=False)
            print(f"\n✓ Summary saved to output/factor_performance_summary.csv")
        else:
            print("  No valid factors to summarize")

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        if factors_results:
            print("Check the 'output' folder for results and plots.")
        else:
            print("No factors were successfully tested. Check the configuration and data sources.")

        if factors_results:
            # 生成详细报告
            report_generator = ReportGenerator()
            report_generator.generate_summary_report(factors_results)

            # 性能解读
            print("\n" + "=" * 60)
            print("PERFORMANCE INTERPRETATION")
            print("=" * 60)
            for result in factors_results:
                from utils.helpers import interpret_factor_performance
                interpret_factor_performance(result['results'], result['name'])

    except Exception as e:
        print(f"\n❌ Main execution error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure baostock logout
        try:
            data_loader._logout()
            print("Baostock logout completed")
        except:
            pass


if __name__ == "__main__":
    main()