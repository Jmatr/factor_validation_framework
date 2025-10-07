import pandas as pd
import numpy as np
import warnings
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from factors.factor_factory import FactorFactory, FactorGroupFactory
from analysis.factor_test import FactorTester
from analysis.performance import PerformanceAnalyzer
from analysis.backtester import FactorBacktester
from analysis.risk_analysis import RiskAnalyzer
from visualization.plotter import FactorPlotter
from visualization.report_generator import ReportGenerator
from utils.helpers import print_factor_results, ensure_directory, interpret_factor_performance
from utils.monitoring import PerformanceMonitor
from config.settings import *
from config.factor_configs import FACTOR_CONFIGS, FACTOR_GROUPS


def test_small_universe():
    """Use small test stock universe"""
    return [
        'sh.600000', 'sh.600036', 'sh.600519', 'sz.000001', 'sz.000858',
        'sz.002415', 'sh.601318', 'sh.601888', 'sz.000002', 'sh.600030',
        'sh.600031', 'sh.600104', 'sh.600276', 'sh.600585', 'sz.000333',
        'sz.000568', 'sz.000625', 'sz.000725', 'sz.000876', 'sz.002304'
    ]


def main():
    # Initialize components
    print("Initializing Factor Analysis Framework...")
    monitor = PerformanceMonitor()
    monitor.start_analysis()

    data_loader = DataLoader()
    data_processor = DataProcessor()
    factor_tester = FactorTester(quantiles=QUANTILES)
    performance_analyzer = PerformanceAnalyzer()
    backtester = FactorBacktester()
    risk_analyzer = RiskAnalyzer()
    plotter = FactorPlotter()
    report_generator = ReportGenerator()

    # Ensure output directories exist
    ensure_directory('output/plots')
    ensure_directory('output/reports')

    try:
        # Step 1: Data Loading
        monitor.start_phase("Data Loading")
        print("\nStep 1: Fetching stock data...")

        sample_stocks = test_small_universe()
        print(f"Using {len(sample_stocks)} stocks for analysis")

        stock_data = data_loader.get_batch_stock_data(sample_stocks, START_DATE, END_DATE)

        if len(stock_data) == 0:
            print("No data fetched. Exiting.")
            return
        monitor.end_phase("Data Loading")

        # Step 2: Data Processing
        monitor.start_phase("Data Processing")
        print("\nStep 2: Processing data...")
        panel_data, common_dates = data_processor.create_panel_data(stock_data)

        if not panel_data or 'close' not in panel_data:
            print("Insufficient data for analysis. Exiting.")
            return

        print(f"Available data fields: {list(panel_data.keys())}")

        # Clean data
        panel_data = data_processor.clean_data(panel_data)

        # Calculate forward returns
        forward_returns = data_processor.calculate_returns(panel_data['close'], periods=RETURN_PERIOD)

        if forward_returns is None or forward_returns.empty:
            print("Cannot calculate forward returns. Exiting.")
            return
        monitor.end_phase("Data Processing")

        # Step 3: Factor Calculation
        monitor.start_phase("Factor Calculation")
        print("\nStep 3: Calculating factors...")

        # Define factors to test from different groups
        factors_to_test = []

        # Momentum factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("momentum", FACTOR_CONFIGS))

        # Value factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("value", FACTOR_CONFIGS))

        # Quality factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("quality", FACTOR_CONFIGS))

        # Technical factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("technical", FACTOR_CONFIGS))

        # Volatility factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("volatility", FACTOR_CONFIGS))

        # Reversal factors
        factors_to_test.extend(FactorGroupFactory.create_factor_group("reversal", FACTOR_CONFIGS))

        # Size factor
        factors_to_test.append(FactorFactory.create_factor("size"))

        print(f"Testing {len(factors_to_test)} factors in total")
        monitor.end_phase("Factor Calculation")

        # Step 4: Factor Testing
        monitor.start_phase("Factor Testing")
        factors_data = {}
        factors_results = []

        for factor in factors_to_test:
            try:
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

                if len(valid_stocks) < MIN_STOCKS_PER_QUANTILE:
                    print(f"  Skipping {factor.name}: insufficient stocks ({len(valid_stocks)})")
                    continue

                print(f"  Testing {factor.name} with {len(valid_stocks)} stocks...")

                # Run factor tests
                results, ic_series, quantile_returns, tmb = factor_tester.run_comprehensive_test(
                    aligned_factor, aligned_returns
                )

                if results is None:
                    continue

                # Store results
                factor_result = {
                    'name': factor.name,
                    'results': results,
                    'ic_series': ic_series,
                    'quantile_returns': quantile_returns,
                    'top_minus_bottom': tmb,
                    'valid_stocks': len(valid_stocks)
                }
                factors_results.append(factor_result)

                # Print results
                print_factor_results(results, factor.name)

                # Generate plots
                print(f"  Generating plots for {factor.name}...")
                try:
                    ic_plot = plotter.plot_ic_analysis(ic_series, factor.name)
                    ic_plot.savefig(f'output/plots/{factor.name}_ic_analysis.png', dpi=150, bbox_inches='tight')
                    plt.close(ic_plot)

                    quantile_plot = plotter.plot_quantile_returns(quantile_returns, factor.name)
                    quantile_plot.savefig(f'output/plots/{factor.name}_quantile_analysis.png', dpi=150,
                                          bbox_inches='tight')
                    plt.close(quantile_plot)

                    print(f"  ✓ {factor.name} analysis completed")
                except Exception as e:
                    print(f"  ✗ Plot generation failed for {factor.name}: {str(e)}")

            except Exception as e:
                print(f"  ✗ Error processing {factor.name}: {str(e)}")
                continue
        monitor.end_phase("Factor Testing")

        # Step 5: Comparative Analysis
        monitor.start_phase("Comparative Analysis")
        print("\nStep 5: Running comparative analysis...")
        if len(factors_results) > 1:
            try:
                # Factor comparison plot
                comparison_plot = plotter.plot_factor_comparison(factors_results)
                comparison_plot.savefig('output/plots/factor_comparison.png', dpi=150, bbox_inches='tight')
                plt.close(comparison_plot)

                # Correlation heatmap
                correlation_plot = plotter.plot_correlation_heatmap(factors_data, list(factors_data.keys()))
                correlation_plot.savefig('output/plots/factor_correlations.png', dpi=150, bbox_inches='tight')
                plt.close(correlation_plot)

                print("✓ Comparative analysis completed")
            except Exception as e:
                print(f"✗ Comparative analysis failed: {str(e)}")
        else:
            print("  Not enough factors for comparative analysis")
        monitor.end_phase("Comparative Analysis")

        # Step 6: Backtesting
        monitor.start_phase("Backtesting")
        print("\nStep 6: Running backtests...")
        if factors_results:
            try:
                backtest_results = backtester.run_multiple_factor_backtest(factors_results, forward_returns)

                if backtest_results:
                    # Plot backtest results
                    backtest_plot = plotter.plot_backtest_results(backtest_results)
                    backtest_plot.savefig('output/plots/backtest_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close(backtest_plot)

                    # Run composite strategy
                    composite_results = backtester.run_equal_weight_composite(factors_results, forward_returns, top_n=3)
                    if composite_results:
                        backtest_results['Composite'] = composite_results
                        print(f"✓ Composite strategy created with {composite_results['constituents']}")

                    print(f"✓ Backtesting completed for {len(backtest_results)} factors")
                else:
                    print("  No valid backtest results")
            except Exception as e:
                print(f"✗ Backtesting failed: {str(e)}")
        monitor.end_phase("Backtesting")

        # Step 7: Risk Analysis
        monitor.start_phase("Risk Analysis")
        print("\nStep 7: Running risk analysis...")
        if factors_data:
            try:
                # Calculate factor correlations
                correlation_matrix = risk_analyzer.calculate_factor_correlation_matrix(factors_data)
                if not correlation_matrix.empty:
                    correlation_matrix.to_csv('output/factor_correlations.csv')
                    print("✓ Factor correlation analysis completed")

                # Calculate risk-adjusted metrics
                factor_returns = {result['name']: result['top_minus_bottom'] for result in factors_results
                                  if 'top_minus_bottom' in result and result['top_minus_bottom'] is not None}

                if factor_returns:
                    risk_metrics = risk_analyzer.calculate_risk_adjusted_metrics(factor_returns)
                    risk_df = pd.DataFrame(risk_metrics).T
                    risk_df.to_csv('output/risk_metrics.csv')
                    print("✓ Risk-adjusted metrics calculated")
            except Exception as e:
                print(f"✗ Risk analysis failed: {str(e)}")
        monitor.end_phase("Risk Analysis")

        # Step 8: Reporting
        monitor.start_phase("Reporting")
        print("\nStep 8: Generating reports...")
        if factors_results:
            # Generate summary report
            summary_data = []
            for result in factors_results:
                summary_data.append({
                    'Factor': result['name'],
                    'Mean_IC': result['results']['ic_mean'],
                    'IC_IR': result['results']['ic_ir'],
                    'Hit_Ratio': result['results'].get('hit_ratio', 0),
                    'TMB_Return_ann': result['results']['tmb_mean_return'] * 252,
                    'TMB_Sharpe': result['results']['tmb_sharpe'],
                    'TMB_tstat': result['results']['tmb_tstat'],
                    'Valid_Stocks': result.get('valid_stocks', 0)
                })

            summary_df = pd.DataFrame(summary_data)
            print("\nFACTOR PERFORMANCE SUMMARY:")
            print("=" * 90)
            print(summary_df.round(4))

            # Save summary to CSV
            summary_df.to_csv('output/factor_performance_summary.csv', index=False)
            print(f"\n✓ Summary saved to output/factor_performance_summary.csv")

            # Generate detailed HTML report
            report_path = report_generator.generate_summary_report(factors_results)

            # Performance interpretation
            print("\n" + "=" * 70)
            print("PERFORMANCE INTERPRETATION")
            print("=" * 70)
            for result in factors_results:
                interpret_factor_performance(result['results'], result['name'])
        else:
            print("  No valid factors to summarize")
        monitor.end_phase("Reporting")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        if factors_results:
            print("Check the 'output' folder for:")
            print("  - IC analysis plots (output/plots/*_ic_analysis.png)")
            print("  - Quantile analysis plots (output/plots/*_quantile_analysis.png)")
            print("  - Factor comparison (output/plots/factor_comparison.png)")
            print("  - Factor correlations (output/plots/factor_correlations.png)")
            print("  - Backtest results (output/plots/backtest_comparison.png)")
            print("  - Performance summary (output/factor_performance_summary.csv)")
            print("  - Detailed report (output/factor_analysis_report.html)")
        else:
            print("No factors were successfully tested. Check the configuration and data sources.")

    except Exception as e:
        print(f"\n❌ Main execution error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure clean shutdown
        try:
            data_loader.safe_logout()
            monitor.end_analysis()
        except:
            pass


if __name__ == "__main__":
    main()