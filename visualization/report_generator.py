import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np

class ReportGenerator:
    def __init__(self):
        pass

    def generate_summary_report(self, factors_results, output_path='output/factor_analysis_report.html'):
        """Generate detailed HTML report"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Factor Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .factor-group {{ margin: 15px 0; padding: 10px; background-color: #f9f9f9; }}
                .significance-1 {{ background-color: #d4edda; }}
                .significance-5 {{ background-color: #fff3cd; }}
                .significance-10 {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Factor Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Period: 2018-01-01 to 2023-12-31</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                {self._generate_executive_summary(factors_results)}
            </div>

            <div class="section">
                <h2>Performance Summary</h2>
                {self._generate_summary_table(factors_results)}
            </div>

            <div class="section">
                <h2>Factor Group Analysis</h2>
                {self._generate_factor_group_analysis(factors_results)}
            </div>

            <div class="section">
                <h2>Key Findings</h2>
                {self._generate_key_findings(factors_results)}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations(factors_results)}
            </div>
        </body>
        </html>
        """

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Detailed report saved to: {output_path}")
        return output_path

    def _generate_executive_summary(self, factors_results):
        """Generate executive summary"""
        if not factors_results:
            return "<p>No factor results available.</p>"

        # Calculate overall statistics
        total_factors = len(factors_results)
        significant_factors_1 = len([f for f in factors_results if abs(f['results']['tmb_tstat']) > 2.58])
        significant_factors_5 = len([f for f in factors_results if abs(f['results']['tmb_tstat']) > 1.96])
        positive_factors = len([f for f in factors_results if f['results']['tmb_mean_return'] > 0])

        best_factor = max(factors_results, key=lambda x: x['results']['tmb_sharpe'])
        worst_factor = min(factors_results, key=lambda x: x['results']['tmb_sharpe'])

        summary_html = f"""
        <div class="executive-summary">
            <p><strong>Total Factors Analyzed:</strong> {total_factors}</p>
            <p><strong>Statistically Significant Factors (1% level):</strong> {significant_factors_1}</p>
            <p><strong>Statistically Significant Factors (5% level):</strong> {significant_factors_5}</p>
            <p><strong>Positive Performing Factors:</strong> {positive_factors}</p>
            <p><strong>Best Performing Factor:</strong> {best_factor['name']} (Sharpe: {best_factor['results']['tmb_sharpe']:.3f})</p>
            <p><strong>Worst Performing Factor:</strong> {worst_factor['name']} (Sharpe: {worst_factor['results']['tmb_sharpe']:.3f})</p>
        </div>
        """

        return summary_html

    def _generate_summary_table(self, factors_results):
        """Generate performance summary table"""
        table_html = """
        <table>
            <tr>
                <th>Factor</th>
                <th>Mean IC</th>
                <th>IC IR</th>
                <th>Hit Ratio</th>
                <th>TMB Return (Ann)</th>
                <th>TMB Sharpe</th>
                <th>TMB t-stat</th>
                <th>Significance</th>
            </tr>
        """

        for result in factors_results:
            results = result['results']
            factor_name = result['name']

            # Determine significance level and styling
            tmb_tstat = abs(results['tmb_tstat'])
            if tmb_tstat > 2.58:
                significance = "***"
                sig_class = "significance-1"
            elif tmb_tstat > 1.96:
                significance = "**"
                sig_class = "significance-5"
            elif tmb_tstat > 1.65:
                significance = "*"
                sig_class = "significance-10"
            else:
                significance = ""
                sig_class = ""

            # Determine color for returns
            return_class = "positive" if results['tmb_mean_return'] > 0 else "negative"

            table_html += f"""
            <tr class="{sig_class}">
                <td><b>{factor_name}</b></td>
                <td>{results['ic_mean']:.4f}</td>
                <td>{results['ic_ir']:.4f}</td>
                <td>{results.get('hit_ratio', 0):.3f}</td>
                <td class="{return_class}">{results['tmb_mean_return'] * 252:.2f}%</td>
                <td>{results['tmb_sharpe']:.3f}</td>
                <td>{results['tmb_tstat']:.2f}</td>
                <td>{significance}</td>
            </tr>
            """

        table_html += "</table>"
        return table_html

    def _generate_factor_group_analysis(self, factors_results):
        """Generate factor group analysis"""
        # Group factors by type
        factor_groups = {
            'Momentum': [],
            'Value': [],
            'Quality': [],
            'Growth': [],
            'Size': [],
            'Volatility': [],
            'Technical': [],
            'Reversal': []
        }

        for result in factors_results:
            factor_name = result['name'].upper()
            if 'MOM' in factor_name:
                factor_groups['Momentum'].append(result)
            elif 'VALUE' in factor_name:
                factor_groups['Value'].append(result)
            elif 'QUALITY' in factor_name:
                factor_groups['Quality'].append(result)
            elif 'GROWTH' in factor_name:
                factor_groups['Growth'].append(result)
            elif 'SIZE' in factor_name:
                factor_groups['Size'].append(result)
            elif 'VOL' in factor_name:
                factor_groups['Volatility'].append(result)
            elif any(tech in factor_name for tech in ['RSI', 'MACD', 'BOLL', 'ATR']):
                factor_groups['Technical'].append(result)
            elif 'REV' in factor_name:
                factor_groups['Reversal'].append(result)
            else:
                factor_groups['Technical'].append(result)  # Default

        group_html = ""
        for group_name, factors in factor_groups.items():
            if not factors:
                continue

            avg_sharpe = np.mean([f['results']['tmb_sharpe'] for f in factors])
            positive_count = len([f for f in factors if f['results']['tmb_mean_return'] > 0])
            total_count = len(factors)

            group_html += f"""
            <div class="factor-group">
                <h3>{group_name} Factors ({total_count} factors)</h3>
                <p>Average Sharpe Ratio: <b>{avg_sharpe:.3f}</b></p>
                <p>Positive Factors: {positive_count}/{total_count} ({positive_count / total_count * 100:.1f}%)</p>
                <p>Best Factor: {max(factors, key=lambda x: x['results']['tmb_sharpe'])['name']}</p>
            </div>
            """

        return group_html

    def _generate_key_findings(self, factors_results):
        """Generate key findings section"""
        if not factors_results:
            return "<p>No findings available.</p>"

        findings = "<ul>"

        # Find best and worst factors
        best_factor = max(factors_results, key=lambda x: x['results']['tmb_sharpe'])
        worst_factor = min(factors_results, key=lambda x: x['results']['tmb_sharpe'])

        # Find most significant factors
        significant_factors = sorted(
            [f for f in factors_results if abs(f['results']['tmb_tstat']) > 1.96],
            key=lambda x: abs(x['results']['tmb_tstat']),
            reverse=True
        )[:3]

        findings += f"<li><b>Top Performing Factor:</b> {best_factor['name']} "
        findings += f"(Annual Return: {best_factor['results']['tmb_mean_return'] * 252:.2f}%, "
        findings += f"Sharpe: {best_factor['results']['tmb_sharpe']:.3f})</li>"

        findings += f"<li><b>Worst Performing Factor:</b> {worst_factor['name']} "
        findings += f"(Annual Return: {worst_factor['results']['tmb_mean_return'] * 252:.2f}%, "
        findings += f"Sharpe: {worst_factor['results']['tmb_sharpe']:.3f})</li>"

        if significant_factors:
            findings += "<li><b>Most Statistically Significant Factors:</b><ul>"
            for factor in significant_factors:
                findings += f"<li>{factor['name']} (t-stat: {factor['results']['tmb_tstat']:.2f})</li>"
            findings += "</ul></li>"

        # IC analysis findings
        high_ic_factors = [f for f in factors_results if abs(f['results']['ic_mean']) > 0.05]
        if high_ic_factors:
            findings += f"<li><b>Factors with Strong Predictive Power (|IC| > 0.05):</b> {len(high_ic_factors)} factors</li>"

        findings += "</ul>"
        return findings

    def _generate_recommendations(self, factors_results):
        """Generate investment recommendations"""
        recommendations = "<ul>"

        if not factors_results:
            recommendations += "<li>No recommendations available due to insufficient data.</li>"
            recommendations += "</ul>"
            return recommendations

        # Find top performing factors
        positive_factors = [f for f in factors_results
                            if f['results']['tmb_mean_return'] > 0 and
                            abs(f['results']['tmb_tstat']) > 1.65]

        if positive_factors:
            top_factors = sorted(positive_factors,
                                 key=lambda x: x['results']['tmb_sharpe'],
                                 reverse=True)[:3]

            recommendations += "<li><b>Recommended Factors for Long Positions:</b><ul>"
            for factor in top_factors:
                recommendations += f"<li>{factor['name']} "
                recommendations += f"(Sharpe: {factor['results']['tmb_sharpe']:.3f}, "
                recommendations += f"Annual Return: {factor['results']['tmb_mean_return'] * 252:.2f}%)</li>"
            recommendations += "</ul></li>"

        # Find factors to avoid
        negative_factors = [f for f in factors_results
                            if f['results']['tmb_mean_return'] < 0 and
                            abs(f['results']['tmb_tstat']) > 1.65]

        if negative_factors:
            worst_factors = sorted(negative_factors,
                                   key=lambda x: x['results']['tmb_sharpe'])[:3]

            recommendations += "<li><b>Factors to Avoid or Consider Shorting:</b><ul>"
            for factor in worst_factors:
                recommendations += f"<li>{factor['name']} "
                recommendations += f"(Sharpe: {factor['results']['tmb_sharpe']:.3f}, "
                recommendations += f"Annual Return: {factor['results']['tmb_mean_return'] * 252:.2f}%)</li>"
            recommendations += "</ul></li>"

        # General recommendations
        recommendations += "<li><b>General Recommendations:</b><ul>"
        recommendations += "<li>Always conduct out-of-sample testing before live trading</li>"
        recommendations += "<li>Consider factor combination and diversification</li>"
        recommendations += "<li>Monitor factor performance regularly for regime changes</li>"
        recommendations += "<li>Implement proper risk management and position sizing</li>"
        recommendations += "</ul></li>"

        recommendations += "</ul>"
        return recommendations