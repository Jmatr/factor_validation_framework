import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class ReportGenerator:
    def __init__(self):
        pass

    def generate_summary_report(self, factors_results, output_path='output/factor_analysis_report.html'):
        """生成HTML格式的详细报告"""

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
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Factor Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>Performance Summary</h2>
                {self._generate_summary_table(factors_results)}
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

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Detailed report saved to: {output_path}")

    def _generate_summary_table(self, factors_results):
        """生成汇总表格"""
        table_html = """
        <table>
            <tr>
                <th>Factor</th>
                <th>Mean IC</th>
                <th>IC IR</th>
                <th>TMB Return (Ann)</th>
                <th>TMB Sharpe</th>
                <th>TMB t-stat</th>
                <th>Significance</th>
            </tr>
        """

        for result in factors_results:
            results = result['results']
            factor_name = result['name']

            # 确定显著性
            tmb_tstat = abs(results['tmb_tstat'])
            if tmb_tstat > 2.58:
                significance = "***"
                sig_class = "positive" if results['tmb_mean_return'] > 0 else "negative"
            elif tmb_tstat > 1.96:
                significance = "**"
                sig_class = "positive" if results['tmb_mean_return'] > 0 else "negative"
            elif tmb_tstat > 1.65:
                significance = "*"
                sig_class = "positive" if results['tmb_mean_return'] > 0 else "negative"
            else:
                significance = ""
                sig_class = ""

            table_html += f"""
            <tr>
                <td><b>{factor_name}</b></td>
                <td>{results['ic_mean']:.4f}</td>
                <td>{results['ic_ir']:.4f}</td>
                <td class="{sig_class}">{results['tmb_mean_return'] * 252:.4f}%</td>
                <td>{results['tmb_sharpe']:.4f}</td>
                <td>{results['tmb_tstat']:.4f}</td>
                <td class="{sig_class}">{significance}</td>
            </tr>
            """

        table_html += "</table>"
        return table_html

    def _generate_key_findings(self, factors_results):
        """生成关键发现"""
        findings = "<ul>"

        best_factor = max(factors_results,
                          key=lambda x: x['results']['tmb_sharpe'])
        worst_factor = min(factors_results,
                           key=lambda x: x['results']['tmb_sharpe'])

        findings += f"<li><b>Best Performing Factor:</b> {best_factor['name']} " \
                    f"(Sharpe: {best_factor['results']['tmb_sharpe']:.4f})</li>"

        findings += f"<li><b>Worst Performing Factor:</b> {worst_factor['name']} " \
                    f"(Sharpe: {worst_factor['results']['tmb_sharpe']:.4f})</li>"

        # 检查是否有显著因子
        significant_factors = [f for f in factors_results
                               if abs(f['results']['tmb_tstat']) > 1.96]

        if significant_factors:
            findings += f"<li><b>Statistically Significant Factors:</b> " \
                        f"{len(significant_factors)} out of {len(factors_results)}</li>"
        else:
            findings += "<li><b>No statistically significant factors found</b></li>"

        findings += "</ul>"
        return findings

    def _generate_recommendations(self, factors_results):
        """生成投资建议"""
        recommendations = "<ul>"

        # 找出表现最好的因子
        positive_factors = [f for f in factors_results
                            if f['results']['tmb_mean_return'] > 0 and
                            abs(f['results']['tmb_tstat']) > 1.65]

        if positive_factors:
            best_factor = max(positive_factors,
                              key=lambda x: x['results']['tmb_sharpe'])
            recommendations += f"<li>Consider incorporating <b>{best_factor['name']}</b> " \
                               f"in your strategy (Sharpe: {best_factor['results']['tmb_sharpe']:.4f})</li>"

        # 警告表现差的因子
        negative_factors = [f for f in factors_results
                            if f['results']['tmb_mean_return'] < 0 and
                            abs(f['results']['tmb_tstat']) > 1.65]

        if negative_factors:
            worst_factor = min(negative_factors,
                               key=lambda x: x['results']['tmb_sharpe'])
            recommendations += f"<li>Avoid or short <b>{worst_factor['name']}</b> " \
                               f"(Negative performance: {worst_factor['results']['tmb_sharpe']:.4f})</li>"

        recommendations += "<li>Always conduct out-of-sample testing before live trading</li>"
        recommendations += "<li>Consider factor combination and risk management</li>"
        recommendations += "</ul>"

        return recommendations