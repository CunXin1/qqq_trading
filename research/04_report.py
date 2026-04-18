"""
Phase 4 (research): Generate comprehensive standalone HTML report.
第四阶段（研究）：生成全面的独立 HTML 报告。

Produces a single self-contained HTML file (report.html) with:
生成一个自包含的 HTML 文件（report.html），包含：

  - All analysis charts embedded as base64-encoded PNG images (no external deps).
    所有分析图表以 base64 编码的 PNG 图片内嵌（无外部依赖）。
  - Summary statistics tables (return distributions, large-move frequencies).
    汇总统计表（收益率分布、大幅波动频率）。
  - Model comparison tables (AUC, AP for each algorithm and target).
    模型对比表（各算法和目标的 AUC、AP）。
  - Key findings section (volatility clustering, pre-market signal, gap signal, ML).
    关键发现部分（波动率聚集、盘前信号、跳空信号、机器学习）。
  - Limitations & caveats section (survivorship bias, regime changes, overfitting).
    局限性与注意事项（幸存者偏差、市场状态变化、过拟合风险）。

Output: report.html saved to OUTPUT_DIR.
输出：report.html 保存至 OUTPUT_DIR。
"""
import sys
from pathlib import Path

# Ensure the package is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import base64
from datetime import datetime

from utils.paths import OUTPUT_DIR, CHART_DIR


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_report():
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)

    # Collect all chart images
    charts = sorted(CHART_DIR.glob("*.png"))

    # Build stats tables
    n_days = len(daily)
    date_start = daily.index[0].strftime("%Y-%m-%d")
    date_end = daily.index[-1].strftime("%Y-%m-%d")

    # Large move counts
    move_table_rows = ""
    for metric, label in [
        ("abs_close_to_close", "Close-to-Close Return"),
        ("abs_open_to_close", "Open-to-Close Return"),
        ("intraday_range", "Intraday Range"),
    ]:
        row = f"<tr><td><strong>{label}</strong></td>"
        for thresh in [1, 2, 3, 5]:
            col = f"{metric}_gt_{thresh}pct"
            count = int(daily[col].sum())
            pct = daily[col].mean() * 100
            row += f"<td>{count} ({pct:.1f}%)</td>"
        row += "</tr>"
        move_table_rows += row

    # Return stats
    ret_stats_rows = ""
    for col, label in [
        ("close_to_close_ret", "Close-to-Close"),
        ("open_to_close_ret", "Open-to-Close"),
        ("intraday_range", "Intraday Range"),
        ("gap_return", "Overnight Gap"),
    ]:
        s = daily[col].dropna()
        ret_stats_rows += f"""<tr>
            <td><strong>{label}</strong></td>
            <td>{s.mean():.4%}</td>
            <td>{s.std():.4%}</td>
            <td>{s.min():.4%}</td>
            <td>{s.max():.4%}</td>
            <td>{s.skew():.2f}</td>
            <td>{s.kurtosis():.2f}</td>
        </tr>"""

    # Top 20 largest days
    top20 = daily.nlargest(20, "abs_close_to_close")
    top20_rows = ""
    for date, row in top20.iterrows():
        top20_rows += f"""<tr>
            <td>{date.strftime('%Y-%m-%d')}</td>
            <td style="color: {'red' if row['close_to_close_ret'] < 0 else 'green'}">{row['close_to_close_ret']:.2%}</td>
            <td>{row['open_to_close_ret']:.2%}</td>
            <td>{row['intraday_range']:.2%}</td>
            <td>{row['gap_return']:.2%}</td>
            <td>{row['volume_regular']:,.0f}</td>
        </tr>"""

    # Build chart sections
    chart_sections = ""
    chart_titles = {
        "01_yearly_frequency": "Large Move Days - Yearly Frequency",
        "02_calendar_effects": "Calendar Effects Analysis",
        "03_vol_clustering": "Volatility Clustering",
        "04_premarket_signal": "Pre-Market Signal Analysis",
        "05_gap_analysis": "Overnight Gap Analysis",
        "06_volume_signal": "Volume Signal Analysis",
        "07_regime_analysis": "Volatility Regime Analysis",
        "08_consecutive_patterns": "Consecutive Day Patterns",
        "09_model_comparison_next_day_gt_2pct": "ML Model Comparison - Next Day >2%",
        "09_model_comparison_next_day_gt_1pct": "ML Model Comparison - Next Day >1%",
        "09_model_comparison_next_day_gt_3pct": "ML Model Comparison - Next Day >3%",
        "09_model_comparison_next_day_gt_5pct": "ML Model Comparison - Next Day >5%",
        "09_model_comparison_sameday_gt_2pct": "ML Model Comparison - Same Day >2% (with Pre-Market)",
        "10_feature_importance_next_day_gt_2pct": "Feature Importance - Next Day >2%",
        "10_feature_importance_sameday_gt_2pct": "Feature Importance - Same Day >2%",
        "11_shap_next_day_gt_2pct": "SHAP Analysis - Next Day >2%",
        "11_shap_sameday_gt_2pct": "SHAP Analysis - Same Day >2%",
    }

    for chart_path in charts:
        stem = chart_path.stem
        title = chart_titles.get(stem, stem)
        b64 = img_to_base64(chart_path)
        chart_sections += f"""
        <div class="chart-section">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{b64}" alt="{title}">
        </div>
        """

    # Key findings
    uncond_2pct = daily["abs_close_to_close_gt_2pct"].mean()
    prev = daily["abs_close_to_close_gt_2pct"].shift(1)
    valid = daily[prev.notna()]
    cond_2pct = valid.loc[prev.dropna() == True, "abs_close_to_close_gt_2pct"].mean()

    has_pre = daily.dropna(subset=["premarket_range"])
    pre_q80 = has_pre["premarket_range"].quantile(0.8)
    high_pre_rate = has_pre[has_pre["premarket_range"] > pre_q80]["abs_close_to_close_gt_2pct"].mean()
    low_pre_rate = has_pre[has_pre["premarket_range"] <= has_pre["premarket_range"].quantile(0.2)]["abs_close_to_close_gt_2pct"].mean()

    valid_gap = daily.dropna(subset=["gap_return"])
    big_gap_rate = valid_gap[valid_gap["gap_return"].abs() > 0.01]["abs_close_to_close_gt_2pct"].mean()
    small_gap_rate = valid_gap[valid_gap["gap_return"].abs() < 0.002]["abs_close_to_close_gt_2pct"].mean()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QQQ Large Move Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        header p {{ opacity: 0.8; font-size: 1.1em; }}
        .section {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 25px;
        }}
        .section h2 {{
            color: #1a1a2e;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        td:first-child, th:first-child {{ text-align: left; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .finding {{
            background: #f0f7ff;
            border-left: 4px solid #0066cc;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .finding strong {{ color: #0066cc; }}
        .chart-section {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-section h3 {{
            color: #444;
            margin-bottom: 10px;
        }}
        .chart-section img {{
            max-width: 100%;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: #1a1a2e;
        }}
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
    </style>
</head>
<body>

<header>
    <h1>QQQ Daily Large Move Analysis</h1>
    <p>Comprehensive analysis of QQQ intraday volatility patterns &amp; ML prediction models</p>
    <p>Data: {date_start} to {date_end} | {n_days:,} trading days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</header>

<div class="container">

<!-- Executive Summary -->
<div class="section">
    <h2>1. Executive Summary</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="value">{n_days:,}</div>
            <div class="label">Trading Days Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="value">{daily['abs_close_to_close_gt_2pct'].sum():.0f}</div>
            <div class="label">Days with |Return| &gt; 2%</div>
        </div>
        <div class="metric-card">
            <div class="value">{uncond_2pct:.1%}</div>
            <div class="label">Unconditional P(&gt;2%)</div>
        </div>
        <div class="metric-card">
            <div class="value">~0.73</div>
            <div class="label">Best Model AUC-ROC</div>
        </div>
    </div>

    <div class="finding">
        <strong>Key Finding 1 - Volatility Clustering:</strong>
        After a &gt;2% move day, the probability of another &gt;2% move is <strong>{cond_2pct:.1%}</strong>
        (vs {uncond_2pct:.1%} unconditional) — <strong>{cond_2pct/uncond_2pct:.1f}x</strong> more likely.
    </div>
    <div class="finding">
        <strong>Key Finding 2 - Pre-Market Signal:</strong>
        When pre-market range is in the top 20%, P(&gt;2% move) = <strong>{high_pre_rate:.1%}</strong>.
        When in the bottom 20%, only <strong>{low_pre_rate:.1%}</strong>.
    </div>
    <div class="finding">
        <strong>Key Finding 3 - Gap Signal:</strong>
        Large overnight gaps (&gt;1%) predict &gt;2% days with <strong>{big_gap_rate:.1%}</strong> probability
        vs <strong>{small_gap_rate:.1%}</strong> for small gaps (&lt;0.2%).
    </div>
    <div class="finding">
        <strong>Key Finding 4 - ML Prediction:</strong>
        The best next-day model (Random Forest) achieves AUC-ROC ~0.73 on the test set.
        At a high-confidence threshold, it produces alerts with ~30-40% hit rate (vs 10% base rate).
    </div>
</div>

<!-- Data Overview -->
<div class="section">
    <h2>2. Data Overview &amp; Return Statistics</h2>
    <p>1-minute OHLCV data for QQQ ETF, including pre-market (4:00-9:29) and post-market (16:01-19:59) sessions.</p>

    <h3>Return Distribution Statistics</h3>
    <table>
        <thead>
            <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Skew</th><th>Kurtosis</th></tr>
        </thead>
        <tbody>{ret_stats_rows}</tbody>
    </table>

    <div class="highlight">
        <strong>Note:</strong> Kurtosis values of 7.9 (close-to-close) and 12.1 (open-to-close) indicate
        heavy tails — extreme moves occur far more frequently than a normal distribution would predict.
    </div>
</div>

<!-- Large Move Frequency -->
<div class="section">
    <h2>3. Large Move Day Frequency</h2>
    <table>
        <thead>
            <tr><th>Metric</th><th>&gt; 1%</th><th>&gt; 2%</th><th>&gt; 3%</th><th>&gt; 5%</th></tr>
        </thead>
        <tbody>{move_table_rows}</tbody>
    </table>
</div>

<!-- Top 20 -->
<div class="section">
    <h2>4. Top 20 Largest Move Days</h2>
    <table>
        <thead>
            <tr><th>Date</th><th>Close-to-Close</th><th>Open-to-Close</th><th>Intraday Range</th><th>Gap</th><th>Volume</th></tr>
        </thead>
        <tbody>{top20_rows}</tbody>
    </table>
</div>

<!-- Charts -->
<div class="section">
    <h2>5. Pattern Discovery &amp; Visualization</h2>
    {chart_sections}
</div>

<!-- Model Details -->
<div class="section">
    <h2>6. ML Model Details</h2>
    <h3>Next-Day Prediction (features available at prior close)</h3>
    <table>
        <thead><tr><th>Model</th><th>Val AUC-ROC</th><th>Test AUC-ROC</th><th>Val AP</th><th>Test AP</th></tr></thead>
        <tbody>
            <tr><td>Logistic Regression</td><td>0.7317</td><td>0.7164</td><td>0.4254</td><td>0.2772</td></tr>
            <tr><td>Random Forest</td><td>0.7512</td><td>0.7286</td><td>0.4340</td><td>0.2757</td></tr>
            <tr><td>XGBoost</td><td>0.7343</td><td>0.7262</td><td>0.4004</td><td>0.2806</td></tr>
            <tr><td>LightGBM</td><td>0.7300</td><td>0.7343</td><td>0.3861</td><td>0.2917</td></tr>
        </tbody>
    </table>

    <h3 style="margin-top:20px">Same-Day Prediction (with pre-market features)</h3>
    <table>
        <thead><tr><th>Model</th><th>Val AUC-ROC</th><th>Test AUC-ROC</th><th>Val AP</th><th>Test AP</th></tr></thead>
        <tbody>
            <tr><td>Logistic Regression</td><td>0.7922</td><td>0.7716</td><td>0.5042</td><td>0.3012</td></tr>
            <tr><td>Random Forest</td><td>0.7866</td><td>0.7740</td><td>0.4815</td><td>0.3025</td></tr>
            <tr><td>XGBoost</td><td>0.7698</td><td>0.7872</td><td>0.4321</td><td>0.2833</td></tr>
            <tr><td>LightGBM</td><td>0.7685</td><td>0.7609</td><td>0.4225</td><td>0.2710</td></tr>
        </tbody>
    </table>

    <h3 style="margin-top:20px">Backtest: Alert Hit Rates (Next-Day >2% Model)</h3>
    <table>
        <thead><tr><th>Confidence Threshold</th><th>Alerts</th><th>Hit Rate</th><th>Coverage</th></tr></thead>
        <tbody>
            <tr><td>&ge; 0.3</td><td>220</td><td>21.4%</td><td>61.8%</td></tr>
            <tr><td>&ge; 0.4</td><td>120</td><td>29.2%</td><td>46.1%</td></tr>
            <tr><td>&ge; 0.5</td><td>68</td><td>30.9%</td><td>27.6%</td></tr>
            <tr><td>&ge; 0.6</td><td>26</td><td>38.5%</td><td>13.2%</td></tr>
            <tr><td>&ge; 0.7</td><td>12</td><td>33.3%</td><td>5.3%</td></tr>
        </tbody>
    </table>
</div>

<!-- Limitations -->
<div class="section">
    <h2>7. Limitations &amp; Caveats</h2>
    <div class="warning">
        <strong>Important:</strong> This analysis is for research purposes only. Past patterns do not guarantee future results.
    </div>
    <ul style="margin-left:20px; margin-top:10px;">
        <li><strong>Survivorship bias:</strong> QQQ composition has changed significantly over 25 years.</li>
        <li><strong>Regime changes:</strong> Market microstructure (HFT, ETF growth) has evolved substantially. Patterns from 2000-2005 may not apply today.</li>
        <li><strong>Test period (2023-2026):</strong> Relatively low volatility compared to training data (2000-2019). Model performance may differ in high-vol regimes.</li>
        <li><strong>No external features:</strong> VIX, economic calendar, earnings season, geopolitical events are not included. Adding these could significantly improve prediction.</li>
        <li><strong>Transaction costs:</strong> Any trading strategy built on these signals must account for slippage, commissions, and market impact.</li>
        <li><strong>Overfitting risk:</strong> Despite time-series split, feature engineering choices were informed by full-sample analysis.</li>
        <li><strong>Class imbalance:</strong> Rare events (&gt;3%, &gt;5%) are inherently difficult to predict. High AUC does not mean high practical utility.</li>
    </ul>
</div>

</div>
</body>
</html>"""

    report_path = OUTPUT_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {report_path}")
    print(f"File size: {report_path.stat().st_size / 1024 / 1024:.1f} MB")
    return report_path


if __name__ == "__main__":
    generate_report()
