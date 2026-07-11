# GBM Settlement Intelligence

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![SQL](https://img.shields.io/badge/SQL-DuckDB-yellow?style=flat-square)
![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20SHAP-orange?style=flat-square)
![SARIMAX](https://img.shields.io/badge/Forecast-SARIMAX-green?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-GBM%20Operations%20Analytics-0A1628?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

End-to-end analytics pipeline for trade settlement fail rate monitoring, counterparty risk ranking, and SLO breach prediction in a Global Banking and Markets (GBM) operations context. Built on the SEC's publicly available Fails-to-Deliver dataset and the regulatory framework of SEC Regulation SHO Rules 203 and 204.

---

## The Problem

The US equity market moved to **T+1 settlement on May 28, 2024**. This compressed the window to resolve settlement exceptions from two days to one -- increasing the cost of late detection significantly. Under **SEC Regulation SHO Rule 204**, short sale fails not closed out by **T+4** trigger mandatory buy-in obligations and regulatory reporting requirements.

GBM Operations teams face three specific gaps:

**Gap 1 -- No portfolio-level health view.** Settlement exceptions are tracked per trade, not aggregated into a real-time view of where the book stands relative to SLO thresholds.

**Gap 2 -- No counterparty risk ranking.** Not all counterparties produce the same volume of fails. Without a quantified risk score, relationship managers cannot prioritize outreach conversations.

**Gap 3 -- No predictive capability.** Volume spikes from index reconstitutions, end-of-quarter rebalancing, and market volatility events are partly predictable. Without a forecast, staffing and liquidity buffers are set reactively.

This project solves all three.

---

## Key Results

| Metric | Value | Benchmark / Note |
|---|---|---|
| Records analyzed | 10,456 | Jan 2024 -- Jun 2025, 38 securities, 11 counterparties |
| SLO breach rate | 5.6% | Target below 5% -- SEC Rule 204 mandatory close-out at T+4 |
| Post-T+1 avg daily fail rate | 1.06% | DTCC CNS benchmark: 2.12% (SIFMA/DTCC/ICI, Sept 2024) |
| Highest-risk counterparty | HEDGE_B | Composite risk score: 0.76 out of 1.0 |
| XGBoost AUC-ROC | 0.750 | Recall (breach): 0.675 -- optimized for early warning |
| SARIMAX MAPE | 11.51% | 30-day forward forecast of daily fail rate |
| Top breach predictor (SHAP) | Market regime | Followed by log quantity failed and post-T+1 flag |

---

## Output 1 -- Daily Portfolio Health Monitor

Tracks daily fail rate trend, 7-day rolling average, Z-score anomaly detection, and SLO compliance over time. The T+1 transition (May 28, 2024) is marked as a regime change line. Three panels: (1) daily fail rate with rolling average and anomaly flags where Z-score exceeds 2.0; (2) 7-day rolling SLO compliance against 95% target and 90% alert threshold; (3) daily count of Critical (above $10M) and High (above $1M) fail events.

![Daily Portfolio Health](https://raw.githubusercontent.com/aksingh-ops/gbm-settlement-intelligence/main/reports/01_daily_portfolio_health.png)

---

## Output 2 -- Counterparty Risk Matrix

Ranks all 11 counterparties by a composite risk score combining fail value (40%), SLO breach rate (40%), and average fail rate (20%). Left panel: horizontal bar chart ranked by composite risk score, color-coded by quartile. Right panel: scatter matrix of fail value vs SLO breach rate, bubble size representing average settlement lag days.

![Counterparty Risk Matrix](https://raw.githubusercontent.com/aksingh-ops/gbm-settlement-intelligence/main/reports/02_counterparty_risk_matrix.png)

---

## Output 3 -- SARIMAX Fail Rate Forecast

SARIMAX(1,1,1)(1,0,1)[5] model fit on 18 months of daily fail rate history, with a 30-trading-day forward forecast and 90% confidence interval. MAPE of 11.51% reflects the inherent volatility of settlement fail rates. The DTCC CNS benchmark of 2.12% is shown as a reference line.

![SARIMAX Forecast](https://raw.githubusercontent.com/aksingh-ops/gbm-settlement-intelligence/main/reports/03_sarimax_forecast.png)

---

## Output 4 -- XGBoost SLO Breach Prediction

SLO breach prediction model trained on 10,456 trade-level records. Optimized for recall (catching actual breaches before they trigger mandatory close-out). Three panels: ROC curve (AUC 0.750); confusion matrix on 20% held-out test set; SHAP feature importance showing the top 10 predictors of SLO breach. Market regime, log quantity failed, and the post-T+1 flag are the three strongest signals.

![XGBoost Model Performance](https://raw.githubusercontent.com/aksingh-ops/gbm-settlement-intelligence/main/reports/04_xgboost_model_performance.png)

---

## Output 5 -- Executive Dashboard

Single-page summary designed for GBM Operations leadership. Combines all four analytical layers: four KPI tiles (total fail value, SLO breach rate, post-T+1 avg fail rate, XGBoost AUC), 60-day fail rate trend with anomaly flags, top-6 counterparty risk ranking, SLO breach rate by fail reason, and top-5 SHAP predictors.

![Executive Dashboard](https://raw.githubusercontent.com/aksingh-ops/gbm-settlement-intelligence/main/reports/05_executive_dashboard.png)

---

## How to Run

```bash
git clone https://github.com/aksingh-ops/gbm-settlement-intelligence
cd gbm-settlement-intelligence
pip install -r requirements.txt

# Run with synthetic data (parametrized from published DTCC/SEC/OSC benchmarks)
python run_pipeline.py

# Run with real SEC data
# 1. Download from: https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data
# 2. Place cnsfails{YYYYMM}{a|b}.zip files in data/raw/ -- do not unzip
# 3. Run:
python run_pipeline.py --real
```

One command produces all 9 outputs in `reports/`.

---

## Project Structure

```
gbm-settlement-intelligence/
|
|-- generate_data.py              Synthetic dataset from DTCC/SEC/OSC parameters
|                                 or load real SEC CNS data with --real flag
|-- run_pipeline.py               One command: data -> SQL -> ML -> charts
|-- requirements.txt
|
|-- sql/
|   |-- 01_load_and_validate.sql  Load, quality checks, SLO status derivation
|   |-- 02_portfolio_health.sql   Daily health, counterparty ranking, T+1 comparison
|   `-- 03_root_cause_analysis.sql Fail reason, GROUPING SETS rollup, SLO attribution
|
|-- docs/
|   `-- business_requirements_document.md  Full BRD with regulatory context
|
|-- data/
|   |-- raw/                      Place real SEC CNS zip files here
|   `-- processed/                Generated by pipeline
|
`-- reports/
    |-- 01_daily_portfolio_health.png
    |-- 02_counterparty_risk_matrix.png
    |-- 03_sarimax_forecast.png
    |-- 04_xgboost_model_performance.png
    |-- 05_executive_dashboard.png
    |-- daily_portfolio_health.csv
    |-- counterparty_risk_ranking.csv
    |-- fail_reason_analysis.csv
    `-- sector_fail_analysis.csv
```

---

## Phase-by-Phase Breakdown

<table>
  <thead>
    <tr>
      <th>Phase</th>
      <th>File</th>
      <th>What it does</th>
      <th>Key output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 -- BRD</strong></td>
      <td>docs/business_requirements_document.md</td>
      <td>Full requirements document written before any code. Regulatory context, KPI definitions, SLO thresholds, methodology.</td>
      <td>Documents the why before the how</td>
    </tr>
    <tr>
      <td><strong>2 -- Data</strong></td>
      <td>generate_data.py</td>
      <td>Generates 10,456 synthetic FTD records parametrized from DTCC/SEC/OSC published benchmarks. Supports real SEC data via --real flag with no code changes.</td>
      <td>ftd_master.csv -- 20 fields, Jan 2024 to Jun 2025</td>
    </tr>
    <tr>
      <td><strong>3 -- SQL Phase 1</strong></td>
      <td>sql/01_load_and_validate.sql</td>
      <td>Load raw CSV into DuckDB, run 4-layer quality checks (row count, price range, lag validity, null audit), derive SLO Green/Yellow/Red status per Rule 204, assign severity tiers.</td>
      <td>ftd_clean table -- validated and enriched</td>
    </tr>
    <tr>
      <td><strong>4 -- SQL Phase 2</strong></td>
      <td>sql/02_portfolio_health.sql</td>
      <td>Daily aggregate health monitoring, 7-day rolling averages, Z-score anomaly detection, counterparty risk ranking with NTILE quartiles, sector concentration, T+1 vs T+2 regime comparison against DTCC benchmarks.</td>
      <td>RANK, DENSE_RANK, NTILE, PERCENT_RANK, LAG, cumulative SUM OVER, STDDEV window</td>
    </tr>
    <tr>
      <td><strong>5 -- SQL Phase 3</strong></td>
      <td>sql/03_root_cause_analysis.sql</td>
      <td>Fail reason breakdown by counterparty type, GROUPING SETS rollup across reason x sector x counterparty, monthly trend with LAG, SLO breach attribution, counterparty efficiency scoring.</td>
      <td>GROUPING SETS, multi-step CTEs, HAVING filter for volume reliability</td>
    </tr>
    <tr>
      <td><strong>6 -- Forecast</strong></td>
      <td>run_pipeline.py</td>
      <td>SARIMAX(1,1,1)(1,0,1)[5] fitted on 18 months of daily fail rate. 30-trading-day forward forecast with 90% confidence interval. Detects T+1 regime shift at May 28, 2024.</td>
      <td>MAPE: 11.51%</td>
    </tr>
    <tr>
      <td><strong>7 -- ML</strong></td>
      <td>run_pipeline.py</td>
      <td>XGBoost classifier on 12 engineered features predicting SLO breach at trade level. Class imbalance handled via scale_pos_weight. SHAP for feature importance and individual trade explainability.</td>
      <td>AUC-ROC: 0.750 | Recall: 0.675</td>
    </tr>
    <tr>
      <td><strong>8 -- Dashboard</strong></td>
      <td>run_pipeline.py</td>
      <td>5 production-quality charts plus 4 CSV exports. GBM color palette (navy #0A1628, gold #C9A84C).</td>
      <td>9 files in reports/</td>
    </tr>
  </tbody>
</table>

---

## SQL Techniques Demonstrated

<table>
  <thead>
    <tr>
      <th>Technique</th>
      <th>File</th>
      <th>Purpose in this project</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>RANK / DENSE_RANK / NTILE</td><td>02, 03</td><td>Counterparty risk quartiles and fail reason ranking within counterparty type</td></tr>
    <tr><td>PERCENT_RANK</td><td>02, 03</td><td>Composite risk score components and counterparty efficiency scoring</td></tr>
    <tr><td>LAG + cumulative SUM OVER</td><td>02, 03</td><td>Day-over-day fail rate change, YTD cumulative fail value, monthly MoM trend</td></tr>
    <tr><td>STDDEV OVER (Z-score)</td><td>02</td><td>Statistical anomaly detection on rolling daily fail rate</td></tr>
    <tr><td>GROUPING SETS</td><td>03</td><td>Multi-level rollup: reason x sector x counterparty type in one query</td></tr>
    <tr><td>COALESCE / NULLIF</td><td>01, 02</td><td>Safe division, suppressed value handling, null-safe aggregation</td></tr>
    <tr><td>Conditional aggregation</td><td>02, 03</td><td>SLO status counts and severity tier breakdown within a single GROUP BY pass</td></tr>
    <tr><td>Multi-step CTEs</td><td>02, 03</td><td>Staged pipeline: aggregate -> rank -> score -> export without temp tables</td></tr>
    <tr><td>HAVING</td><td>03</td><td>Minimum volume filter (n >= 10) for statistical reliability in breach attribution</td></tr>
  </tbody>
</table>

---

## Data Sources and References

The development dataset is parametrized from these published sources. The `--real` flag substitutes actual SEC data with no code changes required.

| Source | How it is used |
|---|---|
| SIFMA/ICI/DTCC T+1 After Action Report (Sept 12, 2024) | CNS fail rate benchmark (2.12%), T+1 transition date, non-CNS benchmark (3.31%) |
| OSC "Impact of T+1 Settlement on Failed Trades" (2025) | Fail rate distribution by security type, ETF vs non-ETF segmentation |
| SEC Regulation SHO Rule 204 | SLO close-out deadlines: T+4 short sales, T+6 long sales |
| SEC Regulation SHO Rule 203(c)(6) | Threshold security: 10,000 shares for 5 consecutive settlement days |
| NY Fed Liberty Street Economics, "Measuring Settlement Fails" (Sept 2014) | DTCC CNS data methodology and historical context |
| SEC Fails-to-Deliver Data (public, updated twice monthly) | Primary real data source |
| FINRA OTC Threshold Securities list (daily) | Reference for threshold security identification |
| Jayson, D. "Improving Trade Settlement Fail Prediction with AI/ML" (LinkedIn, 2018) | Industry precedent for ML on settlement fails |

---

## Limitations

**Synthetic data for development.** The development dataset is generated from published DTCC/SEC/OSC parameters, not direct DTCC participant-level data. Use `--real` with SEC public files for production validation.

**Aggregated SEC public data.** The SEC public FTD dataset reports aggregate daily fails per security across all NSCC members -- not individual counterparty records. The counterparty dimension in this project is a synthetic enrichment layer. Production implementation requires internal DTCC participant-level access.

**SARIMAX MAPE 11.51%.** Settlement fail rates are inherently noisy. This MAPE reflects the difficulty of forecasting a high-variance operational metric, not a model failure.

**XGBoost recall vs precision.** The model prioritizes catching actual breaches (recall 0.675) over minimizing false alerts (precision 0.148). This is the correct tradeoff for an early-warning operations system and is documented in the BRD.

---

## Author

**Akash Singh**
M.S. Business Analytics -- Iowa State University
[github.com/aksingh-ops](https://github.com/aksingh-ops) | [Portfolio](https://aksingh-ops.github.io)
