# Business Requirements Document
## GBM Settlement Intelligence — Fail Rate Monitoring and SLO Analytics

**Document version:** 1.0
**Date:** June 2025
**Author:** Akash Singh
**Role context:** GBM Operations Analytics — Business Analyst
**Status:** Final

---

## 1. Executive Summary

This document defines the analytical requirements for the GBM Settlement Intelligence platform — a monitoring, forecasting, and predictive analytics system for trade settlement fail rates across Global Banking and Markets (GBM) operations.

The platform addresses a core operational challenge: identifying settlement failures before they breach regulatory thresholds, attributing failures to specific counterparties and operational causes, and forecasting future settlement volumes to enable proactive staffing and liquidity management.

The US equity market transitioned to T+1 settlement on May 28, 2024 (SIFMA/ICI/DTCC T+1 After Action Report, September 2024). This compressed the window available to resolve settlement exceptions from two days to one, increasing the analytical urgency of early-warning systems.

**Business sponsor:** GBM Operations Management, Salt Lake City
**Primary users:** Operations analysts, counterparty relationship managers, risk controllers

---

## 2. Business Context

### 2.1 Regulatory framework

**SEC Regulation SHO Rule 204** requires participants of a registered clearing agency to deliver securities by settlement date or close out fails by specific deadlines:
- Short sale fails: close out by beginning of regular trading hours on T+4
- Long sale fails: close out by beginning of regular trading hours on T+6

**SEC Regulation SHO Rule 203(c)(6) — Threshold Securities:** A security becomes a "threshold security" when aggregate fails exceed 10,000 shares AND exceed 0.5% of shares outstanding for five consecutive settlement days. Threshold status triggers enhanced close-out obligations and regulatory scrutiny.

**DTCC CNS (Continuous Net Settlement) benchmarks** (SIFMA/ICI/DTCC T+1 After Action Report, September 2024):
- Average CNS Fail Rate, July 2024: 2.12%
- Average DTC non-CNS Fail Rate, July 2024: 3.31%
- These figures confirm that post-T+1 fail rates remained consistent with T+2 era benchmarks

**OSC research findings** ("The Impact of T+1 Settlement on Failed Trades", Ontario Securities Commission, 2025):
- Average daily fail rates remained below 2% for non-ETF securities post-T+1
- ETF fail rates also remained below 6% daily, well within historical ranges
- No statistically significant structural change in fail rates attributable to T+1

### 2.2 Business problem statement

GBM Operations currently identifies settlement exceptions reactively — after a fail event has already occurred and potentially breached SLO thresholds. The gap between detection and resolution consumes significant analyst time and creates regulatory exposure when exceptions age past mandatory close-out deadlines.

Three specific problems drive this project:

**Problem 1 — No portfolio-level health monitoring.** Settlement performance is tracked per-trade but not aggregated into a portfolio-level view that shows trends, anomalies, and SLO compliance rates in real time.

**Problem 2 — No counterparty risk ranking.** Not all counterparties produce the same level of settlement exceptions. Without a quantified risk ranking, relationship managers cannot prioritize outreach conversations.

**Problem 3 — No predictive capability.** Settlement volume spikes (end-of-quarter rebalancing, index reconstitutions, market volatility events) are predictable in advance. Without forecasting, staffing and liquidity buffers are set reactively rather than proactively.

---

## 3. Scope

### 3.1 In scope
- Daily settlement fail rate monitoring across all GBM equity securities
- SLO compliance tracking (Green / Yellow / Red tiering per SEC Rule 204 deadlines)
- Counterparty risk ranking using a composite score (fail value, SLO breach rate, fail rate)
- Root cause analysis by fail reason, counterparty type, and sector
- SARIMAX time series forecasting of daily settlement fail volumes (30-day forward)
- XGBoost classification model to predict SLO breach risk at the trade level
- SHAP explainability layer for model output interpretability
- Executive dashboard (5 charts) and 4 CSV report outputs

### 3.2 Out of scope
- Real-time intraday settlement monitoring (requires direct DTCC feed integration)
- Fixed income and derivatives settlement (separate data pipeline required)
- Cross-border FX settlement (requires SWIFT messaging integration)
- Client-level P&L attribution of settlement fails

### 3.3 Data sources
- **Primary:** SEC Fails-to-Deliver public dataset (NSCC CNS system aggregate data, available at https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data)
- **Synthetic proxy:** Dataset parametrized from DTCC T+1 After Action Report benchmarks (CNS fail rate 2.12%, non-CNS 3.31%) and OSC research paper distributions for development and testing
- **Reference:** FINRA OTC Threshold Securities list (daily)

---

## 4. Functional Requirements

### 4.1 Data pipeline

| Req ID | Requirement | Priority |
|---|---|---|
| DP-01 | Load SEC CNS fail data from pipe-delimited files into DuckDB | High |
| DP-02 | Validate row counts, price ranges, and settlement date continuity on every load | High |
| DP-03 | Derive SLO status (Green/Yellow/Red) based on settlement lag vs Rule 204 thresholds | High |
| DP-04 | Flag threshold securities per SEC Rule 203(c)(6) definition | High |
| DP-05 | Assign fail severity tiers (Critical ≥$10M, High ≥$1M, Medium ≥$100K, Low) | Medium |
| DP-06 | Support --real flag to switch between synthetic and real SEC data without code changes | Medium |

### 4.2 Portfolio health monitoring

| Req ID | Requirement | Priority |
|---|---|---|
| PH-01 | Calculate daily aggregate fail rate across all securities and counterparties | High |
| PH-02 | Compute 7-day rolling average fail rate for trend smoothing | High |
| PH-03 | Calculate Z-score on daily fail rate to flag statistical anomalies (threshold: Z > 2) | High |
| PH-04 | Track daily SLO compliance rate (percentage of trades resolving before T+4) | High |
| PH-05 | Report T+1 vs T+2 regime comparison against DTCC published benchmarks | Medium |
| PH-06 | Produce running YTD cumulative fail value | Medium |

### 4.3 Counterparty risk ranking

| Req ID | Requirement | Priority |
|---|---|---|
| CR-01 | Rank all counterparties by composite risk score (40% fail value, 40% SLO breach rate, 20% avg fail rate) | High |
| CR-02 | Assign risk quartiles using NTILE(4) for operational triage | High |
| CR-03 | Calculate counterparty efficiency score (inverse of lag, rate, and breach) for relationship management | Medium |
| CR-04 | Export counterparty risk ranking as CSV for monthly relationship reviews | Medium |

### 4.4 Forecasting

| Req ID | Requirement | Priority |
|---|---|---|
| FC-01 | Fit SARIMAX(1,1,1)(1,0,1)[5] model on daily fail rate time series | High |
| FC-02 | Produce 30 trading day forward forecast with 90% confidence intervals | High |
| FC-03 | Report in-sample MAPE as model accuracy metric | Medium |
| FC-04 | Visualize forecast with historical context and T+1 transition marker | Medium |

### 4.5 Predictive model

| Req ID | Requirement | Priority |
|---|---|---|
| ML-01 | Train XGBoost classifier to predict SLO breach risk at trade level | High |
| ML-02 | Handle class imbalance (~6% breach rate) using scale_pos_weight | High |
| ML-03 | Evaluate model using AUC-ROC, precision, recall, and F1 on held-out test set | High |
| ML-04 | Generate SHAP values for feature importance and individual trade explainability | High |
| ML-05 | Optimize for recall over precision (missing a breach is more costly than a false alert) | Medium |

---

## 5. Non-Functional Requirements

| Req ID | Requirement |
|---|---|
| NFR-01 | Pipeline completes in under 5 minutes on standard analytics hardware |
| NFR-02 | All SQL logic runs in DuckDB — no external database dependency for development |
| NFR-03 | Code is modular: data generation, SQL phases, ML, and charting are independently runnable |
| NFR-04 | All metrics traceable to published regulatory or industry sources |
| NFR-05 | README documents the --real flag for direct SEC data substitution |

---

## 6. KPI Definitions and SLO Thresholds

### Settlement SLO status (SEC Rule 204 framework)

| Status | Condition | Action |
|---|---|---|
| Green | Settlement lag ≤ T+1 | No action required |
| Yellow | Settlement lag T+2 to T+3 | Monitor; escalate if no resolution by T+3 |
| Red | Settlement lag ≥ T+4 | Mandatory close-out; regulatory reporting may be required |

### Portfolio health KPIs

| KPI | Definition | Target | Alert |
|---|---|---|---|
| Daily fail rate | Shares failed / daily volume | ≤ 2.12% (DTCC CNS benchmark) | > 5% |
| SLO compliance rate | Trades resolving before T+4 / total | ≥ 95% | < 90% |
| Critical fail events | Trades with fail value ≥ $10M | ≤ 2 per day | > 5 per day |
| Threshold securities | Securities triggering SEC Rule 203 | 0 | Any |

---

## 7. Analytical Methodology

### 7.1 Fail rate definition

`Fail Rate = Shares Failed / Total Daily Volume for that Security`

This follows the methodology used in the OSC research paper which uses traded volume (not shares outstanding) as the denominator, consistent with how DTCC reports CNS fail rates.

### 7.2 Composite counterparty risk score

```
Score = 0.40 × PERCENT_RANK(fail_value) 
      + 0.40 × PERCENT_RANK(slo_breach_rate)
      + 0.20 × PERCENT_RANK(avg_fail_rate)
```

Higher score = higher risk. Weights reflect that financial exposure (fail value) and regulatory risk (SLO breach rate) are the primary risk dimensions, with operational frequency (fail rate) as a secondary signal.

### 7.3 Anomaly detection

Z-score on the rolling 7-day average daily fail rate, using the full historical mean and standard deviation as the baseline. Z > 2.0 triggers an anomaly flag. This methodology is consistent with the approach used in the SLO Monitoring Dashboard project.

### 7.4 Threshold security determination

Per SEC Regulation SHO Rule 203(c)(6): a security is flagged as a threshold candidate when:
1. Aggregate shares failed ≥ 10,000 in a single settlement cycle
2. The fail persists for 5 or more consecutive settlement days

This is a simplified approximation of the full threshold calculation (which also requires ≥ 0.5% of shares outstanding) applied to the available data.

---

## 8. Limitations

1. **Synthetic data for development:** The development dataset uses synthetic data parametrized from published DTCC/SEC/OSC benchmarks. Results should be interpreted as directional. The --real flag enables the pipeline to ingest actual SEC CNS data for production validation.

2. **Aggregated data structure:** The SEC public dataset reports aggregate fails per security per settlement date, not individual trade-level records. Counterparty-level analysis in the development dataset is synthetic. Production implementation would require firm-internal DTCC participant-level data.

3. **PDC proxy, not certified metric:** The fail rate metric derived from this dataset is directional, not the certified DTCC CNS fail rate. Certified rates require direct DTCC data access.

4. **T+1 transition timing:** The regime change on May 28, 2024 creates a structural break in the time series. SARIMAX models trained across both regimes should be interpreted with awareness of this break.

5. **XGBoost model recall vs precision tradeoff:** The model is optimized for recall (detecting actual breaches) at the cost of precision (some false alerts). This is the correct tradeoff for an early-warning operations system but should be communicated clearly to end users.

---

## 9. References

1. SIFMA, ICI, and DTCC. "T+1 After Action Report." September 12, 2024. https://www.sifma.org/resources/guides-playbooks/t1-after-action-report

2. Ontario Securities Commission. "The Impact of T+1 Settlement on Failed Trades." 2025. https://www.osc.ca/en/news-events/reports-and-publications/impact-t1-settlement-failed-trades

3. SEC. "Fails-to-Deliver Data." https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data

4. SEC. "Regulation SHO — Rules 203 and 204 FAQ." https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions-8

5. Fleming, Michael, et al. "Measuring Settlement Fails." Federal Reserve Bank of New York Liberty Street Economics. September 2014. https://libertystreeteconomics.newyorkfed.org/2014/09/measuring-settlement-fails/

6. DTCC. "Daily Total US Treasury and Agency Fails." https://www.dtcc.com/charts/daily-total-us-treasury-trade-fails

7. FINRA. "OTC Threshold Securities." https://www.finra.org/finra-data/browse-catalog/otc-threshold

8. Jayson, Dean. "Improving Trade Settlement Fail Prediction with Artificial Intelligence and Machine Learning." LinkedIn, August 2018.
