-- ============================================================
-- Phase 2: Portfolio Health Monitoring and Fail Rate Analysis
-- File: sql/02_portfolio_health.sql
-- Tool: DuckDB
--
-- Purpose: Build the core portfolio health monitoring layer:
--   -- Daily and weekly fail rate trends by sector and asset class
--   -- Rolling SLO compliance tracking
--   -- Counterparty risk ranking
--   -- Threshold security identification (Reg SHO Rule 203)
--
-- SQL techniques:
--   CTEs, RANK, DENSE_RANK, NTILE, LAG, running SUM OVER,
--   PERCENT_RANK, conditional aggregation, GROUPING SETS
-- ============================================================

-- -------------------------------------------------------
-- 2A: Daily aggregate fail metrics -- the operations dashboard spine
-- -------------------------------------------------------
CREATE OR REPLACE TABLE daily_portfolio_health AS
WITH daily_base AS (
    SELECT
        settlement_date,
        is_post_t1,
        regime_base_mult,
        COUNT(*)                                        AS total_fail_records,
        COUNT(DISTINCT ticker)                          AS tickers_with_fails,
        COUNT(DISTINCT counterparty_id)                 AS counterparties_with_fails,
        SUM(quantity_failed)                            AS total_shares_failed,
        SUM(fail_value_usd)                             AS total_fail_value,
        AVG(fail_rate_pct)                              AS avg_fail_rate_pct,
        MEDIAN(fail_rate_pct)                           AS median_fail_rate_pct,
        SUM(slo_breach)                                 AS slo_breaches,
        SUM(threshold_flag)                             AS threshold_securities,
        -- SLO compliance rate: Green + Yellow / Total
        ROUND(
            SUM(CASE WHEN slo_status IN ('Green','Yellow')
                     THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2)                                              AS slo_compliance_pct,
        SUM(CASE WHEN fail_severity = 'Critical'
                 THEN 1 ELSE 0 END)                     AS critical_fails,
        SUM(CASE WHEN fail_severity = 'High'
                 THEN 1 ELSE 0 END)                     AS high_fails
    FROM ftd_clean
    GROUP BY settlement_date, is_post_t1, regime_base_mult
)
SELECT
    *,
    -- 7-day rolling average fail rate (anomaly detection baseline)
    ROUND(AVG(avg_fail_rate_pct) OVER (
        ORDER BY settlement_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 4)                                               AS rolling_7d_avg_fail_rate,
    -- Rolling SLO compliance
    ROUND(AVG(slo_compliance_pct) OVER (
        ORDER BY settlement_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                               AS rolling_7d_slo_pct,
    -- Day-over-day change in fail rate
    ROUND(avg_fail_rate_pct - LAG(avg_fail_rate_pct) OVER (
        ORDER BY settlement_date
    ), 4)                                               AS fail_rate_dod_change,
    -- Running total fail value (YTD)
    SUM(total_fail_value) OVER (
        ORDER BY settlement_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                   AS ytd_fail_value,
    -- Z-score for anomaly flagging (> 2 SD = alert)
    ROUND(
        (avg_fail_rate_pct - AVG(avg_fail_rate_pct) OVER ()) /
        NULLIF(STDDEV(avg_fail_rate_pct) OVER (), 0)
    , 3)                                                AS fail_rate_zscore
FROM daily_base
ORDER BY settlement_date;

-- -------------------------------------------------------
-- 2B: Counterparty risk ranking -- who is causing the most fails?
-- -------------------------------------------------------
CREATE OR REPLACE TABLE counterparty_risk_ranking AS
WITH cp_metrics AS (
    SELECT
        counterparty_id,
        counterparty_type,
        counterparty_size,
        COUNT(*)                                        AS total_fail_events,
        COUNT(DISTINCT ticker)                          AS tickers_affected,
        COUNT(DISTINCT settlement_date)                 AS active_fail_days,
        SUM(quantity_failed)                            AS total_shares_failed,
        ROUND(SUM(fail_value_usd) / 1e6, 2)            AS fail_value_millions,
        ROUND(AVG(fail_rate_pct), 4)                    AS avg_fail_rate_pct,
        ROUND(AVG(settlement_lag_days), 2)              AS avg_settlement_lag,
        SUM(slo_breach)                                 AS slo_breaches,
        ROUND(SUM(slo_breach) * 100.0 / COUNT(*), 2)   AS slo_breach_rate_pct,
        MAX(days_failed_consecutive)                    AS max_consecutive_fails,
        SUM(CASE WHEN fail_severity = 'Critical'
                 THEN 1 ELSE 0 END)                     AS critical_fail_events
    FROM ftd_clean
    GROUP BY counterparty_id, counterparty_type, counterparty_size
)
SELECT
    *,
    RANK() OVER (ORDER BY fail_value_millions DESC)     AS rank_by_value,
    RANK() OVER (ORDER BY slo_breach_rate_pct DESC)     AS rank_by_slo_breach,
    RANK() OVER (ORDER BY avg_fail_rate_pct DESC)       AS rank_by_fail_rate,
    NTILE(4) OVER (ORDER BY fail_value_millions DESC)   AS risk_quartile,
    ROUND(PERCENT_RANK() OVER (
        ORDER BY fail_value_millions ASC
    ) * 100, 1)                                         AS risk_percentile,
    -- Combined risk score (weighted: 40% value, 40% SLO breach, 20% rate)
    ROUND(
        0.40 * PERCENT_RANK() OVER (ORDER BY fail_value_millions) +
        0.40 * PERCENT_RANK() OVER (ORDER BY slo_breach_rate_pct) +
        0.20 * PERCENT_RANK() OVER (ORDER BY avg_fail_rate_pct)
    , 4)                                                AS composite_risk_score
FROM cp_metrics
ORDER BY composite_risk_score DESC;

-- -------------------------------------------------------
-- 2C: Sector and asset class fail concentration
-- -------------------------------------------------------
CREATE OR REPLACE TABLE sector_fail_analysis AS
SELECT
    sector,
    asset_class,
    is_post_t1,
    COUNT(DISTINCT ticker)                              AS tickers,
    COUNT(*)                                            AS fail_records,
    ROUND(SUM(fail_value_usd) / 1e6, 2)                AS fail_value_M,
    ROUND(AVG(fail_rate_pct), 4)                        AS avg_fail_rate,
    ROUND(SUM(slo_breach) * 100.0 / COUNT(*), 2)       AS slo_breach_pct,
    -- Sector share of total fails
    ROUND(
        SUM(fail_value_usd) * 100.0 /
        SUM(SUM(fail_value_usd)) OVER (PARTITION BY is_post_t1)
    , 2)                                                AS pct_of_total_fails,
    -- Rank by fail concentration
    RANK() OVER (
        PARTITION BY is_post_t1
        ORDER BY SUM(fail_value_usd) DESC
    )                                                   AS sector_rank
FROM ftd_clean
GROUP BY sector, asset_class, is_post_t1
ORDER BY is_post_t1, fail_value_M DESC;

-- -------------------------------------------------------
-- 2D: Threshold security watch list (SEC Reg SHO Rule 203)
-- Securities with >= 10,000 shares failed for 5+ consecutive days
-- -------------------------------------------------------
CREATE OR REPLACE TABLE threshold_watch_list AS
WITH persistent_fails AS (
    SELECT
        ticker,
        sector,
        counterparty_id,
        settlement_date,
        quantity_failed,
        days_failed_consecutive,
        fail_value_usd,
        fail_rate_pct,
        -- Flag per SEC Rule 203(c)(6) definition
        CASE WHEN days_failed_consecutive >= 5
              AND quantity_failed >= 10000
             THEN 1 ELSE 0
        END                                             AS reg_sho_threshold
    FROM ftd_clean
)
SELECT
    ticker,
    sector,
    COUNT(DISTINCT counterparty_id)                     AS counterparties_failing,
    MAX(days_failed_consecutive)                        AS max_consecutive_days,
    SUM(reg_sho_threshold)                              AS threshold_events,
    ROUND(AVG(fail_rate_pct), 4)                        AS avg_fail_rate,
    ROUND(SUM(fail_value_usd) / 1e6, 2)                AS total_fail_value_M,
    MIN(settlement_date)                                AS first_fail_date,
    MAX(settlement_date)                                AS last_fail_date,
    CASE
        WHEN MAX(days_failed_consecutive) >= 10 THEN 'Watch -- Escalate'
        WHEN MAX(days_failed_consecutive) >= 5  THEN 'Monitor -- Reg SHO'
        ELSE 'Routine'
    END                                                 AS action_required
FROM persistent_fails
GROUP BY ticker, sector
HAVING SUM(reg_sho_threshold) > 0
    OR MAX(days_failed_consecutive) >= 5
ORDER BY max_consecutive_days DESC, total_fail_value_M DESC;

-- -------------------------------------------------------
-- 2E: T+1 vs T+2 regime comparison
-- Validate against SIFMA/DTCC After Action Report benchmarks:
--   CNS fail rate post-T+1: 2.12% (July 2024)
-- -------------------------------------------------------
SELECT
    CASE WHEN is_post_t1 = 1 THEN 'T+1 Regime (post May 28 2024)'
         ELSE 'T+2 Regime (pre May 28 2024)'
    END                                                 AS regime,
    COUNT(DISTINCT settlement_date)                     AS trading_days,
    COUNT(*)                                            AS total_fail_records,
    ROUND(AVG(fail_rate_pct), 4)                        AS avg_fail_rate_pct,
    ROUND(MEDIAN(fail_rate_pct), 4)                     AS median_fail_rate_pct,
    ROUND(AVG(settlement_lag_days), 3)                  AS avg_settlement_lag,
    ROUND(SUM(slo_breach) * 100.0 / COUNT(*), 2)       AS slo_breach_rate_pct,
    ROUND(SUM(fail_value_usd) / 1e9, 2)                AS total_fail_value_B
FROM ftd_clean
GROUP BY is_post_t1
ORDER BY is_post_t1;
