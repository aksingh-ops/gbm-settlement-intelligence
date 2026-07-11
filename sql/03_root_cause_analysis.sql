-- ============================================================
-- Phase 3: Fail Reason and Operational Root Cause Analysis
-- File: sql/03_root_cause_analysis.sql
-- Tool: DuckDB
--
-- Purpose: Break down fail events by operational root cause,
--   identify systemic patterns, and produce actionable
--   recommendations for operations management.
--
-- SQL techniques:
--   GROUPING SETS, ROLLUP, conditional aggregation,
--   FIRST_VALUE, LAG for trend detection, NTILE quartiles
-- ============================================================

-- -------------------------------------------------------
-- 3A: Fail reason breakdown -- where is the operational gap?
-- -------------------------------------------------------
CREATE OR REPLACE TABLE fail_reason_analysis AS
SELECT
    fail_reason,
    counterparty_type,
    COUNT(*)                                            AS occurrences,
    ROUND(SUM(fail_value_usd) / 1e6, 2)                AS fail_value_M,
    ROUND(AVG(fail_rate_pct), 4)                        AS avg_fail_rate,
    ROUND(AVG(settlement_lag_days), 2)                  AS avg_lag_days,
    ROUND(SUM(slo_breach) * 100.0 / COUNT(*), 2)       AS slo_breach_pct,
    COUNT(DISTINCT ticker)                              AS tickers_affected,
    COUNT(DISTINCT counterparty_id)                     AS counterparties_involved,
    -- Rank within each counterparty type
    RANK() OVER (
        PARTITION BY counterparty_type
        ORDER BY SUM(fail_value_usd) DESC
    )                                                   AS rank_within_cp_type,
    -- Share of fail value for this reason
    ROUND(
        SUM(fail_value_usd) * 100.0 /
        SUM(SUM(fail_value_usd)) OVER (PARTITION BY counterparty_type)
    , 2)                                                AS pct_of_cp_type_fails
FROM ftd_clean
GROUP BY fail_reason, counterparty_type
ORDER BY counterparty_type, fail_value_M DESC;

-- -------------------------------------------------------
-- 3B: Multi-level rollup -- reason x sector x counterparty type
-- Uses GROUPING SETS to produce multiple aggregation levels in one query
-- -------------------------------------------------------
CREATE OR REPLACE TABLE fail_reason_rollup AS
SELECT
    COALESCE(fail_reason, 'ALL REASONS')                AS fail_reason,
    COALESCE(sector, 'ALL SECTORS')                     AS sector,
    COALESCE(counterparty_type, 'ALL COUNTERPARTIES')   AS counterparty_type,
    COUNT(*)                                            AS records,
    ROUND(SUM(fail_value_usd) / 1e6, 2)                AS fail_value_M,
    ROUND(AVG(slo_breach) * 100, 2)                     AS slo_breach_pct,
    GROUPING(fail_reason, sector, counterparty_type)    AS grouping_level
FROM ftd_clean
GROUP BY GROUPING SETS (
    (fail_reason, sector, counterparty_type),   -- most granular
    (fail_reason, sector),                       -- reason + sector
    (fail_reason),                               -- reason only
    ()                                           -- grand total
)
ORDER BY grouping_level, fail_value_M DESC;

-- -------------------------------------------------------
-- 3C: Monthly trend by fail reason
-- Detect whether specific operational issues are worsening
-- -------------------------------------------------------
CREATE OR REPLACE TABLE fail_reason_monthly_trend AS
WITH monthly_reason AS (
    SELECT
        month_start,
        fail_reason,
        COUNT(*)                                        AS occurrences,
        ROUND(SUM(fail_value_usd) / 1e6, 2)            AS fail_value_M,
        ROUND(AVG(settlement_lag_days), 2)              AS avg_lag
    FROM ftd_clean
    GROUP BY month_start, fail_reason
)
SELECT
    month_start,
    fail_reason,
    occurrences,
    fail_value_M,
    avg_lag,
    -- Month-over-month change
    ROUND(fail_value_M - LAG(fail_value_M) OVER (
        PARTITION BY fail_reason
        ORDER BY month_start
    ), 2)                                               AS mom_change_M,
    -- Running total per reason
    SUM(fail_value_M) OVER (
        PARTITION BY fail_reason
        ORDER BY month_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                   AS cumulative_fail_value_M,
    RANK() OVER (
        PARTITION BY month_start
        ORDER BY fail_value_M DESC
    )                                                   AS monthly_rank
FROM monthly_reason
ORDER BY month_start, fail_value_M DESC;

-- -------------------------------------------------------
-- 3D: SLO breach attribution -- which combination of
--   counterparty type + fail reason + sector produces
--   the most regulatory exposure?
-- -------------------------------------------------------
CREATE OR REPLACE TABLE slo_breach_attribution AS
SELECT
    counterparty_type,
    fail_reason,
    sector,
    COUNT(*)                                            AS total_records,
    SUM(slo_breach)                                     AS slo_breaches,
    ROUND(SUM(slo_breach) * 100.0 / COUNT(*), 2)       AS breach_rate_pct,
    ROUND(SUM(
        CASE WHEN slo_breach = 1 THEN fail_value_usd ELSE 0 END
    ) / 1e6, 2)                                         AS breach_value_M,
    -- Risk tier: High (>20% breach rate), Medium (10-20%), Low (<10%)
    CASE
        WHEN SUM(slo_breach) * 100.0 / COUNT(*) >= 20 THEN 'High Risk'
        WHEN SUM(slo_breach) * 100.0 / COUNT(*) >= 10 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END                                                 AS risk_tier,
    RANK() OVER (
        ORDER BY SUM(slo_breach) * 100.0 / COUNT(*) DESC
    )                                                   AS breach_rate_rank
FROM ftd_clean
GROUP BY counterparty_type, fail_reason, sector
HAVING COUNT(*) >= 10  -- minimum volume threshold for statistical reliability
ORDER BY breach_rate_pct DESC
LIMIT 25;

-- -------------------------------------------------------
-- 3E: Operational efficiency score per counterparty
-- Combining breach rate, avg lag, and fail rate into a single score
-- for performance management conversations
-- -------------------------------------------------------
WITH cp_agg AS (
    SELECT
        counterparty_id,
        counterparty_type,
        ROUND(AVG(settlement_lag_days), 3)              AS avg_lag,
        ROUND(AVG(fail_rate_pct), 4)                    AS avg_fail_rate,
        ROUND(AVG(slo_breach) * 100, 2)                 AS slo_breach_pct
    FROM ftd_clean
    GROUP BY counterparty_id, counterparty_type
),
scored AS (
    SELECT *,
        ROUND(
            (1 - PERCENT_RANK() OVER (ORDER BY avg_lag))        * 0.35 +
            (1 - PERCENT_RANK() OVER (ORDER BY avg_fail_rate))  * 0.35 +
            (1 - PERCENT_RANK() OVER (ORDER BY slo_breach_pct)) * 0.30
        , 4)                                            AS efficiency_score
    FROM cp_agg
)
SELECT *,
    RANK() OVER (ORDER BY efficiency_score DESC)        AS efficiency_rank
FROM scored
ORDER BY efficiency_score DESC;
