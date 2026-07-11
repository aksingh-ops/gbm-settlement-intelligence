-- ============================================================
-- Phase 1: Data Load, Validation, and Quality Controls
-- File: sql/01_load_and_validate.sql
-- Tool: DuckDB
--
-- Purpose: Load FTD master dataset, run data quality checks,
--   and build the clean analytical base table.
--
-- Regulatory context:
--   SEC Regulation SHO Rule 204 -- mandatory close-out framework
--   DTCC CNS settlement data methodology (NY Fed Liberty Street, 2014)
--   Threshold definition: SEC Rule 203(c)(6)
--
-- References:
--   SIFMA/ICI/DTCC T+1 After Action Report, September 2024
--   OSC "Impact of T+1 Settlement on Failed Trades", 2025
-- ============================================================

-- Load raw dataset
CREATE OR REPLACE TABLE ftd_raw AS
SELECT * FROM read_csv_auto('data/processed/ftd_master.csv', header=true);

-- Data quality audit -- run first before any analysis
SELECT
    COUNT(*)                                    AS total_records,
    COUNT(DISTINCT ticker)                      AS unique_tickers,
    COUNT(DISTINCT counterparty_id)             AS unique_counterparties,
    COUNT(DISTINCT settlement_date)             AS unique_dates,
    SUM(CASE WHEN quantity_failed IS NULL
             THEN 1 ELSE 0 END)                AS null_quantity,
    SUM(CASE WHEN price_usd <= 0
             THEN 1 ELSE 0 END)                AS zero_or_neg_price,
    SUM(CASE WHEN settlement_lag_days < 1
             OR settlement_lag_days > 10
             THEN 1 ELSE 0 END)                AS invalid_lag,
    MIN(settlement_date)                        AS earliest_date,
    MAX(settlement_date)                        AS latest_date
FROM ftd_raw;

-- Build clean analytical table with all derived fields
CREATE OR REPLACE TABLE ftd_clean AS
SELECT
    CAST(settlement_date AS DATE)               AS settlement_date,
    ticker,
    sector,
    market_cap_tier,
    asset_class,
    counterparty_id,
    counterparty_type,
    counterparty_size,
    CAST(quantity_failed AS BIGINT)             AS quantity_failed,
    CAST(price_usd AS DOUBLE)                   AS price_usd,
    CAST(fail_value_usd AS DOUBLE)              AS fail_value_usd,
    CAST(daily_volume_shares AS BIGINT)         AS daily_volume_shares,
    CAST(fail_rate_pct AS DOUBLE)               AS fail_rate_pct,
    CAST(days_failed_consecutive AS INTEGER)    AS days_failed_consecutive,
    CAST(settlement_lag_days AS INTEGER)        AS settlement_lag_days,
    CAST(slo_breach AS INTEGER)                 AS slo_breach,
    CAST(threshold_flag AS INTEGER)             AS threshold_flag,
    fail_reason,
    CAST(regime_base_mult AS DOUBLE)            AS regime_base_mult,
    CAST(is_post_t1 AS INTEGER)                 AS is_post_t1,
    -- SLO status per SEC Rule 204 close-out framework
    CASE
        WHEN settlement_lag_days <= 1 THEN 'Green'
        WHEN settlement_lag_days <= 3 THEN 'Yellow'
        ELSE 'Red'                              -- T+4 = mandatory close-out
    END                                         AS slo_status,
    -- Fail severity tiers for operations prioritization
    CASE
        WHEN fail_value_usd >= 10_000_000 THEN 'Critical'
        WHEN fail_value_usd >= 1_000_000  THEN 'High'
        WHEN fail_value_usd >= 100_000    THEN 'Medium'
        ELSE 'Low'
    END                                         AS fail_severity,
    DATE_TRUNC('week',  CAST(settlement_date AS DATE)) AS week_start,
    DATE_TRUNC('month', CAST(settlement_date AS DATE)) AS month_start,
    EXTRACT(YEAR    FROM CAST(settlement_date AS DATE)) AS year,
    EXTRACT(QUARTER FROM CAST(settlement_date AS DATE)) AS quarter
FROM ftd_raw
WHERE quantity_failed > 0
  AND price_usd > 0
  AND settlement_lag_days BETWEEN 1 AND 10;

-- Referential integrity check: every record has a valid ticker and counterparty
SELECT
    'Records with missing ticker'       AS check_name,
    COUNT(*) AS failures
FROM ftd_clean WHERE ticker IS NULL
UNION ALL
SELECT
    'Records with missing counterparty',
    COUNT(*) FROM ftd_clean WHERE counterparty_id IS NULL
UNION ALL
SELECT
    'Records with zero fail value',
    COUNT(*) FROM ftd_clean WHERE fail_value_usd <= 0;

-- SLO compliance summary
SELECT
    slo_status,
    COUNT(*)                                            AS records,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    ROUND(SUM(fail_value_usd) / 1e9, 2)                AS fail_value_billions
FROM ftd_clean
GROUP BY slo_status
ORDER BY slo_status;
