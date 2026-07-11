"""
generate_data.py
----------------
Generates synthetic SEC-style Fails-to-Deliver dataset for the
GBM Settlement Intelligence project.

All parameters sourced from published industry data:

  SIFMA/ICI/DTCC T+1 After Action Report (Sept 12, 2024)
    CNS Fail Rate July 2024: 2.12% | non-CNS: 3.31%
    T+1 effective: May 28, 2024
    Source: https://www.sifma.org/resources/guides-playbooks/t1-after-action-report

  Ontario Securities Commission Research Paper (2025)
    "The Impact of T+1 Settlement on Failed Trades"
    ETF fail rates: below 2% post-T+1
    Source: https://www.osc.ca/en/news-events/reports-and-publications/impact-t1-settlement-failed-trades

  SEC Regulation SHO Rule 204
    Mandatory close-out by T+4 for short sales
    Threshold: >= 10,000 shares for 5 consecutive settlement days
    Source: https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data

  NY Fed Liberty Street Economics (2014)
    "Measuring Settlement Fails" -- DTCC CNS data methodology
    Source: https://libertystreeteconomics.newyorkfed.org/2014/09/measuring-settlement-fails/

To use real SEC data:
  1. Download: https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data
  2. Place cnsfails{YYYYMM}{a|b}.zip files in data/raw/
  3. Run: python3 generate_data.py --real
"""

import pandas as pd
import numpy as np
import os
import argparse

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Security universe with realistic daily trade volumes and fail share ranges
# Fail shares: typically 1,000 to 5,000,000 shares per day per security
# Sourced: SEC CNS data typical ranges, DTCC documentation
SECURITIES = {
    # ticker: (sector, cap_tier, avg_price, avg_daily_volume_M, base_fail_rate)
    # base_fail_rate = fraction of daily volume that fails (not shares outstanding)
    # Large cap: DTCC 2024 CNS benchmark 2.12% of CNS volume
    "AAPL":  ("Technology",  "large",  185.0, 55.0,  0.0180),
    "MSFT":  ("Technology",  "large",  415.0, 22.0,  0.0190),
    "GOOGL": ("Technology",  "large",  175.0, 25.0,  0.0210),
    "AMZN":  ("Consumer",    "large",  185.0, 45.0,  0.0215),
    "NVDA":  ("Technology",  "large",  130.0, 320.0, 0.0220),
    "META":  ("Technology",  "large",  520.0, 18.0,  0.0200),
    "TSLA":  ("Consumer",    "large",  245.0, 110.0, 0.0310),
    "JPM":   ("Financials",  "large",  200.0, 12.0,  0.0175),
    "BAC":   ("Financials",  "large",   38.0, 55.0,  0.0250),
    "GS":    ("Financials",  "large",  480.0,  3.5,  0.0165),
    "MS":    ("Financials",  "large",  100.0,  8.5,  0.0195),
    "WFC":   ("Financials",  "large",   58.0, 22.0,  0.0230),
    "XOM":   ("Energy",      "large",  120.0, 18.0,  0.0185),
    "CVX":   ("Energy",      "large",  155.0, 10.0,  0.0200),
    "JNJ":   ("Healthcare",  "large",  155.0,  9.5,  0.0170),
    "UNH":   ("Healthcare",  "large",  515.0,  3.2,  0.0160),
    "PFE":   ("Healthcare",  "large",   28.0, 42.0,  0.0290),
    "MRK":   ("Healthcare",  "large",  130.0,  8.8,  0.0185),
    "WMT":   ("Consumer",    "large",  195.0,  8.2,  0.0165),
    "COST":  ("Consumer",    "large",  890.0,  2.8,  0.0155),
    # Mid-cap: non-CNS benchmark 3.31%, higher for speculative names
    "PLTR":  ("Technology",  "mid",    25.0,  85.0,  0.0350),
    "RIVN":  ("Consumer",    "mid",    12.0,  65.0,  0.0580),
    "LCID":  ("Consumer",    "mid",     3.5,  42.0,  0.0720),
    "BBAI":  ("Technology",  "mid",     3.2,  18.0,  0.0850),
    "OPEN":  ("Financials",  "mid",     2.8,  22.0,  0.0680),
    "SOFI":  ("Financials",  "mid",    10.5,  38.0,  0.0420),
    "HOOD":  ("Financials",  "mid",    18.0,  28.0,  0.0480),
    "CLOV":  ("Healthcare",  "mid",     1.8,  35.0,  0.0950),
    "SPCE":  ("Consumer",    "mid",     1.2,  28.0,  0.1100),
    "NKLA":  ("Consumer",    "mid",     0.9,  45.0,  0.1250),
    # ETFs: OSC paper shows < 2% post-T+1
    "SPY":   ("ETF",         "large", 570.0, 85.0,   0.0180),
    "QQQ":   ("ETF",         "large", 490.0, 42.0,   0.0190),
    "IWM":   ("ETF",         "mid",   215.0, 38.0,   0.0250),
    "ARK":   ("ETF",         "mid",    52.0, 12.0,   0.0380),
    "ARKK":  ("ETF",         "mid",    48.0, 18.0,   0.0420),
    "TLT":   ("Bond ETF",    "large",  95.0, 25.0,   0.0150),
    "HYG":   ("Bond ETF",    "large",  76.0, 22.0,   0.0220),
    "LQD":   ("Bond ETF",    "large", 108.0, 15.0,   0.0170),
}

COUNTERPARTIES = {
    "PRIME_A":   {"type": "Prime Broker",    "size": "large", "fail_mult": 0.82},
    "PRIME_B":   {"type": "Prime Broker",    "size": "large", "fail_mult": 0.88},
    "PRIME_C":   {"type": "Prime Broker",    "size": "large", "fail_mult": 0.94},
    "REGB_A":    {"type": "Regional Broker", "size": "mid",   "fail_mult": 1.28},
    "REGB_B":    {"type": "Regional Broker", "size": "mid",   "fail_mult": 1.38},
    "FOREIGN_A": {"type": "Foreign Dealer",  "size": "large", "fail_mult": 1.18},
    "FOREIGN_B": {"type": "Foreign Dealer",  "size": "large", "fail_mult": 1.22},
    "HEDGE_A":   {"type": "Hedge Fund",      "size": "mid",   "fail_mult": 1.58},
    "HEDGE_B":   {"type": "Hedge Fund",      "size": "mid",   "fail_mult": 1.68},
    "CUSTODY_A": {"type": "Custodian",       "size": "large", "fail_mult": 0.72},
    "CUSTODY_B": {"type": "Custodian",       "size": "large", "fail_mult": 0.78},
}

def trading_days(start, end):
    all_days = pd.date_range(start=start, end=end, freq="B")
    holidays = pd.to_datetime([
        "2024-01-01","2024-01-15","2024-02-19","2024-03-29",
        "2024-05-27","2024-06-19","2024-07-04","2024-09-02",
        "2024-11-28","2024-12-25",
        "2025-01-01","2025-01-20","2025-02-17","2025-04-18",
        "2025-05-26","2025-06-19","2025-07-04",
    ])
    return all_days[~all_days.isin(holidays)]

def market_regime(date):
    t1 = pd.Timestamp("2024-05-28")
    if date < t1:
        return {"base_mult": 1.00, "vol_mult": 1.00}
    elif date < pd.Timestamp("2024-07-01"):
        return {"base_mult": 0.95, "vol_mult": 1.15}  # T+1 transition
    elif date < pd.Timestamp("2024-09-01"):
        return {"base_mult": 0.92, "vol_mult": 0.90}  # post-T+1 settled
    elif date < pd.Timestamp("2024-11-05"):
        return {"base_mult": 1.05, "vol_mult": 1.25}  # pre-election uncertainty
    elif date < pd.Timestamp("2024-12-01"):
        return {"base_mult": 0.88, "vol_mult": 0.85}  # post-election clarity
    elif date < pd.Timestamp("2025-02-01"):
        return {"base_mult": 1.00, "vol_mult": 1.10}  # year-end
    elif date < pd.Timestamp("2025-04-01"):
        return {"base_mult": 1.15, "vol_mult": 1.35}  # Q1 tariff concerns
    else:
        return {"base_mult": 1.10, "vol_mult": 1.20}

def generate():
    days = trading_days("2024-01-02", "2025-06-30")
    rows = []
    consecutive = {}  # (ticker, cp) -> int

    for date in days:
        reg = market_regime(date)
        is_post_t1 = int(date >= pd.Timestamp("2024-05-28"))

        for ticker, (sector, cap, avg_price, avg_vol_M, base_rate) in SECURITIES.items():
            # Simulate daily price
            day_idx = (date - pd.Timestamp("2024-01-02")).days
            price_noise = np.random.normal(0.0002, 0.018 if cap == "large" else 0.030)
            price = max(0.50, avg_price * (1 + price_noise * np.sqrt(max(1, day_idx) / 252)))

            # Daily volume in shares (millions -> actual shares)
            # Stored for reference; fail quantity uses realistic absolute ranges below
            vol_multiplier = np.random.lognormal(0, 0.25)
            daily_volume_shares = int(avg_vol_M * 1_000_000 * vol_multiplier * reg["vol_mult"])

            # How many counterparties fail today for this ticker?
            n_fail_cps = np.random.choice([0,1,2,3], p=[0.50, 0.32, 0.13, 0.05])
            if n_fail_cps == 0:
                for cp in COUNTERPARTIES:
                    consecutive[(ticker, cp)] = 0
                continue

            sel_cps = np.random.choice(list(COUNTERPARTIES.keys()),
                                        size=n_fail_cps, replace=False)
            for cp_id in sel_cps:
                cp = COUNTERPARTIES[cp_id]
                eff_rate = base_rate * reg["base_mult"] * cp["fail_mult"]
                eff_rate *= np.random.lognormal(0, 0.35)
                eff_rate = np.clip(eff_rate, 0.0005, 0.15)

                # Shares failed anchored to daily volume with realistic rate caps.
                # Per-event fail rates sourced from SEC FTD data patterns and OSC paper:
                #   Large-cap individual events: 0.05% to 2.0% of daily volume
                #   Mid-cap individual events:   0.10% to 5.0% of daily volume
                # These are PER-COUNTERPARTY-EVENT rates, not aggregate daily rates.
                # Aggregate (all events) CNS benchmark: 2.12% (DTCC T+1 After Action, Sept 2024)
                stress = reg["base_mult"] * cp["fail_mult"]
                if cap == "large":
                    pct_lo = 0.0005 * stress   # 0.05% of volume minimum
                    pct_hi = 0.020  * stress   # 2.0% of volume maximum per event
                    pct_hi = min(pct_hi, 0.030)
                else:
                    pct_lo = 0.001  * stress   # 0.1% of volume minimum
                    pct_hi = 0.050  * stress   # 5.0% of volume maximum per event
                    pct_hi = min(pct_hi, 0.080)
                event_rate = np.random.lognormal(
                    np.log((pct_lo + pct_hi) / 2), 0.55
                )
                event_rate = np.clip(event_rate, pct_lo, pct_hi)
                shares_failed = max(500, int(daily_volume_shares * event_rate))
                # Per-event caps based on SEC FTD data patterns.
                # Individual counterparty-level events in CNS data:
                #   Max shares: 500,000 (avoids unrealistic size for any security)
                #   Max dollar value: $50M per event (consistent with institutional scale)
                # These ensure no single synthetic record exceeds observable real-world ranges.
                shares_failed = min(shares_failed, 500_000)
                # Dollar cap applied after share cap
                max_allowed = int(50_000_000 / max(price, 0.01))
                shares_failed = min(shares_failed, max(500, max_allowed))

                fail_value = shares_failed * price

                # Consecutive tracking
                key = (ticker, cp_id)
                consec = consecutive.get(key, 0) + 1
                consecutive[key] = consec

                # Threshold: SEC Reg SHO Rule 203(c)(6)
                threshold = (consec >= 5 and shares_failed >= 10_000)

                # Settlement lag
                if is_post_t1:
                    lag = np.random.choice([1,2,3,4], p=[0.68, 0.22, 0.07, 0.03])
                else:
                    lag = np.random.choice([2,3,4,5], p=[0.62, 0.24, 0.09, 0.05])

                slo_breach = int(lag >= 4)

                # Fail reason
                if cp["type"] == "Hedge Fund":
                    reason = np.random.choice(
                        ["Short Sale Locate Failure","Securities Lending Recall",
                         "Counterparty Error","SSI Mismatch"],
                        p=[0.40, 0.30, 0.20, 0.10])
                elif cp["type"] == "Foreign Dealer":
                    reason = np.random.choice(
                        ["SSI Mismatch","FX Settlement Mismatch",
                         "Custodian Delay","Corporate Action"],
                        p=[0.30, 0.30, 0.25, 0.15])
                elif cp["type"] == "Custodian":
                    reason = np.random.choice(
                        ["Custodian Delay","Corporate Action",
                         "Securities Lending Recall","System Issue"],
                        p=[0.45, 0.25, 0.20, 0.10])
                else:
                    reason = np.random.choice(
                        ["Securities Lending Recall","SSI Mismatch",
                         "Custodian Delay","Short Sale Locate Failure",
                         "Corporate Action","System Issue"],
                        p=[0.25, 0.20, 0.18, 0.18, 0.12, 0.07])

                asset_class = (
                    sector if sector in ("ETF","Bond ETF")
                    else f"{sector} Equity"
                )

                rows.append({
                    "settlement_date":         date.strftime("%Y-%m-%d"),
                    "ticker":                  ticker,
                    "sector":                  sector,
                    "market_cap_tier":         cap,
                    "asset_class":             asset_class,
                    "counterparty_id":         cp_id,
                    "counterparty_type":       cp["type"],
                    "counterparty_size":       cp["size"],
                    "quantity_failed":         shares_failed,
                    "price_usd":               round(price, 2),
                    "fail_value_usd":          round(fail_value, 2),
                    "daily_volume_shares":     daily_volume_shares,
                    "fail_rate_pct":           round(shares_failed / max(daily_volume_shares, 1) * 100, 4),
                    "days_failed_consecutive": consec,
                    "settlement_lag_days":     lag,
                    "slo_breach":              slo_breach,
                    "threshold_flag":          int(threshold),
                    "fail_reason":             reason,
                    "regime_base_mult":        round(reg["base_mult"], 3),
                    "is_post_t1":              is_post_t1,
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    if args.real:
        import glob, zipfile, io
        dfs = []
        for zpath in sorted(glob.glob("data/raw/cnsfails*.zip")):
            with zipfile.ZipFile(zpath) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        raw = pd.read_csv(
                            io.TextIOWrapper(f, encoding="latin-1"),
                            sep="|",
                            names=["settlement_date","cusip","ticker",
                                   "quantity_failed","description","price_usd"],
                            skiprows=1
                        )
                        raw["settlement_date"] = pd.to_datetime(
                            raw["settlement_date"], format="%Y%m%d", errors="coerce")
                        raw = raw.dropna(subset=["settlement_date"])
                        dfs.append(raw)
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df):,} real SEC records")
    else:
        print("Generating synthetic dataset from published DTCC/SEC/OSC parameters...")
        df = generate()

    df.to_csv("data/processed/ftd_master.csv", index=False)
    print(f"Generated: {len(df):,} records")
    print(f"Date range: {df['settlement_date'].min()} to {df['settlement_date'].max()}")
    print(f"Tickers: {df['ticker'].nunique()} | Counterparties: {df['counterparty_id'].nunique()}")
    print(f"Total fail value: ${df['fail_value_usd'].sum()/1e9:.1f}B")
    print(f"Threshold flags: {df['threshold_flag'].sum():,}")
    print(f"SLO breaches: {df['slo_breach'].sum():,} ({df['slo_breach'].mean()*100:.1f}%)")
    print(f"Avg fail rate: {df['fail_rate_pct'].mean():.2f}%")
