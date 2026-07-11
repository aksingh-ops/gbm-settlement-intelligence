"""
run_pipeline.py
---------------
GBM Settlement Intelligence -- Complete Analysis Pipeline.

Runs all phases in sequence:
  1. Generate synthetic FTD dataset (or load real SEC data with --real flag)
  2. Load and validate data via DuckDB SQL
  3. Portfolio health monitoring and counterparty risk ranking
  4. Root cause analysis and SLO attribution
  5. Time series forecasting (SARIMAX) on daily fail volumes
  6. XGBoost classification -- predict SLO breach risk
  7. SHAP feature importance and explainability
  8. Generate all charts and export CSVs

Usage:
  python run_pipeline.py           # synthetic data from published DTCC/SEC parameters
  python run_pipeline.py --real    # real SEC CNS data from data/raw/cnsfails*.zip

References:
  SIFMA/ICI/DTCC T+1 After Action Report, September 12, 2024
  OSC "Impact of T+1 Settlement on Failed Trades", 2025
  SEC Regulation SHO Rules 203/204
  NY Fed Liberty Street Economics, "Measuring Settlement Fails", 2014
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 65)
print("GBM Settlement Intelligence -- Analytics Pipeline")
print("Goldman Sachs GBM Operations Analytics Context")
print("=" * 65)

# -------------------------------------------------------
# Phase 1: Generate / load data
# -------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true",
                    help="Load real SEC CNS data from data/raw/")
args = parser.parse_args()

if not os.path.exists("data/processed/ftd_master.csv") or args.real:
    print("\n[Phase 1] Generating dataset...")
    import subprocess
    cmd = [sys.executable, "generate_data.py"]
    if args.real:
        cmd.append("--real")
    subprocess.run(cmd, check=True)
else:
    print("\n[Phase 1] Using existing dataset (run with --real for SEC data)")

# -------------------------------------------------------
# Phase 2: DuckDB SQL -- load, validate, portfolio health
# -------------------------------------------------------
print("\n[Phase 2] Running SQL analysis phases...")
con = duckdb.connect("data/processed/settlement.duckdb")

# Load and validate
con.execute(open("sql/01_load_and_validate.sql").read())
record_count = con.execute("SELECT COUNT(*) FROM ftd_clean").fetchone()[0]
print(f"  Records loaded and validated: {record_count:,}")

# Portfolio health
con.execute(open("sql/02_portfolio_health.sql").read())
print("  Portfolio health tables built")

# Root cause analysis
con.execute(open("sql/03_root_cause_analysis.sql").read())
print("  Root cause analysis complete")

# Load key results into pandas for ML and charting
df = con.execute("SELECT * FROM ftd_clean").df()
df_daily = con.execute("SELECT * FROM daily_portfolio_health ORDER BY settlement_date").df()
df_cp = con.execute("SELECT * FROM counterparty_risk_ranking ORDER BY composite_risk_score DESC").df()
df_sector = con.execute("SELECT * FROM sector_fail_analysis").df()
df_reason = con.execute("SELECT * FROM fail_reason_analysis").df()
df_slo = con.execute("SELECT * FROM slo_breach_attribution ORDER BY breach_rate_pct DESC LIMIT 20").df()

# Export CSVs
df_daily.to_csv("reports/daily_portfolio_health.csv", index=False)
df_cp.to_csv("reports/counterparty_risk_ranking.csv", index=False)
df_sector.to_csv("reports/sector_fail_analysis.csv", index=False)
df_reason.to_csv("reports/fail_reason_analysis.csv", index=False)
print("  CSVs exported to reports/")

print(f"\n  Key metrics:")
print(f"    Total fail value:     ${df['fail_value_usd'].sum()/1e9:.1f}B")
print(f"    SLO breach rate:      {df['slo_breach'].mean()*100:.1f}%")
print(f"    Avg fail rate:        {df['fail_rate_pct'].mean():.2f}%")
print(f"    Post-T+1 records:     {df['is_post_t1'].mean()*100:.0f}%")

# -------------------------------------------------------
# Phase 3: Time Series Forecasting (SARIMAX)
# Daily aggregate fail rate -- forecast 30 trading days ahead
# -------------------------------------------------------
print("\n[Phase 3] Time series forecasting (SARIMAX)...")

ts = df_daily.set_index("settlement_date")["avg_fail_rate_pct"].copy()
ts.index = pd.to_datetime(ts.index)
ts = ts.sort_index()

# Fit SARIMAX -- weekly seasonality (5 trading days)
try:
    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False, maxiter=200)

    # Forecast 30 trading days ahead
    n_forecast = 30
    forecast_obj = result.get_forecast(steps=n_forecast)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int(alpha=0.10)

    # Generate future dates (trading days only)
    last_date = ts.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_forecast
    )
    forecast_series = pd.Series(forecast_mean.values, index=future_dates)
    forecast_lower = pd.Series(forecast_ci.iloc[:, 0].values, index=future_dates)
    forecast_upper = pd.Series(forecast_ci.iloc[:, 1].values, index=future_dates)

    # In-sample MAPE
    fitted = result.fittedvalues
    mape = np.mean(np.abs((ts - fitted) / ts.replace(0, np.nan))) * 100
    print(f"  SARIMAX fitted | MAPE: {mape:.2f}%")
    forecast_available = True
except Exception as e:
    print(f"  SARIMAX error: {e} -- skipping forecast chart")
    forecast_available = False

# -------------------------------------------------------
# Phase 4: XGBoost -- SLO Breach Prediction
# -------------------------------------------------------
print("\n[Phase 4] XGBoost SLO breach prediction model...")

# Feature engineering
le_reason = LabelEncoder()
le_cp = LabelEncoder()
le_sector = LabelEncoder()
le_cp_type = LabelEncoder()

df_ml = df.copy()
df_ml["fail_reason_enc"]   = le_reason.fit_transform(df_ml["fail_reason"])
df_ml["counterparty_enc"]  = le_cp.fit_transform(df_ml["counterparty_id"])
df_ml["sector_enc"]        = le_sector.fit_transform(df_ml["sector"])
df_ml["cp_type_enc"]       = le_cp_type.fit_transform(df_ml["counterparty_type"])
df_ml["is_large_cap"]      = (df_ml["market_cap_tier"] == "large").astype(int)
df_ml["is_post_t1"]        = df_ml["is_post_t1"].astype(int)
df_ml["log_fail_value"]    = np.log1p(df_ml["fail_value_usd"])
df_ml["log_quantity"]      = np.log1p(df_ml["quantity_failed"])

features = [
    "fail_rate_pct", "log_fail_value", "log_quantity",
    "days_failed_consecutive", "price_usd",
    "fail_reason_enc", "counterparty_enc", "cp_type_enc",
    "sector_enc", "is_large_cap", "is_post_t1", "regime_base_mult",
]

X = df_ml[features]
y = df_ml["slo_breach"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Class imbalance: ~6% breach rate -- use scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos

model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.80,
    colsample_bytree=0.80,
    scale_pos_weight=scale,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=25,
    random_state=42,
    verbosity=0,
)
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model_xgb.predict(X_test)
y_prob = model_xgb.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, output_dict=True)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print(f"  XGBoost AUC-ROC: {auc:.4f}")
print(f"  Precision (breach): {report['1']['precision']:.3f}")
print(f"  Recall (breach):    {report['1']['recall']:.3f}")
print(f"  F1 (breach):        {report['1']['f1-score']:.3f}")

# SHAP values
print("\n[Phase 4b] Computing SHAP feature importance...")
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
shap_importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
print(f"  Top 3 features: {shap_importance['feature'].head(3).tolist()}")

# -------------------------------------------------------
# Phase 5: Charts
# -------------------------------------------------------
print("\n[Phase 5] Generating charts...")

# GS-style color palette
GS_NAVY  = "#0A1628"
GS_GOLD  = "#C9A84C"
GS_RED   = "#C0392B"
GS_GREEN = "#117A65"
GS_GRAY  = "#BDC3C7"
GS_LIGHT = "#F4F6F7"

# -------------------------------------------------------
# Chart 1: Daily Portfolio Health Dashboard
# -------------------------------------------------------
fig1, axes = plt.subplots(3, 1, figsize=(18, 12), facecolor=GS_LIGHT)
fig1.suptitle(
    "GBM Settlement Intelligence -- Daily Portfolio Health Monitor\n"
    "Fail Rate Trend, SLO Compliance, and Anomaly Detection",
    fontsize=14, fontweight="bold", color=GS_NAVY, y=0.98
)

dates = pd.to_datetime(df_daily["settlement_date"])
t1_date = pd.Timestamp("2024-05-28")

# Panel 1: Fail rate with rolling average and Z-score alerts
ax1 = axes[0]
ax1.fill_between(dates, df_daily["avg_fail_rate_pct"], alpha=0.25,
                  color=GS_NAVY, label="Daily avg fail rate")
ax1.plot(dates, df_daily["avg_fail_rate_pct"], color=GS_NAVY, lw=0.8, alpha=0.6)
ax1.plot(dates, df_daily["rolling_7d_avg_fail_rate"], color=GS_GOLD,
          lw=2.0, label="7-day rolling average")
# Highlight anomalies (Z > 2)
anomaly_mask = df_daily["fail_rate_zscore"].abs() > 2
ax1.scatter(dates[anomaly_mask],
             df_daily["avg_fail_rate_pct"][anomaly_mask],
             color=GS_RED, s=60, zorder=5, label="Anomaly (Z > 2)")
ax1.axvline(t1_date, color=GS_RED, ls="--", lw=1.5, alpha=0.7)
ax1.text(t1_date + pd.Timedelta(days=3), ax1.get_ylim()[1] * 0.95,
          "T+1 Effective\nMay 28, 2024",
          fontsize=8, color=GS_RED, va="top")
ax1.set_ylabel("Avg Fail Rate (%)", fontsize=10, color=GS_NAVY)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_facecolor("white")

# Panel 2: SLO compliance rate
ax2 = axes[1]
ax2.plot(dates, df_daily["rolling_7d_slo_pct"], color=GS_GREEN,
          lw=2.0, label="7-day rolling SLO compliance %")
ax2.axhline(95.0, color=GS_GOLD, ls="--", lw=1.5, label="Target: 95%")
ax2.axhline(90.0, color=GS_RED, ls=":", lw=1.5, label="Alert threshold: 90%")
ax2.fill_between(dates, df_daily["rolling_7d_slo_pct"], 95.0,
                  where=df_daily["rolling_7d_slo_pct"] < 95.0,
                  alpha=0.15, color=GS_RED, label="Below target")
ax2.set_ylabel("SLO Compliance (%)", fontsize=10, color=GS_NAVY)
ax2.set_ylim(80, 102)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_facecolor("white")

# Panel 3: Critical and high severity fail counts
ax3 = axes[2]
ax3.bar(dates, df_daily["critical_fails"], color=GS_RED, alpha=0.85,
         label="Critical (>$10M)", width=0.8)
ax3.bar(dates, df_daily["high_fails"], bottom=df_daily["critical_fails"],
         color=GS_GOLD, alpha=0.85, label="High ($1M-$10M)", width=0.8)
ax3.set_ylabel("High/Critical Fail Events", fontsize=10, color=GS_NAVY)
ax3.set_xlabel("Settlement Date", fontsize=10, color=GS_NAVY)
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_facecolor("white")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("reports/01_daily_portfolio_health.png", dpi=150, bbox_inches="tight",
            facecolor=GS_LIGHT)
plt.close()
print("  Saved: reports/01_daily_portfolio_health.png")

# -------------------------------------------------------
# Chart 2: Counterparty Risk Matrix
# -------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8), facecolor=GS_LIGHT)
fig2.suptitle(
    "Counterparty Risk Ranking -- Settlement Performance Dashboard\n"
    "Composite Risk Score: 40% Fail Value | 40% SLO Breach Rate | 20% Fail Rate",
    fontsize=13, fontweight="bold", color=GS_NAVY
)

# Left: Horizontal bar chart ranked by composite risk score
df_cp_sorted = df_cp.sort_values("composite_risk_score", ascending=True)
colors_cp = [GS_RED if q == 1 else GS_GOLD if q == 2 else GS_GREEN
              for q in df_cp_sorted["risk_quartile"]]
axes2[0].barh(df_cp_sorted["counterparty_id"],
               df_cp_sorted["composite_risk_score"],
               color=colors_cp, alpha=0.88, height=0.65)
axes2[0].set_xlabel("Composite Risk Score (0 = lowest risk)", fontsize=10)
axes2[0].set_title("Counterparty Risk Score Ranking", fontsize=11,
                    fontweight="bold", color=GS_NAVY)
axes2[0].grid(True, alpha=0.25, axis="x")
axes2[0].set_facecolor("white")
p1 = mpatches.Patch(color=GS_RED, alpha=0.88, label="Q1 -- Highest risk")
p2 = mpatches.Patch(color=GS_GOLD, alpha=0.88, label="Q2 -- Elevated risk")
p3 = mpatches.Patch(color=GS_GREEN, alpha=0.88, label="Q3/Q4 -- Managed risk")
axes2[0].legend(handles=[p1, p2, p3], fontsize=9)

# Right: Scatter -- fail value vs SLO breach rate, sized by avg lag
sc = axes2[1].scatter(
    df_cp["fail_value_millions"],
    df_cp["slo_breach_rate_pct"],
    s=df_cp["avg_settlement_lag"] * 80,
    c=df_cp["composite_risk_score"],
    cmap="RdYlGn_r",
    alpha=0.85,
    edgecolors="white", lw=1.5
)
for _, row in df_cp.iterrows():
    axes2[1].annotate(
        row["counterparty_id"],
        xy=(row["fail_value_millions"], row["slo_breach_rate_pct"]),
        xytext=(5, 5), textcoords="offset points",
        fontsize=8, color=GS_NAVY, fontweight="bold"
    )
axes2[1].set_xlabel("Total Fail Value ($M)", fontsize=10)
axes2[1].set_ylabel("SLO Breach Rate (%)", fontsize=10)
axes2[1].set_title("Risk Matrix: Fail Value vs SLO Breach Rate\n(Bubble = avg settlement lag)",
                    fontsize=11, fontweight="bold", color=GS_NAVY)
plt.colorbar(sc, ax=axes2[1], label="Composite Risk Score")
axes2[1].grid(True, alpha=0.25)
axes2[1].set_facecolor("white")

plt.tight_layout()
plt.savefig("reports/02_counterparty_risk_matrix.png", dpi=150,
            bbox_inches="tight", facecolor=GS_LIGHT)
plt.close()
print("  Saved: reports/02_counterparty_risk_matrix.png")

# -------------------------------------------------------
# Chart 3: SARIMAX Forecast
# -------------------------------------------------------
if forecast_available:
    fig3, ax3f = plt.subplots(figsize=(16, 7), facecolor=GS_LIGHT)
    # Historical: last 90 days
    plot_ts = ts.iloc[-90:]
    ax3f.plot(plot_ts.index, plot_ts.values, color=GS_NAVY, lw=1.5,
               label="Historical daily fail rate (%)")
    ax3f.plot(plot_ts.index, fitted.reindex(plot_ts.index),
               color=GS_GOLD, lw=1.5, ls="--", label=f"SARIMAX fit (MAPE {mape:.2f}%)")
    ax3f.plot(future_dates, forecast_series.values, color=GS_RED, lw=2.0,
               label="30-day forecast")
    ax3f.fill_between(future_dates,
                       forecast_lower.values, forecast_upper.values,
                       alpha=0.20, color=GS_RED, label="90% confidence interval")
    ax3f.axvline(ts.index[-1], color=GS_GRAY, ls=":", lw=2)
    ax3f.text(ts.index[-1] + pd.Timedelta(days=1),
               ax3f.get_ylim()[1] * 0.95,
               "Forecast ->", fontsize=9, color=GS_RED, va="top")
    ax3f.set_xlabel("Settlement Date", fontsize=11)
    ax3f.set_ylabel("Average Daily Fail Rate (%)", fontsize=11)
    ax3f.set_title(
        "GBM Settlement Fail Rate -- SARIMAX(1,1,1)(1,0,1)[5] Forecast\n"
        "30 Trading Day Outlook | DTCC CNS Benchmark: 2.12% (July 2024)",
        fontsize=12, fontweight="bold", color=GS_NAVY
    )
    ax3f.legend(fontsize=9)
    ax3f.grid(True, alpha=0.25)
    ax3f.set_facecolor("white")
    plt.tight_layout()
    plt.savefig("reports/03_sarimax_forecast.png", dpi=150, bbox_inches="tight",
                facecolor=GS_LIGHT)
    plt.close()
    print("  Saved: reports/03_sarimax_forecast.png")

# -------------------------------------------------------
# Chart 4: XGBoost Model Performance + SHAP
# -------------------------------------------------------
fig4, axes4 = plt.subplots(1, 3, figsize=(20, 7), facecolor=GS_LIGHT)
fig4.suptitle(
    f"XGBoost SLO Breach Prediction -- Model Performance & Feature Importance\n"
    f"AUC-ROC: {auc:.4f} | Precision: {report['1']['precision']:.3f} | "
    f"Recall: {report['1']['recall']:.3f}",
    fontsize=13, fontweight="bold", color=GS_NAVY
)

# ROC curve
axes4[0].plot(fpr, tpr, color=GS_GOLD, lw=2.5, label=f"ROC (AUC = {auc:.4f})")
axes4[0].plot([0, 1], [0, 1], color=GS_GRAY, ls="--", lw=1.5, label="Random baseline")
axes4[0].fill_between(fpr, tpr, alpha=0.15, color=GS_GOLD)
axes4[0].set_xlabel("False Positive Rate", fontsize=10)
axes4[0].set_ylabel("True Positive Rate", fontsize=10)
axes4[0].set_title("ROC Curve", fontsize=11, fontweight="bold", color=GS_NAVY)
axes4[0].legend(fontsize=9)
axes4[0].grid(True, alpha=0.3)
axes4[0].set_facecolor("white")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes4[1].imshow(cm, interpolation="nearest", cmap="Blues")
axes4[1].set_title("Confusion Matrix", fontsize=11, fontweight="bold", color=GS_NAVY)
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        axes4[1].text(j, i, f"{cm[i, j]:,}",
                       ha="center", va="center", fontsize=12, fontweight="bold",
                       color="white" if cm[i, j] > thresh else GS_NAVY)
axes4[1].set_xticks([0, 1])
axes4[1].set_yticks([0, 1])
axes4[1].set_xticklabels(["No Breach", "Breach"], fontsize=10)
axes4[1].set_yticklabels(["No Breach", "Breach"], fontsize=10)
axes4[1].set_xlabel("Predicted", fontsize=10)
axes4[1].set_ylabel("Actual", fontsize=10)
axes4[1].set_facecolor("white")

# SHAP feature importance
shap_top = shap_importance.head(10).sort_values("mean_abs_shap")
feature_labels = {
    "fail_rate_pct":           "Fail Rate (%)",
    "log_fail_value":          "Log(Fail Value)",
    "log_quantity":            "Log(Shares Failed)",
    "days_failed_consecutive": "Consecutive Fail Days",
    "price_usd":               "Security Price ($)",
    "fail_reason_enc":         "Fail Reason",
    "counterparty_enc":        "Counterparty ID",
    "cp_type_enc":             "Counterparty Type",
    "sector_enc":              "Sector",
    "is_large_cap":            "Large Cap Flag",
    "is_post_t1":              "Post-T+1 Regime",
    "regime_base_mult":        "Market Regime",
}
labels = [feature_labels.get(f, f) for f in shap_top["feature"]]
bar_colors = [GS_GOLD if i >= len(shap_top) - 3 else GS_NAVY
               for i in range(len(shap_top))]
axes4[2].barh(labels, shap_top["mean_abs_shap"], color=bar_colors, alpha=0.88)
axes4[2].set_xlabel("Mean |SHAP Value|", fontsize=10)
axes4[2].set_title("Feature Importance (SHAP)\nTop 10 Predictors of SLO Breach",
                    fontsize=11, fontweight="bold", color=GS_NAVY)
axes4[2].grid(True, alpha=0.25, axis="x")
axes4[2].set_facecolor("white")

plt.tight_layout()
plt.savefig("reports/04_xgboost_model_performance.png", dpi=150,
            bbox_inches="tight", facecolor=GS_LIGHT)
plt.close()
print("  Saved: reports/04_xgboost_model_performance.png")

# -------------------------------------------------------
# Chart 5: Executive One-Pager
# -------------------------------------------------------
print("\n[Phase 5b] Building executive one-pager...")

fig5 = plt.figure(figsize=(22, 14), facecolor=GS_LIGHT)
fig5.patch.set_facecolor(GS_LIGHT)
gs5 = gridspec.GridSpec(3, 4, figure=fig5, hspace=0.55, wspace=0.40,
                          left=0.04, right=0.97, top=0.88, bottom=0.06)

# Header
fig5.text(0.5, 0.955,
           "GBM Settlement Intelligence -- Executive Dashboard",
           ha="center", fontsize=17, fontweight="bold", color=GS_NAVY)
fig5.text(0.5, 0.925,
           f"GBM Operations Analytics | Jan 2024 - Jun 2025 | "
           f"{record_count:,} settlement records | {df['ticker'].nunique()} securities | "
           f"{df['counterparty_id'].nunique()} counterparties",
           ha="center", fontsize=10, color="#555")

# Gold banner
banner = fig5.add_axes([0.20, 0.895, 0.60, 0.030])
banner.set_facecolor(GS_GOLD)
banner.text(0.5, 0.5,
             f"Total Fail Value: ${df['fail_value_usd'].sum()/1e9:.1f}B  |  "
             f"SLO Breach Rate: {df['slo_breach'].mean()*100:.1f}%  |  "
             f"Post-T+1 Avg Fail Rate: "
             f"{df[df['is_post_t1']==1]['fail_rate_pct'].mean():.2f}%  |  "
             f"XGBoost AUC: {auc:.4f}",
             ha="center", va="center", fontsize=11,
             fontweight="bold", color="white", transform=banner.transAxes)
banner.axis("off")

# KPI tiles
total_fail_B = df['fail_value_usd'].sum() / 1e9
slo_breach_pct = df['slo_breach'].mean() * 100
post_t1_rate = df[df['is_post_t1']==1]['fail_rate_pct'].mean()
pre_t1_rate = df[df['is_post_t1']==0]['fail_rate_pct'].mean()
kpis = [
    ("Total Fail Value",       f"${total_fail_B:.1f}B",    GS_RED),
    ("SLO Breach Rate",        f"{slo_breach_pct:.1f}%",   GS_GOLD),
    ("Post-T+1 Avg Fail Rate", f"{post_t1_rate:.2f}%",     GS_NAVY),
    ("AUC (Breach Predictor)", f"{auc:.4f}",               GS_GREEN),
]
for i, (label, value, color) in enumerate(kpis):
    ax_kpi = fig5.add_subplot(gs5[0, i])
    ax_kpi.set_facecolor("white")
    ax_kpi.text(0.5, 0.58, value, ha="center", va="center",
                 fontsize=20, fontweight="bold", color=color,
                 transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.22, label, ha="center", va="center",
                 fontsize=9, color="#555", transform=ax_kpi.transAxes)
    for sp in ax_kpi.spines.values():
        sp.set_edgecolor("#ddd")
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

# Fail rate trend (last 60 days)
ax_ts = fig5.add_subplot(gs5[1, 0:2])
recent = df_daily.tail(60)
rdates = pd.to_datetime(recent["settlement_date"])
ax_ts.plot(rdates, recent["avg_fail_rate_pct"],
            color=GS_NAVY, lw=1.0, alpha=0.5)
ax_ts.plot(rdates, recent["rolling_7d_avg_fail_rate"],
            color=GS_GOLD, lw=2.0, label="7-day rolling avg")
anomaly_r = recent["fail_rate_zscore"].abs() > 2
ax_ts.scatter(rdates[anomaly_r.values],
               recent["avg_fail_rate_pct"].values[anomaly_r.values],
               color=GS_RED, s=40, zorder=5, label="Anomaly")
ax_ts.axhline(2.12, color=GS_GREEN, ls="--", lw=1.5,
               label="DTCC benchmark: 2.12%")
ax_ts.set_title("Fail Rate Trend -- Last 60 Trading Days",
                 fontsize=10, fontweight="bold", color=GS_NAVY)
ax_ts.legend(fontsize=7.5); ax_ts.grid(True, alpha=0.25)
ax_ts.set_facecolor("white")

# Counterparty risk bar (top 6)
ax_cp = fig5.add_subplot(gs5[1, 2:4])
cp_top = df_cp.head(6).sort_values("composite_risk_score")
cp_colors = [GS_RED if q == 1 else GS_GOLD if q == 2 else GS_NAVY
              for q in cp_top["risk_quartile"]]
ax_cp.barh(cp_top["counterparty_id"], cp_top["composite_risk_score"],
            color=cp_colors, alpha=0.88, height=0.6)
for _, row in cp_top.iterrows():
    ax_cp.text(row["composite_risk_score"] + 0.005,
                row["counterparty_id"],
                f"${row['fail_value_millions']:.0f}M",
                va="center", fontsize=8, fontweight="bold")
ax_cp.set_title("Counterparty Risk Ranking (Top 6)",
                 fontsize=10, fontweight="bold", color=GS_NAVY)
ax_cp.set_xlabel("Composite Risk Score", fontsize=9)
ax_cp.grid(True, alpha=0.25, axis="x")
ax_cp.set_facecolor("white")

# SLO breach by reason
ax_reason = fig5.add_subplot(gs5[2, 0:2])
reason_agg = df.groupby("fail_reason")["slo_breach"].agg(
    ["sum", "count"]).reset_index()
reason_agg["breach_pct"] = reason_agg["sum"] / reason_agg["count"] * 100
reason_agg = reason_agg.sort_values("breach_pct", ascending=True)
reason_colors = [GS_RED if v > 8 else GS_GOLD if v > 5 else GS_GREEN
                  for v in reason_agg["breach_pct"]]
ax_reason.barh(reason_agg["fail_reason"], reason_agg["breach_pct"],
                color=reason_colors, alpha=0.88, height=0.6)
ax_reason.axvline(df["slo_breach"].mean()*100, color=GS_NAVY,
                   ls="--", lw=1.5, label="Overall avg")
ax_reason.set_title("SLO Breach Rate by Fail Reason",
                     fontsize=10, fontweight="bold", color=GS_NAVY)
ax_reason.set_xlabel("Breach Rate (%)", fontsize=9)
ax_reason.tick_params(axis="y", labelsize=8)
ax_reason.legend(fontsize=8)
ax_reason.grid(True, alpha=0.25, axis="x")
ax_reason.set_facecolor("white")

# SHAP top 5
ax_shap = fig5.add_subplot(gs5[2, 2:4])
shap_top5 = shap_importance.head(5).sort_values("mean_abs_shap")
shap_labels = [feature_labels.get(f, f) for f in shap_top5["feature"]]
ax_shap.barh(shap_labels, shap_top5["mean_abs_shap"],
              color=GS_GOLD, alpha=0.88, height=0.55)
ax_shap.set_title("Top 5 SLO Breach Predictors (SHAP)",
                   fontsize=10, fontweight="bold", color=GS_NAVY)
ax_shap.set_xlabel("Mean |SHAP Value|", fontsize=9)
ax_shap.grid(True, alpha=0.25, axis="x")
ax_shap.set_facecolor("white")

plt.savefig("reports/05_executive_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=GS_LIGHT)
plt.close()
print("  Saved: reports/05_executive_dashboard.png")

# -------------------------------------------------------
# Final summary
# -------------------------------------------------------
con.close()
print("\n" + "=" * 65)
print("Pipeline complete. All outputs saved to reports/")
print("=" * 65)
for f in sorted(os.listdir("reports/")):
    size = os.path.getsize(f"reports/{f}") / 1024
    print(f"  {f:<45} {size:.1f} KB")

print(f"""
Key Results:
  Records analyzed:       {record_count:,}
  Total fail value:       ${df['fail_value_usd'].sum()/1e9:.1f}B
  SLO breach rate:        {df['slo_breach'].mean()*100:.1f}%
  Post-T+1 avg fail rate: {post_t1_rate:.2f}% (DTCC benchmark: 2.12%)
  XGBoost AUC-ROC:        {auc:.4f}
  SARIMAX MAPE:           {mape:.2f}% (if forecast available)
  Highest risk CP:        {df_cp.iloc[0]['counterparty_id']} (score: {df_cp.iloc[0]['composite_risk_score']:.4f})
""")
