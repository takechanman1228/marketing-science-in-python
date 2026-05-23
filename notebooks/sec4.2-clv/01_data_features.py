# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Marketing Science
#     language: python
#     name: marketing-science
# ---

# %% [markdown]
# # Purchase-Incidence + Amount ML CLV — Step 1: Data & Feature Engineering
#
# This notebook builds the feature table and **customer-month panel** for the
# decomposed ML CLV pipeline. We mirror the BTYD (BG/NBD + Gamma-Gamma)
# structure:
#
# | Component | BTYD | ML (this pipeline) |
# |-----------|------|--------------------|
# | Purchase | BG/NBD → P(alive) × λ | LightGBM classifier → P(buy in any period) |
# | Spend | Gamma-Gamma → E[spend per transaction] | LightGBM regressor → E[spend per period \| buy] |
# | Aggregation | Σ E[purchases_t] × E[spend] / (1+r)^t | Σ P(buy) × E[spend\|buy] / (1+r)^t |
#
# **Dataset**: Dunnhumby "The Complete Journey" (same as Chapter 4.1)
#
# **Temporal split**:
# - Calibration: weeks 1–52 (Year 1) — features observed here
# - Holdout: weeks 53–102 (50 weeks) — divided into 12 four-week periods
#
# **Feature groups** (~30 features per household):
#
# | Group | Features | Source |
# |-------|----------|--------|
# | A. RFM-T | recency, frequency, monetary_avg, monetary_total, tenure | transactions |
# | B. Behavioral | basket size, inter-purchase gaps, product breadth | transactions |
# | C. Discount | discount ratio, coupon usage, total discount % | transactions |
# | D. Product mix | department concentration, grocery share | transactions + product |
# | E. Trends | spend trend, frequency trend, new customer flag | transactions (H1 vs H2) |
# | F. Campaign | campaigns targeted, coupons redeemed, response rate | campaign/coupon tables |
# | G. Demographics | age, income, household size, kids, homeowner | hh_demographic (801 HHs) |

# %%
import warnings
warnings.filterwarnings("ignore")

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from _retail_data import (
    load_dunnhumby_transactions,
    load_dunnhumby_products,
    load_dunnhumby_demographics,
    load_dunnhumby_campaigns,
)
from msbook.paths import chapter_images, chapter_artifacts

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir": "../Retail_open_data_set/Dunnhumby_kaggle/archive/",
    "cal_end_week": 52,       # last week of calibration period
    "cal_mid_week": 26,       # H1/H2 boundary for trend features
    "holdout_end_week": 102,  # last week in dataset
    "random_state": 42,
}

# Customer-month panel parameters (aligned with PyMC demo)
N_HOLDOUT_PERIODS = 12        # 12 four-week periods from the 50-week holdout
WEEKS_PER_PERIOD = 4
DISCOUNT_RATE = 0.01          # monthly, same as PyMC demo
CLV_HORIZON = 120             # months (10 years), same as PyMC demo

ACCENT_BLUE = "#2171b5"
ACCENT_ORANGE = "#e6550d"
ACCENT_GREEN = "#31a354"

pd.set_option("display.float_format", "{:.2f}".format)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

_FIG_DIR = chapter_images(part="4", chapter="sec4.2")
# ML CLV pipeline artifacts (cross-script: 01 → 02 → 03) live under artifacts/.
_TABLES = chapter_artifacts(part="4", chapter="sec4.2-clv") / "tables"
_TABLES.mkdir(parents=True, exist_ok=True)


def savefig(fig, name, **kw):
    kw.setdefault("dpi", 150)
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(_FIG_DIR / name, **kw)

_step = 0
def step(title):
    global _step
    _step += 1
    print(f"\n{'='*60}")
    print(f"  Step {_step}: {title}")
    print(f"{'='*60}\n")


# %% [markdown]
# ## 1. Load Raw Data

# %%
step("Load raw data")

d = CONFIG["data_dir"]
txn = load_dunnhumby_transactions(data_dir=d)
product = load_dunnhumby_products(data_dir=d)
demographics = load_dunnhumby_demographics(data_dir=d)
campaigns = load_dunnhumby_campaigns(data_dir=d)

print(f"Transactions:  {len(txn):>10,} rows  |  {txn['household_key'].nunique():,} households")
print(f"Products:      {len(product):>10,} rows")
print(f"Demographics:  {len(demographics):>10,} rows  |  {demographics['household_key'].nunique()} households")
print(f"Week range:    {txn['WEEK_NO'].min()}–{txn['WEEK_NO'].max()}")

# %% [markdown]
# ## 2. Temporal Split

# %%
step("Temporal split — calibration vs. holdout")

cal_end = CONFIG["cal_end_week"]
cal_txn = txn[txn["WEEK_NO"] <= cal_end].copy()
hold_txn = txn[txn["WEEK_NO"] > cal_end].copy()

# Only keep households that appear in the calibration period
cal_households = cal_txn["household_key"].unique()

print(f"Calibration (weeks 1–{cal_end}):  {len(cal_txn):>10,} rows  |  {len(cal_households):,} households")
print(f"Holdout (weeks {cal_end+1}–{CONFIG['holdout_end_week']}):  {len(hold_txn):>10,} rows  |  {hold_txn['household_key'].nunique():,} households")

# Max DAY in calibration (for filtering campaign data)
cal_max_day = cal_txn["DAY"].max()

# %% [markdown]
# ## 3. Feature Group A — RFM-T (5 features)

# %%
step("Feature Group A: RFM-T")

rfm = (
    cal_txn.groupby("household_key")
    .agg(
        last_week=("WEEK_NO", "max"),
        first_week=("WEEK_NO", "min"),
        frequency=("BASKET_ID", "nunique"),
        monetary_total=("SALES_VALUE", "sum"),
    )
)
rfm["recency"] = cal_end - rfm["last_week"]
rfm["tenure_weeks"] = rfm["last_week"] - rfm["first_week"]
rfm["monetary_avg"] = rfm["monetary_total"] / rfm["frequency"]
rfm = rfm[["recency", "frequency", "monetary_avg", "monetary_total", "tenure_weeks"]]

print(f"RFM-T features: {list(rfm.columns)}")
print(rfm.describe().round(2))

# %% [markdown]
# ## 4. Feature Group B — Behavioral (7 features)

# %%
step("Feature Group B: Behavioral")

# Basket size stats
basket_sizes = cal_txn.groupby(["household_key", "BASKET_ID"]).size().reset_index(name="items")
basket_stats = basket_sizes.groupby("household_key")["items"].agg(["mean", "std"])
basket_stats.columns = ["avg_basket_size", "basket_size_std"]
basket_stats["basket_size_std"] = basket_stats["basket_size_std"].fillna(0)

# Inter-purchase intervals (using DAY column)
purchase_days = (
    cal_txn[["household_key", "DAY"]]
    .drop_duplicates()
    .sort_values(["household_key", "DAY"])
)
purchase_days["prev_day"] = purchase_days.groupby("household_key")["DAY"].shift(1)
purchase_days["gap"] = purchase_days["DAY"] - purchase_days["prev_day"]
inter_stats = (
    purchase_days.dropna(subset=["gap"])
    .groupby("household_key")["gap"]
    .agg(["mean", "std"])
)
inter_stats.columns = ["inter_purchase_mean", "inter_purchase_std"]
inter_stats["inter_purchase_std"] = inter_stats["inter_purchase_std"].fillna(0)

# Breadth features
breadth = (
    cal_txn.groupby("household_key")
    .agg(
        n_unique_products=("PRODUCT_ID", "nunique"),
        n_unique_stores=("STORE_ID", "nunique"),
        n_shopping_days=("DAY", "nunique"),
    )
)

behavioral = breadth.join(basket_stats).join(inter_stats)
print(f"Behavioral features: {list(behavioral.columns)}")
print(behavioral.describe().round(2))

# %% [markdown]
# ## 5. Feature Group C — Discount (3 features)

# %%
step("Feature Group C: Discount")

disc = cal_txn.groupby("household_key").agg(
    n_rows=("SALES_VALUE", "count"),
    n_with_retail_disc=("RETAIL_DISC", lambda x: (x < 0).sum()),
    n_with_coupon=("COUPON_DISC", lambda x: (x < 0).sum()),
    total_sales=("SALES_VALUE", "sum"),
    total_retail_disc=("RETAIL_DISC", lambda x: x[x < 0].sum()),
    total_coupon_disc=("COUPON_DISC", lambda x: x[x < 0].sum()),
    total_coupon_match=("COUPON_MATCH_DISC", lambda x: x[x < 0].sum()),
)

disc["discount_ratio"] = disc["n_with_retail_disc"] / disc["n_rows"]
disc["coupon_usage_rate"] = disc["n_with_coupon"] / disc["n_rows"]
# Total discount as % of sales (discounts are negative, so negate)
disc["total_discount_pct"] = (
    -(disc["total_retail_disc"] + disc["total_coupon_disc"] + disc["total_coupon_match"])
    / disc["total_sales"]
)

discount_features = disc[["discount_ratio", "coupon_usage_rate", "total_discount_pct"]]
print(f"Discount features: {list(discount_features.columns)}")
print(discount_features.describe().round(4))

# %% [markdown]
# ## 6. Feature Group D — Product Mix (4 features)

# %%
step("Feature Group D: Product mix")

# Join transactions with product to get DEPARTMENT
cal_txn_dept = cal_txn.merge(
    product[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="left"
)

# Department-level spend per household
dept_spend = (
    cal_txn_dept.groupby(["household_key", "DEPARTMENT"])["SALES_VALUE"]
    .sum()
    .reset_index()
)
hh_totals = dept_spend.groupby("household_key")["SALES_VALUE"].sum().rename("hh_total")
dept_spend = dept_spend.merge(hh_totals, on="household_key")
dept_spend["share"] = dept_spend["SALES_VALUE"] / dept_spend["hh_total"]

# n_departments
n_depts = dept_spend.groupby("household_key")["DEPARTMENT"].nunique().rename("n_departments")

# top_dept_share
top_share = dept_spend.groupby("household_key")["share"].max().rename("top_dept_share")

# herfindahl_dept (concentration index)
dept_spend["share_sq"] = dept_spend["share"] ** 2
herfindahl = dept_spend.groupby("household_key")["share_sq"].sum().rename("herfindahl_dept")

# pct_grocery
grocery_mask = dept_spend["DEPARTMENT"] == "GROCERY"
pct_grocery = (
    dept_spend.loc[grocery_mask]
    .set_index("household_key")["share"]
    .reindex(cal_households, fill_value=0)
    .rename("pct_grocery")
)

product_features = pd.DataFrame(index=cal_households)
product_features = product_features.join(n_depts).join(top_share).join(herfindahl).join(pct_grocery)

print(f"Product features: {list(product_features.columns)}")
print(product_features.describe().round(4))

# %% [markdown]
# ## 7. Feature Group E — Trends (3 features)

# %%
step("Feature Group E: Trends (H1 vs H2 of calibration)")

mid = CONFIG["cal_mid_week"]
h1 = cal_txn[cal_txn["WEEK_NO"] <= mid]
h2 = cal_txn[cal_txn["WEEK_NO"] > mid]

# Spend trend
h1_spend = h1.groupby("household_key")["SALES_VALUE"].sum().reindex(cal_households, fill_value=0)
h2_spend = h2.groupby("household_key")["SALES_VALUE"].sum().reindex(cal_households, fill_value=0)
spend_trend = (h2_spend - h1_spend) / (h1_spend + 1)

# Frequency trend
h1_freq = h1.groupby("household_key")["BASKET_ID"].nunique().reindex(cal_households, fill_value=0)
h2_freq = h2.groupby("household_key")["BASKET_ID"].nunique().reindex(cal_households, fill_value=0)
freq_trend = (h2_freq - h1_freq) / (h1_freq + 1)

# New customer flag (first purchase in H2)
first_week = cal_txn.groupby("household_key")["WEEK_NO"].min()
is_new = (first_week > mid).astype(int)

trend_features = pd.DataFrame({
    "spend_trend": spend_trend,
    "freq_trend": freq_trend,
    "is_new_customer": is_new,
}, index=cal_households)

print(f"Trend features: {list(trend_features.columns)}")
print(trend_features.describe().round(4))
print(f"New customers (first purchase in weeks {mid+1}–{cal_end}): {is_new.sum()}")

# %% [markdown]
# ## 8. Feature Group F — Campaign (3 features)

# %%
step("Feature Group F: Campaign")

campaign_desc = campaigns["campaign_desc"]
campaign_table = campaigns["campaign_table"]
coupon_redempt = campaigns["coupon_redempt"]

# Filter to campaigns that started during calibration
cal_campaigns = campaign_desc[campaign_desc["START_DAY"] <= cal_max_day]["CAMPAIGN"]
cal_campaign_table = campaign_table[campaign_table["CAMPAIGN"].isin(cal_campaigns)]

# n_campaigns_targeted
n_targeted = (
    cal_campaign_table.groupby("household_key")["CAMPAIGN"]
    .nunique()
    .reindex(cal_households, fill_value=0)
    .rename("n_campaigns_targeted")
)

# n_coupons_redeemed (filtered to calibration period)
cal_redempt = coupon_redempt[coupon_redempt["DAY"] <= cal_max_day]
n_redeemed = (
    cal_redempt.groupby("household_key")["COUPON_UPC"]
    .count()
    .reindex(cal_households, fill_value=0)
    .rename("n_coupons_redeemed")
)

# Response rate
campaign_features = pd.DataFrame({
    "n_campaigns_targeted": n_targeted,
    "n_coupons_redeemed": n_redeemed,
}, index=cal_households)
campaign_features["campaign_response_rate"] = np.where(
    campaign_features["n_campaigns_targeted"] > 0,
    campaign_features["n_coupons_redeemed"] / campaign_features["n_campaigns_targeted"],
    0,
)

print(f"Campaign features: {list(campaign_features.columns)}")
print(campaign_features.describe().round(4))
print(f"Households targeted by >=1 campaign: {(n_targeted > 0).sum()}")

# %% [markdown]
# ## 9. Feature Group G — Demographics (5 features)
#
# Only 801 of ~2,500 households have demographics. LightGBM handles NaN
# natively, so we keep them as missing for the ~1,700 without data.

# %%
step("Feature Group G: Demographics")

AGE_MAP = {
    "19-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6,
}
INCOME_MAP = {
    "Under 15K": 1, "15-24K": 2, "25-34K": 3, "35-49K": 4,
    "50-74K": 5, "75-99K": 6, "100-124K": 7, "125-149K": 8,
    "150-174K": 9, "175-199K": 10, "200-249K": 11, "250K+": 12,
}
SIZE_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}

demo = demographics.set_index("household_key").reindex(cal_households)
demo_features = pd.DataFrame(index=cal_households)
demo_features["age_ordinal"] = demo["AGE_DESC"].map(AGE_MAP)
demo_features["income_ordinal"] = demo["INCOME_DESC"].map(INCOME_MAP)
demo_features["household_size"] = demo["HOUSEHOLD_SIZE_DESC"].map(SIZE_MAP)
demo_features["has_kids"] = (
    demo["KID_CATEGORY_DESC"]
    .map(lambda x: 0 if x == "None/Unknown" else 1 if pd.notna(x) else np.nan)
)
demo_features["is_homeowner"] = (
    demo["HOMEOWNER_DESC"]
    .map(lambda x: 1 if x in ("Homeowner", "Probable Homeowner") else 0 if pd.notna(x) else np.nan)
)

n_with_demo = demo_features["age_ordinal"].notna().sum()
print(f"Demographics features: {list(demo_features.columns)}")
print(f"Households with demographics: {n_with_demo} / {len(cal_households)}")
print(demo_features.describe().round(2))

# %% [markdown]
# ## 10. Combine All Features

# %%
step("Combine all features")

# Build the master feature table
features = (
    rfm
    .join(behavioral)
    .join(discount_features)
    .join(product_features)
    .join(trend_features)
    .join(campaign_features)
    .join(demo_features)
)

# Define feature groups for downstream use
FEATURE_GROUPS = {
    "A_rfm": ["recency", "frequency", "monetary_avg", "monetary_total", "tenure_weeks"],
    "B_behavioral": [
        "n_unique_products", "n_unique_stores", "avg_basket_size",
        "basket_size_std", "inter_purchase_mean", "inter_purchase_std",
        "n_shopping_days",
    ],
    "C_discount": ["discount_ratio", "coupon_usage_rate", "total_discount_pct"],
    "D_product": ["n_departments", "top_dept_share", "herfindahl_dept", "pct_grocery"],
    "E_trends": ["spend_trend", "freq_trend", "is_new_customer"],
    "F_campaign": ["n_campaigns_targeted", "n_coupons_redeemed", "campaign_response_rate"],
    "G_demographics": ["age_ordinal", "income_ordinal", "household_size", "has_kids", "is_homeowner"],
}

all_feature_cols = [c for group in FEATURE_GROUPS.values() for c in group]

print(f"Feature table: {features.shape[0]} households x {len(all_feature_cols)} features")
print(f"\nFeature groups:")
for group_name, cols in FEATURE_GROUPS.items():
    n_na = features[cols].isna().any(axis=1).sum()
    print(f"  {group_name:20s}: {len(cols):2d} features  (NaN rows: {n_na})")

print(f"\nMissing values per feature:")
missing = features[all_feature_cols].isna().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, n in missing.items():
        print(f"  {col:30s}: {n:5d} ({n/len(features):.1%})")
else:
    print("  None (except demographics)")

# %% [markdown]
# ## 11. Build Customer-Month Panel from Holdout
#
# We divide the 50-week holdout into **12 four-week periods** (the last period
# absorbs the remaining 2 weeks: weeks 97–102 = 6 weeks). For each
# customer x period we record:
#
# - `purchased` (binary): did they make any purchase?
# - `period_spend`: total SALES_VALUE (0 if no purchase)
#
# This gives us ~2,500 x 12 = ~30,000 panel rows with directly observable
# monthly purchase behavior — no conversion assumptions needed.

# %%
step("Build customer-month panel from holdout")

# Define period boundaries: 12 four-week periods starting at week 53
# Period 1: weeks 53-56, Period 2: weeks 57-60, ..., Period 12: weeks 97-102
period_bounds = []
for i in range(N_HOLDOUT_PERIODS):
    start_week = cal_end + 1 + i * WEEKS_PER_PERIOD  # 53, 57, 61, ...
    if i < N_HOLDOUT_PERIODS - 1:
        end_week = start_week + WEEKS_PER_PERIOD - 1  # 56, 60, 64, ...
    else:
        end_week = CONFIG["holdout_end_week"]  # Period 12 absorbs extra weeks (97-102)
    period_bounds.append((i + 1, start_week, end_week))

print("Period boundaries:")
for period_id, sw, ew in period_bounds:
    n_weeks = ew - sw + 1
    print(f"  Period {period_id:2d}: weeks {sw}–{ew} ({n_weeks} weeks)")

# Assign each holdout transaction to a period
hold_txn = hold_txn[hold_txn["household_key"].isin(cal_households)].copy()

def assign_period(week_no):
    for period_id, sw, ew in period_bounds:
        if sw <= week_no <= ew:
            return period_id
    return None

hold_txn["period"] = hold_txn["WEEK_NO"].apply(assign_period)
hold_txn = hold_txn.dropna(subset=["period"])
hold_txn["period"] = hold_txn["period"].astype(int)

# Aggregate: for each customer x period, compute spend
period_agg = (
    hold_txn.groupby(["household_key", "period"])["SALES_VALUE"]
    .sum()
    .rename("period_spend")
    .reset_index()
)

# Build the full panel: all customers x all periods (fill missing with 0)
panel_idx = pd.MultiIndex.from_product(
    [cal_households, range(1, N_HOLDOUT_PERIODS + 1)],
    names=["household_key", "period"],
)
panel = pd.DataFrame(index=panel_idx).reset_index()
panel = panel.merge(period_agg, on=["household_key", "period"], how="left")
panel["period_spend"] = panel["period_spend"].fillna(0)

# Binary purchase indicator
panel["purchased"] = (panel["period_spend"] > 0).astype(int)

print(f"\nPanel shape: {panel.shape[0]:,} rows ({len(cal_households):,} customers x {N_HOLDOUT_PERIODS} periods)")
print(f"Purchase rate: {panel['purchased'].mean():.3f} (fraction of customer-periods with a purchase)")
print(f"Mean spend (all): ${panel['period_spend'].mean():.2f}")
print(f"Mean spend (purchase-periods only): ${panel.loc[panel['purchased']==1, 'period_spend'].mean():.2f}")

# %% [markdown]
# ## 12. Holdout Summary per Customer (for evaluation)

# %%
step("Holdout summary per customer")

# Total holdout spend
holdout_spend = (
    hold_txn.groupby("household_key")["SALES_VALUE"]
    .sum()
    .reindex(cal_households, fill_value=0)
    .rename("y_holdout_spend")
)

# Number of active periods per customer
n_active_periods = (
    panel[panel["purchased"] == 1]
    .groupby("household_key")
    .size()
    .reindex(cal_households, fill_value=0)
    .rename("n_active_periods")
)

# Add holdout-level targets to features
features["y_holdout_spend"] = holdout_spend
features["n_active_periods"] = n_active_periods

print(f"Holdout spend distribution:")
print(holdout_spend.describe().round(2))
print(f"Zero-spend households: {(holdout_spend == 0).sum()} ({(holdout_spend == 0).mean():.1%})")
print(f"\nActive periods distribution:")
print(n_active_periods.describe().round(2))

# %% [markdown]
# ## 13. Save Outputs

# %%
step("Save outputs")

# Save feature table (customer-level)
features.to_parquet(_TABLES / "ml_clv_features.parquet")
print(f"Saved: {_TABLES / 'ml_clv_features.parquet'} ({len(features)} rows)")

# Save customer-month panel
panel.to_parquet(_TABLES / "ml_clv_panel.parquet", index=False)
print(f"Saved: {_TABLES / 'ml_clv_panel.parquet'} ({len(panel)} rows)")

# Save feature group definitions
import json
with open(_TABLES / "feature_groups.json", "w") as f:
    json.dump(FEATURE_GROUPS, f, indent=2)
print(f"Saved: {_TABLES / 'feature_groups.json'}")

# Save constants for downstream scripts
constants = {
    "N_HOLDOUT_PERIODS": N_HOLDOUT_PERIODS,
    "WEEKS_PER_PERIOD": WEEKS_PER_PERIOD,
    "DISCOUNT_RATE": DISCOUNT_RATE,
    "CLV_HORIZON": CLV_HORIZON,
}
with open(_TABLES / "constants.json", "w") as f:
    json.dump(constants, f, indent=2)
print(f"Saved: {_TABLES / 'constants.json'}")

# %% [markdown]
# ## 14. Target Distribution

# %%
step("Target distribution")

y = features["y_holdout_spend"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw distribution
ax = axes[0]
ax.hist(y[y > 0], bins=50, color=ACCENT_BLUE, edgecolor="white", alpha=0.8)
ax.axvline(y.median(), color=ACCENT_ORANGE, linestyle="--", label=f"Median: ${y.median():,.0f}")
ax.axvline(y.mean(), color=ACCENT_GREEN, linestyle="--", label=f"Mean: ${y.mean():,.0f}")
ax.set_xlabel("Holdout Spend ($)")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Holdout-Period Spend\n(zero-spend excluded)")
ax.legend()

# Right: purchase rate per customer
ax = axes[1]
buy_rate = panel.groupby("household_key")["purchased"].mean()
ax.hist(buy_rate, bins=13, color=ACCENT_BLUE, edgecolor="white", alpha=0.8)
ax.set_xlabel("Purchase Rate (fraction of 12 periods)")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Customer Purchase Rate\n(observed in holdout)")

fig.tight_layout()
savefig(fig, "ml_clv_target_distribution.png")
plt.show()

print(f"Holdout spend summary:")
print(f"  Households: {len(y):,}")
print(f"  Zero-spend: {(y == 0).sum()} ({(y == 0).mean():.1%})")
print(f"  Mean:   ${y.mean():,.2f}")
print(f"  Median: ${y.median():,.2f}")
print(f"  Max:    ${y.max():,.2f}")

# %% [markdown]
# ## Summary
#
# Two files saved:
# - **`ml_clv_features.parquet`**: ~2,500 households × 30 features + holdout targets
# - **`ml_clv_panel.parquet`**: ~30,000 customer-period rows with `purchased` (binary) and `period_spend`
#
# Constants aligned with PyMC demo:
# - `DISCOUNT_RATE = 0.01` (monthly)
# - `CLV_HORIZON = 120` (months = 10 years)
# - `N_HOLDOUT_PERIODS = 12` (four-week periods)
#
# Next: `02_modeling.py` trains purchase classifier + amount regressor.
