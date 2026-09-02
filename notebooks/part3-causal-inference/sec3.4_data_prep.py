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
# # Dunnhumby Store-Level Daily Series — Data Preparation
#
# Chapter 3.4 (causal impact with time series) needs one treated series with
# real, messy structure plus several untreated series that could serve as
# controls. The Dunnhumby transactions aggregate into ~711 **daily**
# observations per store, long enough to fit a pre-period model and project a
# counterfactual.
#
# **Goal:** store-level daily unit sales (plus a chain total) as a committed
# seed parquet, so the chapter notebook re-runs without touching the raw
# 2.6M-row transaction file.
#
# One data trap handled here: the `COUPON/MISC ITEMS` commodity contains fuel
# kiosk rows with QUANTITY in the thousands of gallons-equivalent units per
# transaction. Left in, they dwarf grocery units at the two stores with gas
# kiosks. We exclude that commodity (and the KIOSK-GAS department) up front.

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.float_format", "{:.2f}".format)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

from msbook.paths import DATA_DIR, chapter_artifacts, chapter_generated

CONFIG = {
    "data_dir": str(DATA_DIR / "raw" / "Dunnhumby_kaggle"),
    "output_path": str(
        chapter_generated(part="3", chapter="sec3.4-causal-impact")
        / "dunnhumby_store_daily.parquet"
    ),
    "selected_stores": [367, 406, 381, 292, 356],  # the five busiest grocery stores
    # Calendar anchor: WEEK_NO 1 -> Monday 2012-01-02. In the
    # raw data WEEK_NO = (DAY + 1) // 7 + 1 holds exactly, so week w spans
    # DAYs 7w-8 .. 7w-2 and DAY d maps to reference_date + (d + 1) days
    # (verified below), so the daily calendar starts on a Monday.
    "reference_date": pd.Timestamp("2012-01-02"),
}

# EDA figures are scratch, not chapter assets: they go to gitignored artifacts/.
_FIG_DIR = chapter_artifacts(part="3", chapter="sec3.4-causal-impact") / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def dual_savefig(fig, name, **kw):
    """Single-saver shim for the scratch EDA figures."""
    kw.setdefault("dpi", 150)
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(_FIG_DIR / name, **kw)


# %% [markdown]
# ## 1. Product Exclusions
#
# Identify fuel/coupon PRODUCT_IDs to drop before aggregating units.

# %%
prod = pd.read_csv(
    os.path.join(CONFIG["data_dir"], "product.csv"),
    usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"],
)
excluded_mask = (
    (prod["COMMODITY_DESC"] == "COUPON/MISC ITEMS")
    | (prod["DEPARTMENT"].str.contains("GAS", na=False))
)
excluded_pids = set(prod.loc[excluded_mask, "PRODUCT_ID"])

print(f"Product catalog: {len(prod):,} products")
print(f"Excluded products (fuel/coupon): {len(excluded_pids):,}")
print(prod.loc[excluded_mask, ["DEPARTMENT", "COMMODITY_DESC"]]
      .value_counts().head(10))


# %% [markdown]
# ## 2. Load Transactions and Aggregate to Store × Day

# %%
def load_daily(config, excluded_pids):
    """Chunked read of transaction_data.csv -> store x DAY unit/revenue sums."""
    usecols = ["PRODUCT_ID", "STORE_ID", "DAY", "WEEK_NO",
               "QUANTITY", "SALES_VALUE"]
    parts = []
    for chunk in pd.read_csv(
        os.path.join(config["data_dir"], "transaction_data.csv"),
        usecols=usecols, chunksize=500_000,
    ):
        mask = (
            chunk["STORE_ID"].isin(config["selected_stores"])
            & ~chunk["PRODUCT_ID"].isin(excluded_pids)
            & (chunk["QUANTITY"] > 0)
            & (chunk["SALES_VALUE"] > 0)
        )
        subset = chunk[mask]
        if len(subset):
            parts.append(
                subset.groupby(["STORE_ID", "DAY", "WEEK_NO"])
                .agg(y=("QUANTITY", "sum"), revenue=("SALES_VALUE", "sum"))
                .reset_index()
            )
    df = (
        pd.concat(parts, ignore_index=True)
        .groupby(["STORE_ID", "DAY", "WEEK_NO"])
        .sum()
        .reset_index()
    )
    print(f"Store-day rows: {len(df):,}  |  DAY {df['DAY'].min()}–{df['DAY'].max()}")
    return df


daily = load_daily(CONFIG, excluded_pids)

# Sanity check: DAY and WEEK_NO must agree with the (DAY+1)//7 + 1 mapping,
# otherwise the calendar anchor drifts off Monday.
implied_week = (daily["DAY"] + 1) // 7 + 1
mismatch = (implied_week != daily["WEEK_NO"]).mean()
print(f"DAY↔WEEK_NO mapping mismatch: {mismatch:.2%} (must be 0%)")
assert mismatch == 0.0

# %% [markdown]
# ## 3. Complete the Daily Panel and Add the Chain Total

# %%
daily["ds"] = CONFIG["reference_date"] + pd.to_timedelta(daily["DAY"] + 1, unit="D")

all_days = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
frames = []
for sid in CONFIG["selected_stores"]:
    sub = (
        daily[daily["STORE_ID"] == sid]
        .set_index("ds")[["y", "revenue"]]
        .reindex(all_days)
        .fillna(0)
        .rename_axis("ds")
        .reset_index()
    )
    sub.insert(0, "unique_id", f"store_{sid}")
    frames.append(sub)

panel = pd.concat(frames, ignore_index=True)

chain = (
    panel.groupby("ds")[["y", "revenue"]].sum().reset_index()
)
chain.insert(0, "unique_id", "chain_total")
panel = pd.concat([panel, chain], ignore_index=True)

zero_days = (panel["y"] == 0).groupby(panel["unique_id"]).sum()
print(f"Panel: {panel['unique_id'].nunique()} series × {len(all_days)} days "
      f"= {len(panel):,} rows")
print(f"Date range: {panel['ds'].min():%Y-%m-%d} to {panel['ds'].max():%Y-%m-%d}")
print("Zero-sales days per series:")
print(zero_days)

# %% [markdown]
# ## 4. Summary Statistics and Overview Plot

# %%
summary = (
    panel.groupby("unique_id")["y"]
    .agg(["mean", "median", "std", "min", "max"])
    .round(1)
)
print(summary)

dow = panel[panel["unique_id"] == "chain_total"].copy()
dow["dow"] = dow["ds"].dt.day_name()
dow_means = dow.groupby("dow")["y"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
print("\nChain-total mean units by day of week:")
print(dow_means.round(0))
print(f"Peak/trough ratio: {dow_means.max() / dow_means.min():.2f}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5),
                         gridspec_kw={"height_ratios": [2, 1]})
sub = panel[panel["unique_id"] == "chain_total"]
axes[0].plot(sub["ds"], sub["y"], lw=0.6, color="tab:blue")
axes[0].set_title("Chain-total daily units (5 stores, fuel excluded)", fontsize=11)
axes[0].set_ylabel("Units")

dow_means.plot(kind="bar", ax=axes[1], color="tab:blue", alpha=0.8, rot=0)
axes[1].set_title("Mean units by day of week", fontsize=11)
axes[1].set_ylabel("Units")
axes[1].set_xlabel("")
axes[1].tick_params(labelsize=8)

fig.tight_layout()
dual_savefig(fig, "01_store_daily_overview.png")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 5. Save Output

# %%
out_cols = ["unique_id", "ds", "y", "revenue"]
panel[out_cols].to_parquet(CONFIG["output_path"], index=False)

size_kb = os.path.getsize(CONFIG["output_path"]) / 1e3
print(f"Saved {CONFIG['output_path']}  ({size_kb:.0f} KB)")
print(f"  {panel['unique_id'].nunique()} series, {len(panel):,} rows")
print(f"  Columns: {out_cols}")
