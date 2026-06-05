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
# # Dunnhumby Grocery — EDA & Data Preparation
#
# This notebook prepares **real-world grocery data** for the demand forecasting
# workflow in Chapter 5.3. We use the
# [Dunnhumby "The Complete Journey"](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
# dataset (CC0 license) — a household panel of ~2,500 frequent shoppers
# at a US grocery retailer over 102 weeks.
#
# **Goal:** Aggregate transactions to a weekly commodity × store panel with
# price and promotional features, outputting clean Nixtla-format parquet
# for the `sec5.3_demand_forecasting` notebook.

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option("display.float_format", "{:.2f}".format)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

from msbook.paths import chapter_generated, chapter_images, chapter_artifacts

CONFIG = {
    "data_dir": "../../data/raw/Dunnhumby_kaggle",
    "output_path": str(
        chapter_generated(part="5", chapter="sec5.3-demand-forecast")
        / "dunnhumby_grocery_weekly.parquet"
    ),
    "img_dir": str(chapter_images(part="5", chapter="sec5.3")),
    "selected_commodities": [
        "SOFT DRINKS",
        "FLUID MILK PRODUCTS",
        "BAKED BREAD/BUNS/ROLLS",
        "CHEESE",
        "BAG SNACKS",
        "YOGURT",
        "COLD CEREAL",
    ],
    "selected_stores": [367, 406, 381, 292, 356],
    "min_weeks": 80,
    # Synthetic reference date for WEEK_NO -> calendar date mapping.
    # Dunnhumby uses DAY (1-711) and WEEK_NO (1-102). We anchor
    # WEEK_NO 1 to a Monday so that downstream tools see a proper
    # weekly frequency.
    "reference_date": pd.Timestamp("2012-01-02"),  # a Monday
}

# chapter_images / chapter_generated already mkdir'd the dirs.
# Scratch tables (eda summary) land in artifacts/ — gitignored.
_SCRATCH_TABLES = chapter_artifacts(part="5", chapter="sec5.3-demand-forecast") / "tables"
_SCRATCH_TABLES.mkdir(parents=True, exist_ok=True)

_FIG_DIR = chapter_images(part="5", chapter="sec5.3")


def dual_savefig(fig, name, **kw):
    """Single-saver shim (no local scratch copy) — see Step 5 of the folder redesign."""
    kw.setdefault("dpi", 150)
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(_FIG_DIR / name, **kw)


# %% [markdown]
# ## 1. Load Product Catalog
#
# The product table provides the hierarchy:
# DEPARTMENT → COMMODITY_DESC → SUB_COMMODITY_DESC → PRODUCT_ID.

# %%
prod = pd.read_csv(
    os.path.join(CONFIG["data_dir"], "product.csv"),
    usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC"],
)
target_pids = set(
    prod[prod["COMMODITY_DESC"].isin(CONFIG["selected_commodities"])]["PRODUCT_ID"]
)
print(f"Product catalog: {len(prod):,} products")
print(f"Target commodities: {len(CONFIG['selected_commodities'])}")
print(f"  Products in scope: {len(target_pids):,}")


# %% [markdown]
# ## 2. Load Transaction Data
#
# The Dunnhumby dataset tracks purchases by ~2,500 household panelists.
# This is **sampled** demand (not total store sales), so we aggregate to
# commodity × store × week to get a meaningful signal.

# %%
def load_transactions(config, target_pids):
    """Read transaction_data.csv in chunks, keeping only target products and stores."""
    usecols = [
        "PRODUCT_ID", "STORE_ID", "WEEK_NO", "DAY",
        "QUANTITY", "SALES_VALUE", "RETAIL_DISC",
    ]
    frames = []
    for chunk in pd.read_csv(
        os.path.join(config["data_dir"], "transaction_data.csv"),
        usecols=usecols, chunksize=500_000,
    ):
        mask = (
            chunk["PRODUCT_ID"].isin(target_pids)
            & chunk["STORE_ID"].isin(config["selected_stores"])
            & (chunk["QUANTITY"] > 0)
            & (chunk["SALES_VALUE"] > 0)
        )
        subset = chunk[mask]
        if len(subset):
            frames.append(subset)

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} transactions  |  "
          f"WEEK_NO {df['WEEK_NO'].min()}–{df['WEEK_NO'].max()}  |  "
          f"{df['STORE_ID'].nunique()} stores, {df['PRODUCT_ID'].nunique()} products")
    return df


txn = load_transactions(CONFIG, target_pids)
txn = txn.merge(
    prod[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left"
)
txn.head()


# %% [markdown]
# ## 3. Weekly Aggregation
#
# Aggregate to **commodity × store × week** level.
# - `y` = total units (QUANTITY)
# - `effective_price` = revenue / units (average price actually paid)
# - `discount_pct` = |RETAIL_DISC| / (SALES_VALUE + |RETAIL_DISC|), the markdown fraction

# %%
# Convert WEEK_NO to Monday-start dates
txn["ds"] = CONFIG["reference_date"] + pd.to_timedelta(
    (txn["WEEK_NO"] - 1) * 7, unit="D"
)

weekly = (
    txn.groupby(["COMMODITY_DESC", "STORE_ID", "WEEK_NO", "ds"])
    .agg(
        y=("QUANTITY", "sum"),
        revenue=("SALES_VALUE", "sum"),
        total_discount=("RETAIL_DISC", lambda x: x.abs().sum()),
    )
    .reset_index()
)

# Derived columns
weekly["effective_price"] = weekly["revenue"] / weekly["y"]
weekly["discount_pct"] = weekly["total_discount"] / (
    weekly["revenue"] + weekly["total_discount"]
)

# Nixtla composite ID
weekly["unique_id"] = (
    weekly["COMMODITY_DESC"] + "__" + weekly["STORE_ID"].astype(str)
)
# Shorter column names for downstream
weekly = weekly.rename(columns={
    "COMMODITY_DESC": "commodity",
    "STORE_ID": "store_id",
})

print(f"Weekly panel: {len(weekly):,} rows, {weekly['unique_id'].nunique()} series")
print(f"Weeks: {weekly['WEEK_NO'].min()}–{weekly['WEEK_NO'].max()}")


# %% [markdown]
# ## 4. Load Promotional (Causal) Data
#
# The causal_data file (37M rows) contains weekly in-store `display` and
# `mailer` flags per product × store. We aggregate to commodity × store × week
# as the fraction of products with an active promotion.

# %%
def load_causal(config, target_pids):
    """Read causal_data.csv in chunks, filtered to target scope."""
    frames = []
    for chunk in pd.read_csv(
        os.path.join(config["data_dir"], "causal_data.csv"),
        chunksize=1_000_000,
    ):
        mask = (
            chunk["PRODUCT_ID"].isin(target_pids)
            & chunk["STORE_ID"].isin(config["selected_stores"])
        )
        subset = chunk[mask]
        if len(subset):
            frames.append(subset)

    df = pd.concat(frames, ignore_index=True)
    print(f"Causal data: {len(df):,} rows  |  "
          f"WEEK_NO {df['WEEK_NO'].min()}–{df['WEEK_NO'].max()}")
    return df


causal = load_causal(CONFIG, target_pids)
causal = causal.merge(
    prod[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left"
)

# Binary flags
causal["has_display"] = (causal["display"].astype(str) != "0").astype(int)
causal["has_mailer"] = (~causal["mailer"].astype(str).isin(["A"])).astype(int)

# Aggregate to commodity × store × week: fraction of product-weeks with promotion
promo_weekly = (
    causal.groupby(["COMMODITY_DESC", "STORE_ID", "WEEK_NO"])
    .agg(
        display_pct=("has_display", "mean"),
        mailer_pct=("has_mailer", "mean"),
    )
    .reset_index()
    .rename(columns={"COMMODITY_DESC": "commodity", "STORE_ID": "store_id"})
)

print(f"Promo features: {len(promo_weekly):,} commodity × store × week rows")
print(f"  display_pct mean: {promo_weekly['display_pct'].mean():.2f}")
print(f"  mailer_pct mean:  {promo_weekly['mailer_pct'].mean():.2f}")


# %% [markdown]
# ## 5. Merge & Complete the Panel

# %%
# Merge promo features into weekly panel
weekly = weekly.merge(
    promo_weekly,
    on=["commodity", "store_id", "WEEK_NO"],
    how="left",
)
# Causal data covers weeks 9-101; fill missing promo weeks with 0
weekly["display_pct"] = weekly["display_pct"].fillna(0)
weekly["mailer_pct"] = weekly["mailer_pct"].fillna(0)

# Complete the panel: every series gets every week (zero-fill missing weeks)
all_weeks_no = range(weekly["WEEK_NO"].min(), weekly["WEEK_NO"].max() + 1)
all_ds = [
    CONFIG["reference_date"] + pd.Timedelta(weeks=w - 1)
    for w in all_weeks_no
]
week_map = pd.DataFrame({"WEEK_NO": list(all_weeks_no), "ds": all_ds})

uids = weekly["unique_id"].unique()
full_index = pd.MultiIndex.from_product(
    [uids, list(all_weeks_no)], names=["unique_id", "WEEK_NO"]
)
weekly = (
    weekly.set_index(["unique_id", "WEEK_NO"])
    .reindex(full_index)
    .reset_index()
)

# Fill missing y with 0
weekly["y"] = weekly["y"].fillna(0)

# Forward-fill metadata and features
for col in ["commodity", "store_id"]:
    weekly[col] = weekly.groupby("unique_id")[col].ffill().bfill()
for col in ["effective_price", "discount_pct"]:
    weekly[col] = weekly.groupby("unique_id")[col].ffill().bfill()
for col in ["display_pct", "mailer_pct"]:
    weekly[col] = weekly[col].fillna(0)

# Restore ds from WEEK_NO
weekly = weekly.drop(columns=["ds"], errors="ignore").merge(week_map, on="WEEK_NO")
weekly["store_id"] = weekly["store_id"].astype(int)

# Filter out series with too few weeks
series_length = weekly.groupby("unique_id")["ds"].nunique()
keep_ids = series_length[series_length >= CONFIG["min_weeks"]].index
dropped = weekly["unique_id"].nunique() - len(keep_ids)
weekly = weekly[weekly["unique_id"].isin(keep_ids)].copy()
print(f"Kept {len(keep_ids)} series with >= {CONFIG['min_weeks']} weeks")
if dropped:
    print(f"  Dropped {dropped} series with insufficient history")

zero_pct = (weekly["y"] == 0).mean()
print(f"Zero-sales weeks: {(weekly['y'] == 0).sum():,} ({zero_pct:.1%})")
print(f"Panel: {weekly['unique_id'].nunique()} series × "
      f"{weekly['ds'].nunique()} weeks = {len(weekly):,} rows")


# %% [markdown]
# ## 6. Summary Statistics

# %%
summary = (
    weekly.groupby("unique_id")
    .agg(
        commodity=("commodity", "first"),
        store_id=("store_id", "first"),
        weeks=("ds", "nunique"),
        total_units=("y", "sum"),
        avg_weekly_units=("y", "mean"),
        std_weekly_units=("y", "std"),
        avg_price=("effective_price", "mean"),
        price_cv=("effective_price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        avg_display=("display_pct", "mean"),
        avg_mailer=("mailer_pct", "mean"),
    )
)
summary["demand_cv"] = summary["std_weekly_units"] / summary["avg_weekly_units"]
summary.to_csv(_SCRATCH_TABLES / "eda_series_summary.csv", index=True)

print(f"Panel summary: {len(summary)} series")
print(f"  Weeks range: {summary['weeks'].min()} – {summary['weeks'].max()}")
print(f"  Mean weekly units: {summary['avg_weekly_units'].mean():.1f}")
print(f"  Median price CV: {summary['price_cv'].median():.3f}")
print(f"  Mean display activity: {summary['avg_display'].mean():.2%}")
print(f"  Mean mailer activity: {summary['avg_mailer'].mean():.2%}")
summary.sort_values("total_units", ascending=False)


# %% [markdown]
# ## 7. EDA Plots

# %%
# Plot 1: Effective price over time by commodity (one store for clarity)
plot_store = CONFIG["selected_stores"][0]
fig, axes = plt.subplots(len(CONFIG["selected_commodities"]), 1,
                         figsize=(14, 2.5 * len(CONFIG["selected_commodities"])),
                         sharex=True)
for ax, commodity in zip(axes, CONFIG["selected_commodities"]):
    for sid in CONFIG["selected_stores"]:
        sub = weekly[(weekly["commodity"] == commodity) & (weekly["store_id"] == sid)]
        sub = sub.sort_values("ds")
        ax.plot(sub["ds"], sub["effective_price"], linewidth=0.8,
                label=f"Store {sid}", alpha=0.7)
    ax.set_ylabel("Price ($)")
    ax.set_title(commodity, fontsize=9)
    ax.legend(fontsize=6, ncol=5)

fig.suptitle("Effective Price Over Time — by Commodity & Store", fontsize=13, y=1.01)
fig.tight_layout()
dual_savefig(fig, "01_price_variation.png")
plt.show()
plt.close(fig)

# %%
# Plot 2: Promotional activity over time
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# display_pct averaged across stores
for commodity in CONFIG["selected_commodities"]:
    sub = weekly[weekly["commodity"] == commodity].groupby("ds")["display_pct"].mean()
    axes[0].plot(sub.index, sub.values, linewidth=0.8, label=commodity)
axes[0].set_ylabel("Display %")
axes[0].set_title("In-Store Display Activity", fontsize=11)
axes[0].legend(fontsize=7, ncol=4)

for commodity in CONFIG["selected_commodities"]:
    sub = weekly[weekly["commodity"] == commodity].groupby("ds")["mailer_pct"].mean()
    axes[1].plot(sub.index, sub.values, linewidth=0.8, label=commodity)
axes[1].set_ylabel("Mailer %")
axes[1].set_title("Mailer/Flyer Activity", fontsize=11)
axes[1].legend(fontsize=7, ncol=4)

fig.suptitle("Promotional Activity Over Time", fontsize=13, y=1.01)
fig.tight_layout()
dual_savefig(fig, "01_promo_activity.png")
plt.show()
plt.close(fig)

# %%
# Plot 3: Weekly demand overview
n_commodities = len(CONFIG["selected_commodities"])
fig, axes = plt.subplots(n_commodities, 1,
                         figsize=(14, 2.5 * n_commodities), sharex=True)
for ax, commodity in zip(axes, CONFIG["selected_commodities"]):
    for sid in CONFIG["selected_stores"]:
        sub = weekly[(weekly["commodity"] == commodity) & (weekly["store_id"] == sid)]
        sub = sub.sort_values("ds")
        ax.plot(sub["ds"], sub["y"], linewidth=0.6, alpha=0.6,
                label=f"Store {sid}")
    ax.set_ylabel("Units")
    ax.set_title(commodity, fontsize=9)
    ax.legend(fontsize=6, ncol=5)

fig.suptitle("Weekly Demand by Commodity & Store", fontsize=13, y=1.01)
fig.tight_layout()
dual_savefig(fig, "01_demand_overview.png")
plt.show()
plt.close(fig)


# %% [markdown]
# ## 8. Save Output

# %%
out_cols = [
    "unique_id", "ds", "y",
    "effective_price", "discount_pct", "display_pct", "mailer_pct",
    "commodity", "store_id",
]
weekly[out_cols].to_parquet(CONFIG["output_path"], index=False)

size_mb = os.path.getsize(CONFIG["output_path"]) / 1e6
print(f"Saved {CONFIG['output_path']}  ({size_mb:.1f} MB)")
print(f"  {weekly['unique_id'].nunique()} series, {len(weekly):,} rows")
print(f"  Date range: {weekly['ds'].min():%Y-%m-%d} to {weekly['ds'].max():%Y-%m-%d}")
print(f"  Columns: {out_cols}")
