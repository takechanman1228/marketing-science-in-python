# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takechanman1228/marketing-science-in-python/blob/main/notebooks/sec4.1-segmentation/traditional_segmentation.ipynb)
#
# *Source of truth is the paired `.py`; this `.ipynb` is generated — see CONTRIBUTING.md.*

# %% [markdown]
# # Customer Segmentation — Traditional Methods
#
# This notebook accompanies **§4.1** of *Marketing Science in Python*.
# It walks through the three layers the chapter calls out — decile, RFM, and
# K-Means — and ends with a CRM-ready export.
#
# For embedding-based segmentation (BERTopic + LLM naming), see the companion
# script `semantic_segmentation.py`.
#
# **Learning objectives**
#
# 1. Use decile analysis to reveal revenue concentration.
# 2. Compute RFM (Recency, Frequency, Monetary) scores with **M-aware** segment labels.
# 3. Engineer 8–10 behavioral features (frequency, spend, recency, basket size,
#    category breadth, discount sensitivity, …) and scale them.
# 4. Choose K with elbow + silhouette **and** a business-judgment sanity check.
# 5. Profile segments with a heatmap and demographic cross-tabs.
# 6. Test temporal stability with a decile migration matrix.
# 7. Export CRM-ready segment assignments.
#
# **Dataset**: Dunnhumby "The Complete Journey" — 2.6 M transaction rows joined
# with product master and household demographics.

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── All tuneable parameters in one place ─────────────────────────────────────
CONFIG = {
    "data_path": "../Retail_open_data_set/Dunnhumby_kaggle/archive/",
    "max_week": 102,           # last week in the dataset
    "rfm_quintiles": 5,        # number of quintile bins for RFM
    "k_range": range(2, 11),   # K values to search for clustering
    "final_k": 3,              # business-judgment choice (validated below)
    "min_cluster_share": 0.10, # 10% floor for actionable segments
    "random_state": 42,
}

ACCENT_BLUE = "#2171b5"
ACCENT_ORANGE = "#e6550d"
ACCENT_GREEN = "#31a354"
ACCENT_PURPLE = "#756bb1"
ACCENT_GREY = "#969696"

pd.set_option("display.float_format", "{:.2f}".format)

try:
    display
except NameError:
    display = print
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

# %%
from msbook.paths import chapter_images, chapter_artifacts

_FIG_DIR = chapter_images(part="4", chapter="sec4.1")
_TABLES  = chapter_artifacts(part="4", chapter="sec4.1-segmentation") / "tables"
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
# ## 1. Data Loading & Feature Engineering
#
# The Dunnhumby dataset contains:
#
# | File | Contents | Key Columns |
# |------|----------|-------------|
# | `transaction_data.csv` | 2.6 M transaction rows | `household_key`, `BASKET_ID`, `SALES_VALUE`, `RETAIL_DISC`, `COUPON_DISC`, `WEEK_NO` |
# | `product.csv` | Product master | `PRODUCT_ID`, `DEPARTMENT`, `COMMODITY_DESC` |
# | `hh_demographic.csv` | Household demographics (801 HHs) | `AGE_DESC`, `INCOME_DESC`, `HOUSEHOLD_SIZE_DESC`, etc. |
#
# We aggregate transactions to one row per household, computing **nine
# behavioral features** that go into clustering. The chapter calls for 6–10
# such variables that capture more than spend, frequency, and recency.
#
# | Feature | Definition |
# |---|---|
# | `total_revenue` | Σ `SALES_VALUE` per household |
# | `frequency` | distinct baskets (`BASKET_ID`) |
# | `recency_weeks` | `max_week - last_purchase_week` |
# | `avg_order_value` | `total_revenue / frequency` |
# | `basket_size` | mean items per basket |
# | `active_weeks` | distinct `WEEK_NO` with a purchase |
# | `discount_share` | discount $ / gross $ (sensitivity to promotions) |
# | `department_breadth` | distinct departments shopped |
# | `top_department_share` | revenue share of the top department (concentration) |

# %%
step("Load raw data")

base = CONFIG["data_path"]
txn = pd.read_csv(base + "transaction_data.csv")
demographics = pd.read_csv(base + "hh_demographic.csv")
try:
    product = pd.read_csv(base + "product.csv")
except FileNotFoundError as e:
    raise FileNotFoundError(
        "product.csv not found in {0}. The 9-feature build needs the "
        "product master for department_breadth and top_department_share. "
        "Download Dunnhumby 'The Complete Journey' (Kaggle) and place "
        "product.csv next to transaction_data.csv.".format(base)
    ) from e

print(f"Transactions:  {len(txn):>10,} rows  |  {txn['household_key'].nunique():,} households")
print(f"Demographics:  {len(demographics):>10,} rows  |  {demographics['household_key'].nunique()} households")
print(f"Products:      {len(product):>10,} rows  |  {product['DEPARTMENT'].nunique()} departments")
print(f"Week range:    {txn['WEEK_NO'].min()}–{txn['WEEK_NO'].max()}")

# %%
step("Feature engineering — one row per household, nine behavioral features")

max_week = CONFIG["max_week"]

# Dunnhumby convention: RETAIL_DISC and COUPON_DISC are stored as
# *negative* values (the discount amount). gross = paid - disc.
txn["DISCOUNT"] = -(txn["RETAIL_DISC"].fillna(0) + txn["COUPON_DISC"].fillna(0))
txn["DISCOUNT"] = txn["DISCOUNT"].clip(lower=0)
txn["GROSS"] = txn["SALES_VALUE"] + txn["DISCOUNT"]

# Join department onto every transaction row
txn = txn.merge(product[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="left")

# Core RFM-style aggregates
agg = (
    txn.groupby("household_key")
    .agg(
        total_revenue=("SALES_VALUE", "sum"),
        gross_spend=("GROSS", "sum"),
        discount_spend=("DISCOUNT", "sum"),
        frequency=("BASKET_ID", "nunique"),
        last_week=("WEEK_NO", "max"),
        active_weeks=("WEEK_NO", "nunique"),
        total_items=("QUANTITY", "sum"),
        department_breadth=("DEPARTMENT", "nunique"),
    )
    .reset_index()
)

# Top-department concentration (max share of any single department's revenue)
dept_rev = (
    txn.groupby(["household_key", "DEPARTMENT"])["SALES_VALUE"].sum().reset_index()
)
top_dept = (
    dept_rev.groupby("household_key")["SALES_VALUE"].max().reset_index()
    .rename(columns={"SALES_VALUE": "top_dept_rev"})
)
agg = agg.merge(top_dept, on="household_key", how="left")
agg["top_department_share"] = (agg["top_dept_rev"] / agg["total_revenue"]).clip(0, 1)

# Derived per-household features
agg["recency_weeks"] = max_week - agg["last_week"]
agg["avg_order_value"] = agg["total_revenue"] / agg["frequency"]
agg["basket_size"] = agg["total_items"] / agg["frequency"]
agg["discount_share"] = (
    agg["discount_spend"] / agg["gross_spend"].replace(0, np.nan)
).fillna(0).clip(0, 1)

customers = agg[[
    "household_key",
    "total_revenue", "frequency", "recency_weeks",
    "avg_order_value", "basket_size", "active_weeks",
    "discount_share", "department_breadth", "top_department_share",
]].copy()

BASE_COLS = [
    "total_revenue", "frequency", "recency_weeks",
    "avg_order_value", "basket_size", "active_weeks",
    "discount_share", "department_breadth", "top_department_share",
]

print(f"Customers: {len(customers):,}")
print(f"\nFeature summary ({len(BASE_COLS)} behavioral variables):")
display(customers[BASE_COLS].describe().round(3))

customers.to_csv(_TABLES / "trad_customer_features.csv", index=False)

# %% [markdown]
# ## 2. Decile Analysis
#
# Divide customers into ten equal groups ranked by revenue. The top decile
# routinely captures 40–60 percent of revenue; the bottom four are rarely
# worth a paid touch. Mapping deciles to **action tiers** is the simplest
# possible segmentation that still drives spend.

# %%
step("Decile analysis")

customers = customers.sort_values("total_revenue", ascending=False)
customers["decile"] = pd.qcut(
    customers["total_revenue"].rank(method="first", ascending=False),
    q=10,
    labels=[f"D{i}" for i in range(1, 11)],
)

decile_profile = (
    customers.groupby("decile", observed=True)
    .agg(
        n_customers=("household_key", "count"),
        total_revenue=("total_revenue", "sum"),
        avg_frequency=("frequency", "mean"),
        avg_order_value=("avg_order_value", "mean"),
    )
)
decile_profile["revenue_share"] = (
    decile_profile["total_revenue"] / decile_profile["total_revenue"].sum()
)
decile_profile["cumulative_share"] = decile_profile["revenue_share"].cumsum()

# Action tiers (refined from the v1 notebook per chapter pass)
action_map = {
    "D1":  "VIP: retain and reward",
    "D2":  "Grow basket or category",
    "D3":  "Grow basket or category",
    "D4":  "Grow basket or category",
    "D5":  "Low-cost nurture",
    "D6":  "Low-cost nurture",
    "D7":  "Low-cost nurture",
    "D8":  "Suppress expensive offers",
    "D9":  "Suppress expensive offers",
    "D10": "Suppress expensive offers",
}
decile_profile["action_tier"] = decile_profile.index.map(action_map)
customers["decile_action_tier"] = customers["decile"].map(action_map)

decile_profile.to_csv(_TABLES / "trad_decile_profile.csv", index=True)
display(decile_profile)

# %%
step("Decile visualizations")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
decile_profile["revenue_share"].plot(kind="bar", ax=ax, color=ACCENT_BLUE)
ax.set_ylabel("Revenue Share")
ax.set_xlabel("Decile")
ax.set_title("Revenue Contribution by Decile")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.tick_params(axis="x", rotation=0)

ax = axes[1]
x = np.arange(1, 11) / 10
ax.plot(x, decile_profile["cumulative_share"].values, "o-",
        color=ACCENT_BLUE, label="Actual")
ax.plot([0, 1], [0, 1], "--", color=ACCENT_GREY, label="Perfect equality")
ax.fill_between(x, decile_profile["cumulative_share"].values, x,
                alpha=0.15, color=ACCENT_BLUE)
ax.set_xlabel("Cumulative % of Customers (by decile)")
ax.set_ylabel("Cumulative % of Revenue")
ax.set_title("Revenue Concentration Curve")
ax.legend()

fig.tight_layout()
savefig(fig, "decile_distribution.png")
plt.show()

d1_share = decile_profile.loc["D1", "revenue_share"]
print(f"D1 (top 10%) contributes {d1_share:.1%} of total revenue.")

# %% [markdown]
# ## 3. RFM Analysis
#
# RFM scores each customer on three quintiles:
#
# - **Recency (R)**: weeks since last purchase (lower = better → higher score)
# - **Frequency (F)**: distinct baskets (higher = better)
# - **Monetary (M)**: total spend (higher = better)
#
# **The label rule uses all three axes.** A common shortcut is to label only
# from R and F — that produces "Champions" who never spent much. We require
# `M ≥ 3` before assigning any high-value label.

# %%
step("RFM analysis (M-aware labels)")

rfm = customers[["household_key", "recency_weeks", "frequency", "total_revenue"]].copy()
rfm.columns = ["household_key", "recency", "frequency", "monetary"]

# Lower recency → higher score (5)
rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_score"] = (
    rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
)


def rfm_segment(row):
    r, f, m = int(row["R"]), int(row["F"]), int(row["M"])
    # High-value tiers — gated on M
    if r >= 4 and f >= 4 and m >= 3:
        return "Champions"
    if r >= 4 and f >= 3 and m >= 3:
        return "Loyal Customers"
    if m >= 4 and f <= 2:
        return "Big Spenders"          # high M with low F (occasional whales)
    if 2 <= r <= 3 and f >= 3 and m >= 3:
        return "At Risk"
    if r <= 2 and f >= 3 and m >= 4:
        return "Can't Lose Them"
    if r >= 4 and f <= 2 and m <= 3:
        return "New Customers"
    if r >= 3 and f >= 2 and m >= 2:
        return "Potential Loyalists"
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating"
    return "Need Attention"


rfm["segment"] = rfm.apply(rfm_segment, axis=1)
rfm.to_csv(_TABLES / "trad_rfm_segments.csv", index=False)

rfm_profile = (
    rfm.groupby("segment")
    .agg(
        customers=("household_key", "count"),
        avg_R=("R", "mean"),
        avg_F=("F", "mean"),
        avg_M=("M", "mean"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    .sort_values("avg_monetary", ascending=False)
)
display(rfm_profile.round(2))

# %%
step("RFM segment distribution")

seg_counts = rfm["segment"].value_counts()
colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
          ACCENT_GREY, "#e377c2", "#bcbd22", "#17becf", "#aec7e8"]

fig, ax = plt.subplots(figsize=(10, 5))
seg_counts.plot(kind="bar", ax=ax, color=colors[:len(seg_counts)])
ax.set_ylabel("Number of Customers")
ax.set_xlabel("")
ax.set_title("RFM Segment Distribution (M-aware labels)")
ax.tick_params(axis="x", rotation=35)
for i, (seg, count) in enumerate(seg_counts.items()):
    ax.text(i, count + 10, str(count), ha="center", fontsize=9)
fig.tight_layout()
savefig(fig, "rfm_distribution.png")
plt.show()

# %% [markdown]
# ## 4. K-Means on Nine Behavioral Features
#
# **The Golden Rule: cluster on behavior, profile with demographics.**
# All nine features in `BASE_COLS` are behavioral — revenue, frequency,
# recency, basket size, breadth, discount sensitivity. Demographics
# (`AGE_DESC`, `INCOME_DESC`, `HOUSEHOLD_SIZE_DESC`) only enter *after* the
# clustering, in the profiling section.

# %%
step("Scale features for clustering")

X_raw = customers[BASE_COLS].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"Base variables ({len(BASE_COLS)}): {BASE_COLS}")
print(f"Shape: {X_scaled.shape}")
print(f"\nScaled means (should be ~0):  {X_scaled.mean(axis=0).round(3)}")
print(f"Scaled stds  (should be ~1):  {X_scaled.std(axis=0).round(3)}")

# %% [markdown]
# We search K = 2..10 with two indicators:
#
# - **Elbow method**: plot within-cluster inertia and look for the "knee".
# - **Silhouette score**: higher is better.
#
# The statistical pick is rarely the final answer. After we see the metrics,
# we look at cluster sizes and demand each segment cover **at least 10%** of
# the base. That filters out the "mathematically clean but operationally
# useless" splits the chapter warns about.

# %%
step("K-Means: elbow + silhouette search")

km_results = []
for k in CONFIG["k_range"]:
    km = KMeans(n_clusters=k, n_init=10, random_state=CONFIG["random_state"])
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    km_results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    print(f"  K={k:2d}  |  Inertia={km.inertia_:>10,.0f}  |  Silhouette={sil:.4f}")

km_df = pd.DataFrame(km_results)

# %%
step("Elbow and silhouette plots")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(km_df["k"], km_df["inertia"], "o-", color=ACCENT_BLUE, linewidth=2)
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia (within-cluster SS)")
ax.set_title("Elbow Method")
ax.set_xticks(list(CONFIG["k_range"]))

ax = axes[1]
ax.plot(km_df["k"], km_df["silhouette"], "o-", color=ACCENT_GREEN, linewidth=2)
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Average Silhouette Score")
ax.set_title("Silhouette Score")
ax.set_xticks(list(CONFIG["k_range"]))

best_k = int(km_df.loc[km_df["silhouette"].idxmax(), "k"])
best_sil = float(km_df["silhouette"].max())
ax.annotate(f"Best silhouette: K={best_k}", xy=(best_k, best_sil),
            xytext=(best_k + 1, best_sil),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red")

fig.tight_layout()
savefig(fig, "elbow_silhouette.png")
plt.show()

print(f"Silhouette favors K={best_k} (score={best_sil:.4f}).")
print(f"Business-judgment choice: K={CONFIG['final_k']} (matches qmd caption).")

# %%
step("Fit final K-Means and validate cluster sizes")

final_k = CONFIG["final_k"]
km_final = KMeans(n_clusters=final_k, n_init=10, random_state=CONFIG["random_state"])
customers["km_segment"] = km_final.fit_predict(X_scaled)

sizes = customers["km_segment"].value_counts(normalize=True).sort_index()
print(f"Final K-Means: K={final_k}")
print(f"\nCluster shares:")
for seg, share in sizes.items():
    flag = "" if share >= CONFIG["min_cluster_share"] else "  ⚠ < 10% floor"
    print(f"  Segment {seg}: {share:.1%}{flag}")
if (sizes < CONFIG["min_cluster_share"]).any():
    print("\n⚠ At least one cluster falls below the 10% floor. "
          "Re-evaluate K or feature set before activating.")

# %% [markdown]
# ### Dendrogram — visual intuition for K
#
# Agglomerative clustering on a 500-household sample shows the merging
# hierarchy. Cutting the tree at different heights yields different K. We
# overlay the K=3 cut so the dendrogram matches the K-Means choice above —
# this is *visual intuition*, not an independent method.

# %%
step("Dendrogram (intuition for K=3)")

from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram

np.random.seed(CONFIG["random_state"])
sample_idx = np.random.choice(len(X_scaled), size=500, replace=False)
X_sample = X_scaled[sample_idx]

Z = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(14, 6))
scipy_dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    ax=ax,
    color_threshold=Z[-3, 2],   # color by K=3 cut
)
ax.set_xlabel("Cluster Size (or Sample Index)")
ax.set_ylabel("Distance (Ward)")
ax.set_title("Agglomerative Clustering Dendrogram (500-household sample)")
ax.axhline(y=Z[-3, 2], color="red", linestyle="--", alpha=0.5,
           label=f"Cut → K={final_k}")
ax.legend()
fig.tight_layout()
savefig(fig, "dendrogram.png")
plt.show()

customers.to_csv(_TABLES / "trad_cluster_assignments.csv", index=False)

# %% [markdown]
# ## 5. Segment Profiling
#
# Now we answer **who is in each segment?**
#
# Step 1: Profile base variables (what they *do*) — heatmap of cluster means.
# Step 2: Profile descriptor variables (who they *are*) — demographic cross-tabs.

# %%
step("Segment profile heatmap (base variables)")

profile = customers.groupby("km_segment")[BASE_COLS].mean()

pop_mean = customers[BASE_COLS].mean()
pop_std = customers[BASE_COLS].std()
profile_z = (profile - pop_mean) / pop_std


def _fmt(val):
    """Compact value formatter that handles the mixed scales in BASE_COLS."""
    if abs(val) >= 100:
        return f"{val:,.0f}"
    if abs(val) >= 1:
        return f"{val:.1f}"
    return f"{val:.2f}"


fig, ax = plt.subplots(figsize=(12, max(3, final_k * 0.8 + 1)))
im = ax.imshow(profile_z.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)

ax.set_xticks(range(len(BASE_COLS)))
ax.set_xticklabels(BASE_COLS, rotation=30, ha="right")
ax.set_yticks(range(final_k))
ax.set_yticklabels([f"Segment {i}" for i in range(final_k)])

for i in range(final_k):
    for j in range(len(BASE_COLS)):
        val = profile.iloc[i, j]
        ax.text(j, i, _fmt(val), ha="center", va="center", fontsize=9,
                color="white" if abs(profile_z.iloc[i, j]) > 1 else "black")

fig.colorbar(im, ax=ax, label="Z-score vs. population mean", shrink=0.8)
ax.set_title("K-Means Segment Profiles — 9 Behavioral Features")
fig.tight_layout()
savefig(fig, "segment_heatmap.png")
plt.show()

# %% [markdown]
# **Naming the segments.** We map K-Means clusters to operating-friendly
# names from the heatmap above. The exact mapping depends on which cluster
# came out on top of which axis when the notebook executed — adjust the
# labels by reading the heatmap (highest z-scores point you at the
# distinguishing feature).

# %%
step("Map K-Means segments to action tiers")

# Rank segments by mean revenue (post-fit) and assign tiers by rank.
# This keeps the labels stable across re-runs even if KMeans relabels.
seg_rev = (
    profile["total_revenue"]
    .sort_values(ascending=False)
    .index.tolist()
)
tier_labels = ["VIP retention", "Growth: upsell", "Low-cost nurture"]
# Pad if K > 3 (defensive)
while len(tier_labels) < final_k:
    tier_labels.append(f"Behavioral tier {len(tier_labels)+1}")

action_map_km = {seg: tier_labels[i] for i, seg in enumerate(seg_rev)}
customers["km_action_tier"] = customers["km_segment"].map(action_map_km)

print("K-Means → action tier mapping (ranked by mean revenue):")
for seg, tier in action_map_km.items():
    n = (customers["km_segment"] == seg).sum()
    print(f"  Segment {seg} (n={n:,}): {tier}")

# %%
step("Descriptor variable cross-tabs (demographics)")

cust_demo = customers.merge(demographics, on="household_key", how="inner")
print(f"Households with demographics: {len(cust_demo):,} / {len(customers):,}")

# %% [markdown]
# > **Note:** Only ~32% of households have demographics; the profiling tables
# > below cover this subset only. The K-Means assignments themselves use the
# > behavioral features available for all 2,500 households.

# %%
desc_cols = ["AGE_DESC", "INCOME_DESC", "HOUSEHOLD_SIZE_DESC"]

for col in desc_cols:
    ct = pd.crosstab(cust_demo["km_segment"], cust_demo[col], normalize="index")
    print(f"\n{'─'*60}")
    print(f"  {col} (row % within each segment)")
    print(f"{'─'*60}")
    display(ct.round(3))

# %% [markdown]
# ## 6. Decile Migration Matrix — Temporal Stability
#
# A segmentation is only useful if it persists over time. We split the 102
# weeks into two halves, build revenue-based quintile-pairs independently
# in each half, and check the migration matrix. Strong diagonals mean the
# value tiers are stable; heavy off-diagonal mass means customers shuffle
# unpredictably.

# %%
step("Decile migration matrix")

mid_week = CONFIG["max_week"] // 2  # week 51

txn_h1 = txn[txn["WEEK_NO"] <= mid_week]
txn_h2 = txn[txn["WEEK_NO"] > mid_week]

rev_h1 = txn_h1.groupby("household_key")["SALES_VALUE"].sum()
rev_h2 = txn_h2.groupby("household_key")["SALES_VALUE"].sum()

common = rev_h1.index.intersection(rev_h2.index)
rev_h1 = rev_h1.loc[common]
rev_h2 = rev_h2.loc[common]

dec_h1 = pd.qcut(rev_h1.rank(method="first", ascending=False), q=5,
                 labels=["D1-D2", "D3-D4", "D5-D6", "D7-D8", "D9-D10"])
dec_h2 = pd.qcut(rev_h2.rank(method="first", ascending=False), q=5,
                 labels=["D1-D2", "D3-D4", "D5-D6", "D7-D8", "D9-D10"])

migration = pd.crosstab(dec_h1, dec_h2, normalize="index")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(migration.values, cmap="YlOrRd", vmin=0, vmax=0.6)

labels = migration.index.tolist()
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Half 2 (Weeks 52–102)")
ax.set_ylabel("Half 1 (Weeks 1–51)")
ax.set_title("Decile Migration Matrix\n(row % → where did each group end up?)")

for i in range(len(labels)):
    for j in range(len(labels)):
        val = migration.iloc[i, j]
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                fontsize=10, color="white" if val > 0.3 else "black")

fig.colorbar(im, ax=ax, label="Proportion", shrink=0.8)
fig.tight_layout()
savefig(fig, "migration_matrix.png")
plt.show()

print(f"Customers in both halves: {len(common):,}")
print(f"Diagonal retention rates:")
for i, lbl in enumerate(labels):
    print(f"  {lbl}: {migration.iloc[i, i]:.0%}")

# %% [markdown]
# ## 7. CRM-Ready Export
#
# Segments only matter if they reach the activation systems. We persist a
# single table joining the two complementary lenses we built — decile tier
# (value) and K-Means segment (behavior) — so it can be loaded into the
# CRM/ad platforms.

# %%
step("Export segment assignments")

crm_export = customers[[
    "household_key", "decile", "decile_action_tier",
    "km_segment", "km_action_tier",
]].copy()
crm_export = crm_export.sort_values("household_key")

display(crm_export.head(10))

# Repo-canonical path for cross-section data passed to sec4.2 (CLV chapter).
# Resolved by msbook so this works from any cwd or notebook subdirectory.
from msbook.paths import chapter_processed

_seg_out = chapter_processed(part="4", chapter="sec4.1-segmentation") / "segment_assignments.csv"
crm_export.to_csv(_seg_out, index=False)
print(f"\nWrote {len(crm_export):,} rows to {_seg_out.relative_to(_seg_out.parents[3])}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Decile analysis** reveals the Pareto pattern: a small fraction of
#    customers drive a disproportionate share of revenue.
# 2. **RFM with M-aware labels** stays interpretable while keeping the
#    "Champions" label honest — high-value labels gate on monetary, not
#    just recency and frequency.
# 3. **Cluster on behavior, profile with demographics.** All nine features
#    fed into K-Means are behavioral; demographics enter only afterwards.
# 4. **K is part statistics, part judgment.** Silhouette and elbow point at
#    a region; the 10% floor and "can marketing describe a different action
#    for each?" decide the number.
# 5. **Migration analysis tests temporal stability** — segments that
#    shuffle unpredictably between periods are not actionable.
# 6. **Operationalize or die.** Ship a `segment_assignments.csv` (or
#    equivalent table) into the CRM and ad platforms; a segmentation that
#    never reaches activation is a slide deck, not an operating artifact.
