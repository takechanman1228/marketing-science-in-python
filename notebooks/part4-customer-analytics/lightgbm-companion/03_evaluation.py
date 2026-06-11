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
# # Purchase-Incidence + Amount ML CLV — Step 3: Evaluation & Storytelling
#
# This notebook evaluates the decomposed ML CLV pipeline and produces:
#
# 1. **Sub-model evaluation**: Purchase AUC-ROC + calibration, Spend MAE/RMSE
# 2. **CLV evaluation**: holdout CLV vs. actual, ranking metrics
# 3. **SHAP analysis**: What drives purchase vs. spend (side-by-side)
# 4. **Decomposition diagnostics**: P(buy) vs. E[spend|buy] scatter
# 5. **Business outputs**: Customer value tiers, model comparison table

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)

from msbook.paths import chapter_images, chapter_artifacts

# ── Configuration ────────────────────────────────────────────────────────────
ACCENT_BLUE = "#2171b5"
ACCENT_ORANGE = "#e6550d"
ACCENT_GREEN = "#31a354"
ACCENT_PURPLE = "#756bb1"
ACCENT_GREY = "#969696"

pd.set_option("display.float_format", "{:.2f}".format)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

_FIG_DIR = chapter_images(part="4", chapter="sec4.2")
_TABLES  = chapter_artifacts(part="4", chapter="sec4.2-clv") / "tables"
_MODELS  = chapter_artifacts(part="4", chapter="sec4.2-clv") / "models"
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
# ## 1. Load Artifacts

# %%
step("Load artifacts")

predictions = pd.read_parquet(_TABLES / "ml_clv_predictions.parquet")
features = pd.read_parquet(_TABLES / "ml_clv_features.parquet")
panel = pd.read_parquet(_TABLES / "ml_clv_panel.parquet")

model_purch_rfm = joblib.load(_MODELS / "model_purch_rfm.joblib")
model_purch_full = joblib.load(_MODELS / "model_purch_full.joblib")
model_spend_rfm = joblib.load(_MODELS / "model_spend_rfm.joblib")
model_spend_full = joblib.load(_MODELS / "model_spend_full.joblib")

with open(_TABLES / "feature_cols.json") as f:
    cols_meta = json.load(f)
rfm_cols = cols_meta["rfm_cols"]
full_cols = cols_meta["full_cols"]

with open(_TABLES / "constants.json") as f:
    CONSTANTS = json.load(f)
N_HOLDOUT_PERIODS = CONSTANTS["N_HOLDOUT_PERIODS"]
DISCOUNT_RATE = CONSTANTS["DISCOUNT_RATE"]
CLV_HORIZON = CONSTANTS["CLV_HORIZON"]

# Test set
test = predictions[predictions["split"] == "test"].copy()
hh_test = test["household_key"].values

# Panel test rows
panel_test = panel[panel["household_key"].isin(hh_test)].copy()

print(f"Test customers: {len(test):,}")
print(f"Test panel rows: {len(panel_test):,}")
print(f"Models loaded: purchase (RFM, Full), spend (RFM, Full)")

# %% [markdown]
# ## 2. Sub-Model Evaluation — Purchase Classifier

# %%
step("Purchase model evaluation — AUC-ROC")

# Panel-level evaluation: predict P(buy) for each test customer-period row
X_panel_test_rfm = features.loc[panel_test["household_key"].values, rfm_cols].reset_index(drop=True)
X_panel_test_full = features.loc[panel_test["household_key"].values, full_cols].reset_index(drop=True)
y_panel_test = panel_test["purchased"].values

p_panel_rfm = model_purch_rfm.predict_proba(X_panel_test_rfm)[:, 1]
p_panel_full = model_purch_full.predict_proba(X_panel_test_full)[:, 1]

auc_rfm = roc_auc_score(y_panel_test, p_panel_rfm)
auc_full = roc_auc_score(y_panel_test, p_panel_full)

print(f"Purchase AUC-ROC (RFM):  {auc_rfm:.4f}")
print(f"Purchase AUC-ROC (Full): {auc_full:.4f}")

# %%
step("Purchase model — calibration plot")

def calibration_plot(y_true, y_pred, n_bins=10, label="", color="blue", ax=None):
    """Binned calibration: predicted P(buy) vs observed buy-rate."""
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    cal = df.groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(cal["mean_pred"], cal["mean_actual"], "o-", color=color, label=label)
    return ax

fig, ax = plt.subplots(figsize=(7, 6))
calibration_plot(y_panel_test, p_panel_rfm, label=f"RFM (AUC={auc_rfm:.3f})", color=ACCENT_GREY, ax=ax)
calibration_plot(y_panel_test, p_panel_full, label=f"Full (AUC={auc_full:.3f})", color=ACCENT_BLUE, ax=ax)
lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "--", color="red", alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Predicted P(buy)")
ax.set_ylabel("Observed Buy Rate")
ax.set_title("Purchase Model Calibration")
ax.legend()

fig.tight_layout()
savefig(fig, "ml_clv_purchase_calibration.png")
plt.show()

# %% [markdown]
# ## 3. Sub-Model Evaluation — Spend Regressor

# %%
step("Spend model evaluation (purchase-rows only)")

panel_test_buy = panel_test[panel_test["purchased"] == 1].copy()
X_buy_test_rfm = features.loc[panel_test_buy["household_key"].values, rfm_cols].reset_index(drop=True)
X_buy_test_full = features.loc[panel_test_buy["household_key"].values, full_cols].reset_index(drop=True)
y_spend_test = panel_test_buy["period_spend"].values

spend_pred_rfm = model_spend_rfm.predict(X_buy_test_rfm)
spend_pred_full = model_spend_full.predict(X_buy_test_full)

for label, preds in [("RFM", spend_pred_rfm), ("Full", spend_pred_full)]:
    mae = mean_absolute_error(y_spend_test, preds)
    rmse = root_mean_squared_error(y_spend_test, preds)
    print(f"Spend {label:6s}  MAE: ${mae:>8,.2f}  |  RMSE: ${rmse:>10,.2f}")

# %% [markdown]
# ## 4. CLV Evaluation — Holdout CLV vs. Actual Spend

# %%
step("CLV holdout evaluation")

y_actual = test["y_holdout_spend"].values

for label, clv_col, clv_10yr_col in [
    ("RFM", "clv_holdout_rfm", "clv_10yr_rfm"),
    ("Full", "clv_holdout_full", "clv_10yr_full"),
]:
    y_pred = test[clv_col].values
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = root_mean_squared_error(y_actual, y_pred)
    rho, pval = spearmanr(y_actual, y_pred)
    # Also check Spearman of 10-year CLV vs holdout spend (rank-preservation)
    rho_10yr, _ = spearmanr(y_actual, test[clv_10yr_col].values)
    print(f"{label:6s}  MAE: ${mae:>8,.2f}  |  RMSE: ${rmse:>10,.2f}  |  Spearman: {rho:.4f}  |  10yr rank corr: {rho_10yr:.4f}")

# %% [markdown]
# ## 5. Decile Lift Chart

# %%
step("Decile lift chart")

def compute_decile_lift(y_true, y_pred, label):
    """Assign predicted-CLV deciles and compute actual-spend lift per decile."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df["decile"] = pd.qcut(
        df["predicted"].rank(method="first", ascending=False),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    summary = df.groupby("decile", observed=True)["actual"].agg(["mean", "sum", "count"])
    overall_mean = df["actual"].mean()
    summary["lift"] = summary["mean"] / overall_mean
    summary["model"] = label
    return summary

lift_rfm = compute_decile_lift(y_actual, test["clv_holdout_rfm"].values, "RFM-only")
lift_full = compute_decile_lift(y_actual, test["clv_holdout_full"].values, "Full behavioral")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(10)
width = 0.35
ax.bar(x - width/2, lift_rfm["lift"].values, width, label="RFM-only (5 features)",
       color=ACCENT_GREY, edgecolor="white")
ax.bar(x + width/2, lift_full["lift"].values, width, label="Full behavioral (25 features)",
       color=ACCENT_BLUE, edgecolor="white")
ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Average = 1.0")
ax.set_xticks(x)
ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
ax.set_xlabel("Predicted CLV Decile (D1 = highest predicted)")
ax.set_ylabel("Lift (actual spend / overall average)")
ax.set_title("Decile Lift Chart — RFM-only vs. Full Behavioral Model")
ax.legend()

# Annotate top decile lifts
ax.annotate(f"{lift_rfm['lift'].iloc[0]:.2f}x",
            xy=(x[0] - width/2, lift_rfm["lift"].iloc[0]),
            ha="center", va="bottom", fontsize=9, color=ACCENT_GREY)
ax.annotate(f"{lift_full['lift'].iloc[0]:.2f}x",
            xy=(x[0] + width/2, lift_full["lift"].iloc[0]),
            ha="center", va="bottom", fontsize=9, color=ACCENT_BLUE)

fig.tight_layout()
savefig(fig, "ml_clv_decile_lift.png")
plt.show()

print("Top-decile lift:")
print(f"  RFM-only: {lift_rfm['lift'].iloc[0]:.2f}x")
print(f"  Full:     {lift_full['lift'].iloc[0]:.2f}x")

# %% [markdown]
# ## 6. Cumulative Gains Curve

# %%
step("Cumulative gains curve")

def cumulative_gains(y_true, y_pred):
    """Compute cumulative % of actual spend captured, sorted by predicted CLV desc."""
    order = np.argsort(-y_pred)
    sorted_actual = y_true[order]
    cumsum = np.cumsum(sorted_actual)
    total = sorted_actual.sum()
    pct_customers = np.arange(1, len(y_true) + 1) / len(y_true)
    pct_captured = cumsum / total
    return pct_customers, pct_captured

pct_cust_rfm, pct_cap_rfm = cumulative_gains(y_actual, test["clv_holdout_rfm"].values)
pct_cust_full, pct_cap_full = cumulative_gains(y_actual, test["clv_holdout_full"].values)
pct_cust_perfect, pct_cap_perfect = cumulative_gains(y_actual, y_actual)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pct_cust_perfect, pct_cap_perfect, "-", color=ACCENT_GREEN, linewidth=1.5,
        alpha=0.5, label="Perfect model")
ax.plot(pct_cust_full, pct_cap_full, "-", color=ACCENT_BLUE, linewidth=2,
        label="Full behavioral")
ax.plot(pct_cust_rfm, pct_cap_rfm, "-", color=ACCENT_GREY, linewidth=2,
        label="RFM-only")
ax.plot([0, 1], [0, 1], "--", color="red", linewidth=0.8, alpha=0.5, label="Random")
ax.set_xlabel("Cumulative % of Customers (ranked by predicted CLV)")
ax.set_ylabel("Cumulative % of Actual Revenue Captured")
ax.set_title("Cumulative Gains Curve")
ax.legend(loc="lower right")

fig.tight_layout()
savefig(fig, "ml_clv_cumulative_gains.png")
plt.show()

for pct in [10, 20, 30]:
    idx = int(len(y_actual) * pct / 100)
    cap_rfm = pct_cap_rfm[idx - 1]
    cap_full = pct_cap_full[idx - 1]
    print(f"Top {pct}% capture:  RFM-only={cap_rfm:.1%}  Full={cap_full:.1%}")

# %% [markdown]
# ## 7. SHAP Analysis — Purchase vs. Spend
#
# The key narrative: **what predicts whether a customer buys** can differ from
# **what predicts how much they spend when they buy**. This decomposition
# gives us richer insight than a single regression model.

# %%
step("SHAP — Purchase model (Full)")

X_test_full = features.loc[hh_test, full_cols]

explainer_purch = shap.TreeExplainer(model_purch_full)
shap_values_purch = explainer_purch.shap_values(X_test_full)
# For binary classifier, shap_values may be a list [class_0, class_1]
if isinstance(shap_values_purch, list):
    shap_values_purch = shap_values_purch[1]

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_purch, X_test_full, show=False, max_display=15)
plt.title("SHAP — What Drives Purchase Probability?", fontsize=13, pad=15)
plt.tight_layout()
savefig(plt.gcf(), "ml_clv_shap_purchase.png")
plt.show()

# %%
step("SHAP — Spend model (Full)")

explainer_spend = shap.TreeExplainer(model_spend_full)
shap_values_spend = explainer_spend.shap_values(X_test_full)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_spend, X_test_full, show=False, max_display=15)
plt.title("SHAP — What Drives Spend Given Purchase?", fontsize=13, pad=15)
plt.tight_layout()
savefig(plt.gcf(), "ml_clv_shap_spend.png")
plt.show()

# %% [markdown]
# ### Side-by-Side Feature Importance Comparison

# %%
step("SHAP feature importance — Purchase vs. Spend")

shap_imp_purch = pd.Series(
    np.abs(shap_values_purch).mean(axis=0), index=full_cols
).sort_values(ascending=False)

shap_imp_spend = pd.Series(
    np.abs(shap_values_spend).mean(axis=0), index=full_cols
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Purchase model importance
ax = axes[0]
shap_imp_purch.head(15).sort_values().plot(kind="barh", ax=ax, color=ACCENT_BLUE)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Purchase Model: What Drives P(buy)?")

# Right: Spend model importance
ax = axes[1]
shap_imp_spend.head(15).sort_values().plot(kind="barh", ax=ax, color=ACCENT_ORANGE)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Spend Model: What Drives E[spend|buy]?")

fig.suptitle("Feature Importance — Purchase vs. Spend (these should differ!)", fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "ml_clv_shap_importance_comparison.png")
plt.show()

# Show rank differences
print("Top 10 features by SHAP importance:")
print(f"{'Rank':<6}{'Purchase Model':<30}{'Spend Model':<30}")
print("-" * 66)
for i in range(10):
    p_feat = shap_imp_purch.index[i]
    s_feat = shap_imp_spend.index[i]
    p_mark = " *" if p_feat not in rfm_cols else ""
    s_mark = " *" if s_feat not in rfm_cols else ""
    print(f"{i+1:<6}{p_feat + p_mark:<30}{s_feat + s_mark:<30}")
print("\n  * = feature NOT available to RFM-only model")

# %% [markdown]
# ### SHAP Dependence Plots

# %%
step("SHAP dependence plots — top non-RFM features")

# Pick top non-RFM features from the purchase model
non_rfm_purch = [f for f in shap_imp_purch.index if f not in rfm_cols][:3]

fig, axes = plt.subplots(1, len(non_rfm_purch), figsize=(5 * len(non_rfm_purch), 5))
if len(non_rfm_purch) == 1:
    axes = [axes]

for ax, feat in zip(axes, non_rfm_purch):
    feat_idx = full_cols.index(feat)
    shap.dependence_plot(
        feat_idx, shap_values_purch, X_test_full,
        ax=ax, show=False,
    )
    ax.set_title(f"Purchase: {feat}", fontsize=11)

fig.suptitle("SHAP Dependence — Key Non-RFM Features (Purchase Model)", fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "ml_clv_shap_dependence.png")
plt.show()

# %% [markdown]
# ## 8. Decomposition Diagnostics
#
# Scatter P(buy) vs. E[spend|buy] to visualize the two components of CLV.
# Different clusters reveal distinct customer segments:
# - High P(buy) + High E[spend|buy] = high-value loyalists
# - High P(buy) + Low E[spend|buy] = frequent small-basket shoppers
# - Low P(buy) + High E[spend|buy] = rare big spenders

# %%
step("Decomposition diagnostics — P(buy) vs. E[spend|buy]")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CLV quartile tiers for coloring
test["clv_tier"] = pd.qcut(
    test["clv_10yr_full"].rank(method="first", ascending=False),
    q=4,
    labels=["Platinum", "Gold", "Silver", "Bronze"],
)
tier_colors = {"Platinum": ACCENT_BLUE, "Gold": ACCENT_GREEN, "Silver": ACCENT_ORANGE, "Bronze": ACCENT_GREY}

# Left: Scatter P(buy) vs E[spend|buy], colored by CLV tier
ax = axes[0]
for tier in ["Bronze", "Silver", "Gold", "Platinum"]:
    mask = test["clv_tier"] == tier
    ax.scatter(
        test.loc[mask, "p_buy_full"],
        test.loc[mask, "e_spend_full"],
        s=15, alpha=0.5, color=tier_colors[tier], label=tier,
    )
ax.set_xlabel("P(buy in any period)")
ax.set_ylabel("E[spend | buy] ($)")
ax.set_title("CLV Decomposition: Purchase Rate vs. Conditional Spend")
ax.legend(title="CLV Tier")

# Right: Histogram of P(buy)
ax = axes[1]
ax.hist(test["p_buy_full"], bins=30, color=ACCENT_BLUE, edgecolor="white", alpha=0.8)
ax.set_xlabel("P(buy in any period)")
ax.set_ylabel("Number of Customers")
ax.set_title("Distribution of Predicted Purchase Probability")

fig.tight_layout()
savefig(fig, "ml_clv_decomposition_scatter.png")
plt.show()

print("P(buy) summary (test, Full model):")
print(test["p_buy_full"].describe().round(4))

# %% [markdown]
# ## 9. Customer Value Tiers

# %%
step("Customer value tiers")

tier_profile = (
    test.groupby("clv_tier", observed=True)
    .agg(
        n_customers=("household_key", "count"),
        avg_predicted_clv=("clv_10yr_full", "mean"),
        avg_actual_spend=("y_holdout_spend", "mean"),
        total_actual_spend=("y_holdout_spend", "sum"),
        avg_p_buy=("p_buy_full", "mean"),
        avg_e_spend=("e_spend_full", "mean"),
        pct_zero_spend=("y_holdout_spend", lambda x: (x == 0).mean()),
    )
)
tier_profile["revenue_share"] = (
    tier_profile["total_actual_spend"] / tier_profile["total_actual_spend"].sum()
)

print("Customer Value Tier Profiles:")
print(tier_profile.to_string(float_format=lambda x: f"{x:,.2f}"))

tier_profile.to_csv(_TABLES / "ml_clv_tier_profiles.csv")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: avg actual spend by tier
ax = axes[0]
colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_GREY]
tier_profile["avg_actual_spend"].plot(kind="bar", ax=ax, color=colors)
ax.set_ylabel("Average Actual Holdout Spend ($)")
ax.set_xlabel("")
ax.set_title("Actual Spend by Predicted Value Tier")
ax.tick_params(axis="x", rotation=0)
for i, val in enumerate(tier_profile["avg_actual_spend"]):
    ax.text(i, val + 20, f"${val:,.0f}", ha="center", fontsize=9)

# Right: revenue share pie
ax = axes[1]
ax.pie(
    tier_profile["revenue_share"],
    labels=tier_profile.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
)
ax.set_title("Revenue Share by Predicted Tier")

fig.tight_layout()
savefig(fig, "ml_clv_value_tiers.png")
plt.show()

# %% [markdown]
# ## 10. Model Comparison Summary

# %%
step("Model comparison summary")

def compute_all_metrics(y_true, y_pred, label, n_features):
    """Compute point and ranking metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    pct_cust, pct_cap = cumulative_gains(y_true, y_pred)
    top10 = pct_cap[int(len(y_true) * 0.1) - 1]
    top20 = pct_cap[int(len(y_true) * 0.2) - 1]
    # Decile 1 lift
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df["decile"] = pd.qcut(
        df["predicted"].rank(method="first", ascending=False), q=10,
        labels=range(1, 11),
    )
    d1_lift = df[df["decile"] == 1]["actual"].mean() / df["actual"].mean()
    return {
        "Model": label,
        "Features": n_features,
        "MAE ($)": f"{mae:,.2f}",
        "RMSE ($)": f"{rmse:,.2f}",
        "Spearman rho": f"{rho:.4f}",
        "Top-10% Capture": f"{top10:.1%}",
        "Top-20% Capture": f"{top20:.1%}",
        "D1 Lift": f"{d1_lift:.2f}x",
    }

comparison = pd.DataFrame([
    compute_all_metrics(y_actual, test["clv_holdout_rfm"].values, "RFM-only (Purchase+Spend)", len(rfm_cols)),
    compute_all_metrics(y_actual, test["clv_holdout_full"].values, "Full behavioral (Purchase+Spend)", len(full_cols)),
])
comparison = comparison.set_index("Model")

print("Model Comparison (holdout CLV vs. actual spend):")
print(comparison.to_string())
print()

comparison.to_csv(_TABLES / "ml_clv_model_comparison.csv")

# Headline stats
print("HEADLINE METRICS:")
for col in ["clv_holdout_rfm", "clv_holdout_full"]:
    _, pct_cap = cumulative_gains(y_actual, test[col].values)
    top10 = pct_cap[int(len(y_actual) * 0.1) - 1]
    label = "RFM" if "rfm" in col else "Full"
    print(f"  {label}: top 10% by predicted CLV captures {top10:.0%} of actual revenue")

# 10-year CLV range check
print(f"\n10-year CLV range (Full model, test set):")
print(f"  Mean:   ${test['clv_10yr_full'].mean():,.2f}")
print(f"  Median: ${test['clv_10yr_full'].median():,.2f}")
print(f"  Min:    ${test['clv_10yr_full'].min():,.2f}")
print(f"  Max:    ${test['clv_10yr_full'].max():,.2f}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Decomposed CLV mirrors BTYD**: By separating purchase incidence from
#    conditional spend, the ML model provides the same interpretable structure
#    as BG/NBD + Gamma-Gamma — but can leverage richer feature sets.
#
# 2. **Different drivers for purchase vs. spend**: SHAP reveals that features
#    predicting *whether* a customer buys differ from those predicting *how much*
#    they spend. This gives stakeholders two distinct levers for intervention.
#
# 3. **ML adds value beyond RFM**: The full behavioral model improves on RFM-only
#    by leveraging discount behavior, product diversity, purchase trends, and
#    campaign responsiveness.
#
# 4. **Limitation — constant purchase rate**: Unlike BTYD's time-varying P(alive),
#    the ML model assumes a constant P(buy) per customer. This works well for
#    stable shoppers but may overestimate CLV for customers showing decline.
#    The discount factor provides partial mitigation.
#
# 5. **Value tiers work**: Platinum customers (top 25% by predicted CLV)
#    concentrate a disproportionate share of actual revenue, validating the
#    model for differential treatment strategies.
