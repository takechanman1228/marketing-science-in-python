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
# # Purchase-Incidence + Amount ML CLV — Step 2: Model Training
#
# We train **four LightGBM models** mirroring the BTYD decomposition:
#
# | Model | Type | Features | Training rows |
# |-------|------|----------|---------------|
# | Purchase (RFM) | Classifier | 5 RFM | All panel rows (train customers) |
# | Purchase (Full) | Classifier | 25 A-F | All panel rows (train customers) |
# | Spend (RFM) | Regressor | 5 RFM | Purchase-rows only (train customers) |
# | Spend (Full) | Regressor | 25 A-F | Purchase-rows only (train customers) |
#
# **CLV calculation**: `CLV = Σ_{t=1}^{T} P(buy) × E[spend|buy] / (1+r)^t`
#
# This parallels BTYD's `E[purchases_in_month_t] × E[spend_per_purchase] / (1+r)^t`.
#
# **Limitation**: The ML model predicts a constant monthly purchase rate per
# customer (no temporal decay), while BTYD models P(alive at t) which decreases
# over time. This means ML CLV is most accurate for customers with stable
# purchasing patterns and may overestimate for customers showing engagement
# decline. The discount factor partially mitigates this.

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    GroupKFold,
)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from msbook.paths import chapter_images, chapter_artifacts

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "test_size": 0.2,
    "val_size": 0.2,       # validation split within train (for early stopping)
    "cv_folds": 5,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": 42,
}

pd.set_option("display.float_format", "{:.2f}".format)

_FIG_DIR = chapter_images(part="4", chapter="sec4.2")
_TABLES  = chapter_artifacts(part="4", chapter="sec4.2-clv") / "tables"
_MODELS  = chapter_artifacts(part="4", chapter="sec4.2-clv") / "models"
_TABLES.mkdir(parents=True, exist_ok=True)
_MODELS.mkdir(parents=True, exist_ok=True)


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
# ## 1. Load Data

# %%
step("Load feature table, panel, and constants")

features = pd.read_parquet(_TABLES / "ml_clv_features.parquet")
panel = pd.read_parquet(_TABLES / "ml_clv_panel.parquet")

with open(_TABLES / "feature_groups.json") as f:
    FEATURE_GROUPS = json.load(f)
with open(_TABLES / "constants.json") as f:
    CONSTANTS = json.load(f)

N_HOLDOUT_PERIODS = CONSTANTS["N_HOLDOUT_PERIODS"]
DISCOUNT_RATE = CONSTANTS["DISCOUNT_RATE"]
CLV_HORIZON = CONSTANTS["CLV_HORIZON"]

# RFM-only features (Group A)
rfm_cols = FEATURE_GROUPS["A_rfm"]

# Full behavioral features (Groups A-F, no demographics)
full_cols = []
for group in ["A_rfm", "B_behavioral", "C_discount", "D_product", "E_trends", "F_campaign"]:
    full_cols.extend(FEATURE_GROUPS[group])

print(f"Households: {len(features):,}")
print(f"Panel rows: {len(panel):,}")
print(f"RFM features ({len(rfm_cols)}): {rfm_cols}")
print(f"Full features ({len(full_cols)}): {full_cols}")
print(f"\nConstants: discount_rate={DISCOUNT_RATE}, horizon={CLV_HORIZON} months, holdout_periods={N_HOLDOUT_PERIODS}")

# %% [markdown]
# ## 2. Train / Test Split on Customers

# %%
step("Train/test split on customers (stratified by holdout purchase activity)")

# Stratify by whether customer purchased at all in holdout
has_purchase = (features["y_holdout_spend"] > 0).astype(int)

all_households = features.index.values
hh_train, hh_test = train_test_split(
    all_households,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=has_purchase,
)

print(f"Train customers: {len(hh_train):,}")
print(f"Test customers:  {len(hh_test):,}")

# Panel splits
panel_train = panel[panel["household_key"].isin(hh_train)].copy()
panel_test = panel[panel["household_key"].isin(hh_test)].copy()

print(f"\nTrain panel rows: {len(panel_train):,}  (purchase rate: {panel_train['purchased'].mean():.3f})")
print(f"Test panel rows:  {len(panel_test):,}  (purchase rate: {panel_test['purchased'].mean():.3f})")

# Purchase-only rows for spend model
panel_train_buy = panel_train[panel_train["purchased"] == 1].copy()
panel_test_buy = panel_test[panel_test["purchased"] == 1].copy()

print(f"\nTrain purchase-rows: {len(panel_train_buy):,}")
print(f"Test purchase-rows:  {len(panel_test_buy):,}")

# %% [markdown]
# ## 3. Prepare Feature Matrices
#
# Each panel row has the same features as its customer (calibration features
# are constant across periods). We merge features onto the panel.

# %%
step("Prepare feature matrices")

def make_panel_X(panel_df, feature_df, cols):
    """Merge customer features onto panel rows."""
    return feature_df.loc[panel_df["household_key"].values, cols].reset_index(drop=True)

# Purchase model: all panel rows
X_purch_train_rfm = make_panel_X(panel_train, features, rfm_cols)
X_purch_train_full = make_panel_X(panel_train, features, full_cols)
y_purch_train = panel_train["purchased"].values
groups_purch_train = panel_train["household_key"].values

# Spend model: purchase-rows only
X_spend_train_rfm = make_panel_X(panel_train_buy, features, rfm_cols)
X_spend_train_full = make_panel_X(panel_train_buy, features, full_cols)
y_spend_train = panel_train_buy["period_spend"].values
groups_spend_train = panel_train_buy["household_key"].values

print(f"Purchase model training: {len(y_purch_train):,} rows")
print(f"Spend model training:   {len(y_spend_train):,} rows")


# %% [markdown]
# ## 4. Hyperparameter Search — Purchase Models (Classifier)

# %%
step("Grid search — Purchase classifier (RFM)")

param_grid_clf = {
    "num_leaves": [15, 31],
    "min_child_samples": [10, 30],
    "learning_rate": [0.03, 0.1],
}

base_clf_params = {
    "n_estimators": CONFIG["n_estimators"],
    "random_state": CONFIG["random_state"],
    "verbose": -1,
}

# GroupKFold ensures no customer appears in both train and validation folds
gkf = GroupKFold(n_splits=CONFIG["cv_folds"])

grid_purch_rfm = GridSearchCV(
    lgb.LGBMClassifier(**base_clf_params),
    param_grid_clf,
    cv=gkf,
    scoring="roc_auc",
    n_jobs=-1,
    refit=False,
)
grid_purch_rfm.fit(X_purch_train_rfm, y_purch_train, groups=groups_purch_train)

print(f"Best params (Purchase RFM): {grid_purch_rfm.best_params_}")
print(f"Best CV AUC: {grid_purch_rfm.best_score_:.4f}")

# %%
step("Grid search — Purchase classifier (Full)")

grid_purch_full = GridSearchCV(
    lgb.LGBMClassifier(**base_clf_params),
    param_grid_clf,
    cv=gkf,
    scoring="roc_auc",
    n_jobs=-1,
    refit=False,
)
grid_purch_full.fit(X_purch_train_full, y_purch_train, groups=groups_purch_train)

print(f"Best params (Purchase Full): {grid_purch_full.best_params_}")
print(f"Best CV AUC: {grid_purch_full.best_score_:.4f}")

# %% [markdown]
# ## 5. Hyperparameter Search — Spend Models (Regressor)

# %%
step("Grid search — Spend regressor (RFM)")

param_grid_reg = {
    "num_leaves": [15, 31],
    "min_child_samples": [10, 30],
    "learning_rate": [0.03, 0.1],
}

base_reg_params = {
    "n_estimators": CONFIG["n_estimators"],
    "random_state": CONFIG["random_state"],
    "verbose": -1,
}

gkf_spend = GroupKFold(n_splits=CONFIG["cv_folds"])

grid_spend_rfm = GridSearchCV(
    lgb.LGBMRegressor(**base_reg_params),
    param_grid_reg,
    cv=gkf_spend,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=False,
)
grid_spend_rfm.fit(X_spend_train_rfm, y_spend_train, groups=groups_spend_train)

print(f"Best params (Spend RFM): {grid_spend_rfm.best_params_}")
print(f"Best CV MAE: ${-grid_spend_rfm.best_score_:,.2f}")

# %%
step("Grid search — Spend regressor (Full)")

grid_spend_full = GridSearchCV(
    lgb.LGBMRegressor(**base_reg_params),
    param_grid_reg,
    cv=gkf_spend,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=False,
)
grid_spend_full.fit(X_spend_train_full, y_spend_train, groups=groups_spend_train)

print(f"Best params (Spend Full): {grid_spend_full.best_params_}")
print(f"Best CV MAE: ${-grid_spend_full.best_score_:,.2f}")

# %% [markdown]
# ## 6. Train Final Models with Early Stopping
#
# We split train customers into train_inner (80%) + validation (20%) for
# early stopping. **The test set is NOT used for early stopping.**

# %%
step("Split train customers into inner-train + validation for early stopping")

hh_train_inner, hh_val = train_test_split(
    hh_train,
    test_size=CONFIG["val_size"],
    random_state=CONFIG["random_state"],
    stratify=has_purchase.loc[hh_train],
)

print(f"Inner-train customers: {len(hh_train_inner):,}")
print(f"Validation customers:  {len(hh_val):,}")

# Panel splits for inner-train and validation
panel_inner = panel_train[panel_train["household_key"].isin(hh_train_inner)]
panel_val = panel_train[panel_train["household_key"].isin(hh_val)]

panel_inner_buy = panel_inner[panel_inner["purchased"] == 1]
panel_val_buy = panel_val[panel_val["purchased"] == 1]

# %%
step("Train final Purchase models with early stopping")

def train_model(ModelClass, best_params, base_params, X_inner, y_inner, X_val, y_val, name):
    """Train a model with early stopping on validation set."""
    model = ModelClass(**base_params, **best_params)
    model.fit(
        X_inner, y_inner,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(CONFIG["early_stopping_rounds"]),
            lgb.log_evaluation(0),
        ],
    )
    print(f"  {name}: best_iteration={model.best_iteration_}")
    return model

# Purchase RFM
X_inner_rfm = make_panel_X(panel_inner, features, rfm_cols)
X_val_rfm = make_panel_X(panel_val, features, rfm_cols)

model_purch_rfm = train_model(
    lgb.LGBMClassifier, grid_purch_rfm.best_params_, base_clf_params,
    X_inner_rfm, panel_inner["purchased"].values,
    X_val_rfm, panel_val["purchased"].values,
    "Purchase RFM",
)

# Purchase Full
X_inner_full = make_panel_X(panel_inner, features, full_cols)
X_val_full = make_panel_X(panel_val, features, full_cols)

model_purch_full = train_model(
    lgb.LGBMClassifier, grid_purch_full.best_params_, base_clf_params,
    X_inner_full, panel_inner["purchased"].values,
    X_val_full, panel_val["purchased"].values,
    "Purchase Full",
)

# %%
step("Train final Spend models with early stopping")

# Spend RFM
X_inner_spend_rfm = make_panel_X(panel_inner_buy, features, rfm_cols)
X_val_spend_rfm = make_panel_X(panel_val_buy, features, rfm_cols)

model_spend_rfm = train_model(
    lgb.LGBMRegressor, grid_spend_rfm.best_params_, base_reg_params,
    X_inner_spend_rfm, panel_inner_buy["period_spend"].values,
    X_val_spend_rfm, panel_val_buy["period_spend"].values,
    "Spend RFM",
)

# Spend Full
X_inner_spend_full = make_panel_X(panel_inner_buy, features, full_cols)
X_val_spend_full = make_panel_X(panel_val_buy, features, full_cols)

model_spend_full = train_model(
    lgb.LGBMRegressor, grid_spend_full.best_params_, base_reg_params,
    X_inner_spend_full, panel_inner_buy["period_spend"].values,
    X_val_spend_full, panel_val_buy["period_spend"].values,
    "Spend Full",
)

# %% [markdown]
# ## 7. Generate Predictions (Customer-Level)
#
# For each customer, we predict:
# - `P(buy)` from the purchase classifier (one prediction per customer)
# - `E[spend|buy]` from the spend regressor (predict for ALL customers)
#
# Then compute CLV via discounted cash flow.

# %%
step("Generate customer-level predictions")

def compute_ml_clv(p_buy, e_spend_given_buy, discount_rate, horizon):
    """CLV = sum_{t=1}^{horizon} P(buy) * E[spend|buy] / (1+r)^t

    Since P(buy) and E[spend|buy] are constant per customer, this simplifies
    to P(buy) * E[spend|buy] * annuity_factor(r, horizon). We use the explicit
    loop for clarity and alignment with pymc-marketing.
    """
    # Vectorized annuity factor: sum of 1/(1+r)^t for t=1..horizon
    t = np.arange(1, horizon + 1)
    annuity = np.sum(1 / (1 + discount_rate) ** t)
    return p_buy * e_spend_given_buy * annuity

# Predict for ALL households (both train and test)
all_hh = features.index.values
X_all_rfm = features.loc[all_hh, rfm_cols]
X_all_full = features.loc[all_hh, full_cols]

# P(buy) — probability of purchase in any given period
p_buy_rfm = model_purch_rfm.predict_proba(X_all_rfm)[:, 1]
p_buy_full = model_purch_full.predict_proba(X_all_full)[:, 1]

# E[spend|buy] — expected spend conditional on purchase
e_spend_rfm = model_spend_rfm.predict(X_all_rfm)
e_spend_full = model_spend_full.predict(X_all_full)

# Clip negative spend predictions to 0
e_spend_rfm = np.clip(e_spend_rfm, 0, None)
e_spend_full = np.clip(e_spend_full, 0, None)

# CLV — holdout-length (12 periods) and 10-year (120 months)
clv_holdout_rfm = compute_ml_clv(p_buy_rfm, e_spend_rfm, DISCOUNT_RATE, N_HOLDOUT_PERIODS)
clv_holdout_full = compute_ml_clv(p_buy_full, e_spend_full, DISCOUNT_RATE, N_HOLDOUT_PERIODS)
clv_10yr_rfm = compute_ml_clv(p_buy_rfm, e_spend_rfm, DISCOUNT_RATE, CLV_HORIZON)
clv_10yr_full = compute_ml_clv(p_buy_full, e_spend_full, DISCOUNT_RATE, CLV_HORIZON)

print(f"Annuity factor (12 periods, r=0.01): {np.sum(1/(1+DISCOUNT_RATE)**np.arange(1,N_HOLDOUT_PERIODS+1)):.4f}")
print(f"Annuity factor (120 months, r=0.01): {np.sum(1/(1+DISCOUNT_RATE)**np.arange(1,CLV_HORIZON+1)):.4f}")

# Build predictions DataFrame
predictions = pd.DataFrame({
    "household_key": all_hh,
    "p_buy_rfm": p_buy_rfm,
    "p_buy_full": p_buy_full,
    "e_spend_rfm": e_spend_rfm,
    "e_spend_full": e_spend_full,
    "clv_holdout_rfm": clv_holdout_rfm,
    "clv_holdout_full": clv_holdout_full,
    "clv_10yr_rfm": clv_10yr_rfm,
    "clv_10yr_full": clv_10yr_full,
    "y_holdout_spend": features.loc[all_hh, "y_holdout_spend"].values,
    "split": ["train" if hh in set(hh_train) else "test" for hh in all_hh],
})

# Quick summary
print(f"\nPredictions summary (test set):")
test_mask = predictions["split"] == "test"
for col in ["p_buy_rfm", "p_buy_full", "e_spend_rfm", "e_spend_full",
            "clv_holdout_rfm", "clv_holdout_full", "clv_10yr_rfm", "clv_10yr_full"]:
    vals = predictions.loc[test_mask, col]
    print(f"  {col:25s}  mean={vals.mean():>10,.2f}  median={vals.median():>10,.2f}")

# %% [markdown]
# ## 8. Quick Test-Set Evaluation

# %%
step("Quick test-set evaluation")

test_pred = predictions[test_mask].copy()
y_actual = test_pred["y_holdout_spend"].values

for label, clv_col in [("RFM", "clv_holdout_rfm"), ("Full", "clv_holdout_full")]:
    y_pred = test_pred[clv_col].values
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = root_mean_squared_error(y_actual, y_pred)
    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(y_actual, y_pred)
    print(f"{label:6s}  MAE: ${mae:>8,.2f}  |  RMSE: ${rmse:>10,.2f}  |  Spearman: {rho:.4f} (p={pval:.2e})")

# Sub-model evaluation
print(f"\nPurchase model — observed vs predicted buy rate (test):")
X_test_rfm = features.loc[hh_test, rfm_cols]
X_test_full = features.loc[hh_test, full_cols]

panel_test_buy_rate = panel_test.groupby("household_key")["purchased"].mean()
for label, model in [("RFM", model_purch_rfm), ("Full", model_purch_full)]:
    X = X_test_rfm if label == "RFM" else X_test_full
    p_pred = model.predict_proba(X)[:, 1]
    # Observed purchase rate across all test panel rows
    obs_rate = panel_test["purchased"].mean()
    pred_rate = p_pred.mean()
    print(f"  {label}: observed={obs_rate:.4f}, predicted_mean={pred_rate:.4f}")

# %% [markdown]
# ## 9. Save All Artifacts

# %%
step("Save predictions and models")

# Predictions
predictions.to_parquet(_TABLES / "ml_clv_predictions.parquet", index=False)
print(f"Saved: {_TABLES / 'ml_clv_predictions.parquet'} ({len(predictions)} rows)")

# Models
for name, mdl in [
    ("model_purch_rfm.joblib",  model_purch_rfm),
    ("model_purch_full.joblib", model_purch_full),
    ("model_spend_rfm.joblib",  model_spend_rfm),
    ("model_spend_full.joblib", model_spend_full),
]:
    joblib.dump(mdl, _MODELS / name)
    print(f"Saved: {_MODELS / name}")

# Save best hyperparameters
hp = {
    "purch_rfm_best_params": grid_purch_rfm.best_params_,
    "purch_rfm_best_iteration": model_purch_rfm.best_iteration_,
    "purch_full_best_params": grid_purch_full.best_params_,
    "purch_full_best_iteration": model_purch_full.best_iteration_,
    "spend_rfm_best_params": grid_spend_rfm.best_params_,
    "spend_rfm_best_iteration": model_spend_rfm.best_iteration_,
    "spend_full_best_params": grid_spend_full.best_params_,
    "spend_full_best_iteration": model_spend_full.best_iteration_,
}
with open(_TABLES / "best_hyperparams.json", "w") as f:
    json.dump(hp, f, indent=2)

# Save feature column lists for evaluation script
cols_meta = {"rfm_cols": rfm_cols, "full_cols": full_cols}
with open(_TABLES / "feature_cols.json", "w") as f:
    json.dump(cols_meta, f, indent=2)

# Save train/test split
split_meta = {
    "hh_train": hh_train.tolist(),
    "hh_test": hh_test.tolist(),
}
with open(_TABLES / "train_test_split.json", "w") as f:
    json.dump(split_meta, f)

print("\nDone. Next: 03_evaluation.py")

# %% [markdown]
# ## Summary
#
# Four LightGBM models trained (purchase-incidence + conditional-amount decomposition):
#
# | Model | Type | Features | What it predicts |
# |-------|------|----------|------------------|
# | Purchase (RFM) | Classifier | 5 RFM | P(buy in any given period) |
# | Purchase (Full) | Classifier | 25 A-F | P(buy in any given period) |
# | Spend (RFM) | Regressor | 5 RFM | E[spend per period \| buy] |
# | Spend (Full) | Regressor | 25 A-F | E[spend per period \| buy] |
#
# CLV = Σ P(buy) × E[spend|buy] / (1+r)^t — directly parallels BTYD.
#
# Artifacts saved under `artifacts/part4/sec4.2-clv/`:
# - `tables/ml_clv_predictions.parquet` — per-customer predictions + actuals
# - `models/model_purch_{rfm,full}.joblib`, `models/model_spend_{rfm,full}.joblib`
# - `tables/best_hyperparams.json` — grid search results
# - `tables/feature_cols.json` — feature lists
# - `tables/train_test_split.json` — customer split for reproducibility
