# LightGBM CLV Companion Pipeline

A three-step LightGBM implementation of customer lifetime value modeling
that complements the chapter's primary PyMC-Marketing demo
(`../sec4.2_clv_pymc_marketing.ipynb`).

## When to use which

- **PyMC-Marketing demo** (Bayesian BTYD: BG/NBD + Gamma-Gamma) — the
  chapter's main path. Best for non-contractual settings with limited
  features and a need for uncertainty quantification.
- **LightGBM pipeline** (here) — best if you already have rich behavioral
  features, a production ML stack, and want to feed BTYD outputs into a
  supervised regressor on a fixed-horizon target. The chapter's "ML
  regression on a fixed-horizon target" section links these scripts as a
  worked end-to-end example.

## Run order

```
01_data_features.py   → builds purchase / amount features from Dunnhumby
                        (calibration vs holdout split). Writes
                        ml_clv_features.parquet and ml_clv_panel.parquet
                        under artifacts/part4/sec4.2-clv/tables/.

02_modeling.py        → trains LightGBM purchase classifier + amount
                        regressor (RFM-only and full-feature variants);
                        saves models to artifacts/part4/sec4.2-clv/models/
                        and predictions back to .../tables/.

03_evaluation.py      → loads the predictions + models and produces
                        evaluation figures + SHAP explanations into
                        images/part4/sec4.2/ and artifact tables.
```

Each script imports `msbook.paths` so it runs from any cwd once
`pip install -e .` has been done from the repo root. Data is loaded via
`notebooks/_retail_data.py` (Dunnhumby loader reading from `data/raw/`).
