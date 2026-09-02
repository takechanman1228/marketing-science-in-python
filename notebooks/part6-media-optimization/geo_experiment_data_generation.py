"""Generate the committed synthetic panel for Chapter 6.3's geo experiment.

The data is created from scratch for this repository. It contains 40 anonymous
geographies observed for 104 Mondays. The final eight weeks are a randomized
matched-pair experiment: one geography in each of eight pairs has its Google
Search spend cut by 30%, with true marginal ROAS planted at 0.20. The panel ends
six weeks after the sec6.2 modeling period, so the experiment can calibrate that
model without a recency gap.

Running this file rewrites the two committed seed CSVs under
``data/generated/part6/sec6.3-geo/``. The generation and assignment are fully
seeded so the fast regression tests can detect accidental drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from msbook.paths import DATA_DIR


SEED = 20260888
N_GEOS = 40
N_WEEKS = 104
N_PAIRS = 8
TEST_WEEKS = 8
START_DATE = "2022-04-04"
SPEND_MULTIPLIER = 0.70
TRUE_ROAS = 0.20


def _ar1_noise(
    rng: np.random.Generator,
    n: int,
    innovation_sd: float,
    rho: float,
) -> np.ndarray:
    """Return a stationary-looking AR(1) path with a fixed warm start."""
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, innovation_sd / np.sqrt(1.0 - rho**2))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0.0, innovation_sd)
    return out


def market_metrics(panel: pd.DataFrame, test_start: str | pd.Timestamp) -> pd.DataFrame:
    """Compute pre-period similarity, size, and normalized trend by geography."""
    work = panel.copy()
    work["week"] = pd.to_datetime(work["week"])
    pre = work.loc[work["week"] < pd.Timestamp(test_start)]
    wide = pre.pivot(index="week", columns="geo", values="revenue").sort_index()
    aggregate = wide.sum(axis=1)

    rows: list[dict[str, float | str]] = []
    x = np.arange(len(wide), dtype=float)
    for geo in wide.columns:
        series = wide[geo]
        leave_one_out = aggregate - series
        mean_revenue = float(series.mean())
        rows.append(
            {
                "geo": str(geo),
                "pre_corr": float(series.corr(leave_one_out)),
                "pre_mean_revenue": mean_revenue,
                "pre_trend": float(np.polyfit(x, series.to_numpy(), 1)[0] / mean_revenue),
            }
        )
    return pd.DataFrame(rows).sort_values("geo").reset_index(drop=True)


def screen_pair_and_assign(
    panel: pd.DataFrame,
    test_start: str | pd.Timestamp,
    n_pairs: int = N_PAIRS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Screen on correlation, greedily match on size/trend, randomize in pairs."""
    metrics = market_metrics(panel, test_start)
    selected = (
        metrics.sort_values(["pre_corr", "geo"], ascending=[False, True])
        .head(2 * n_pairs)
        .copy()
    )

    # Size-first adjacent matching makes the block construction easy to audit.
    # Normalized trends remain in the assignment table as a balance diagnostic
    # and break ties between nearly equal-size markets.
    ordered = list(
        selected.sort_values(["pre_mean_revenue", "pre_trend", "geo"]).index
    )
    pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered), 2)]

    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []
    for pair_number, (left, right) in enumerate(pairs, start=1):
        geos = sorted([str(selected.loc[left, "geo"]), str(selected.loc[right, "geo"])])
        treated_index = int(rng.integers(0, 2))
        for index, geo in enumerate(geos):
            row = selected.loc[selected["geo"] == geo].iloc[0]
            records.append(
                {
                    "geo": geo,
                    "pair": pair_number,
                    "role": "test" if index == treated_index else "control",
                    "pre_corr": float(row["pre_corr"]),
                    "pre_mean_revenue": float(row["pre_mean_revenue"]),
                    "pre_trend": float(row["pre_trend"]),
                }
            )

    return pd.DataFrame(records).sort_values(["pair", "role", "geo"]).reset_index(drop=True)


def generate_dataset(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return observed panel, planted-effect truth, and frozen assignment."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS, dtype=float)

    # A smooth market-wide cycle plus a modest common shock. The cycle rises
    # between the last pre-period weeks and the test, making naive before/after
    # look too optimistic while a contemporaneous control removes it.
    seasonal = 0.070 * np.sin(2.0 * np.pi * (t - 92.0) / 52.0)
    seasonal += 0.025 * np.cos(4.0 * np.pi * (t - 6.0) / 52.0)
    common_shock = _ar1_noise(rng, N_WEEKS, innovation_sd=0.006, rho=0.55)
    common_demand = 1.0 + 0.0010 * t + seasonal + common_shock

    spend_cycle = 1.0 + 0.0012 * t + 0.10 * np.sin(2.0 * np.pi * (t - 88.0) / 52.0)
    # The first 16 geographies are eight realistic matched-market pairs. Their
    # sizes still come from a log-normal distribution, but each pair gets small
    # independent perturbations around a shared size/trend. The remaining 24
    # markets are deliberately noisier screening candidates and donor markets.
    # Fixed quantiles of a LogNormal distribution give a realistic size spread
    # while keeping adjacent candidate pairs distinguishable under noise.
    size_z = np.array([-1.50, -1.05, -0.62, -0.20, 0.20, 0.62, 1.05, 1.50])
    pair_sizes = np.exp(-0.5 * 0.38**2 + 0.38 * size_z)
    pair_trends = rng.normal(0.0, 0.00020, size=N_PAIRS)
    pair_intercepts = rng.normal(0.0, 0.014, size=N_PAIRS)
    geo_sizes = np.empty(N_GEOS, dtype=float)
    geo_trends = np.empty(N_GEOS, dtype=float)
    geo_intercepts = np.empty(N_GEOS, dtype=float)
    for i in range(2 * N_PAIRS):
        pair_index = i // 2
        geo_sizes[i] = pair_sizes[pair_index] * np.exp(rng.normal(0.0, 0.008))
        geo_trends[i] = pair_trends[pair_index] + rng.normal(0.0, 0.00002)
        geo_intercepts[i] = pair_intercepts[pair_index] + rng.normal(0.0, 0.002)
    geo_sizes[2 * N_PAIRS:] = rng.lognormal(
        mean=-0.5 * 0.45**2, sigma=0.45, size=N_GEOS - 2 * N_PAIRS
    )
    geo_trends[2 * N_PAIRS:] = rng.normal(0.0, 0.00045, size=N_GEOS - 2 * N_PAIRS)
    geo_intercepts[2 * N_PAIRS:] = rng.normal(0.0, 0.025, size=N_GEOS - 2 * N_PAIRS)

    rows: list[dict[str, float | str]] = []
    for i in range(N_GEOS):
        geo = f"Geo {i + 1:02d}"
        size = float(geo_sizes[i])
        candidate_pair = i < 2 * N_PAIRS
        revenue_noise = _ar1_noise(
            rng,
            N_WEEKS,
            innovation_sd=0.015 if candidate_pair else 0.070,
            rho=0.45,
        )
        spend_noise = _ar1_noise(
            rng,
            N_WEEKS,
            innovation_sd=0.030 if candidate_pair else 0.060,
            rho=0.30,
        )

        baseline_revenue = (
            150_000.0
            * size
            * (common_demand + geo_intercepts[i] + geo_trends[i] * t + revenue_noise)
        )
        baseline_spend = 18_000.0 * size * spend_cycle * np.exp(spend_noise)

        for week, spend, revenue in zip(weeks, baseline_spend, baseline_revenue):
            rows.append(
                {
                    "geo": geo,
                    "week": week.strftime("%Y-%m-%d"),
                    "spend": round(float(spend), 2),
                    "revenue": round(float(revenue), 2),
                }
            )

    baseline_panel = pd.DataFrame(rows).sort_values(["geo", "week"]).reset_index(drop=True)
    test_start = weeks[-TEST_WEEKS]
    test_end = weeks[-1]
    assignment = screen_pair_and_assign(baseline_panel, test_start, seed=seed)
    treated_geos = set(assignment.loc[assignment["role"] == "test", "geo"])

    panel = baseline_panel.copy()
    week_values = pd.to_datetime(panel["week"])
    treated_window = panel["geo"].isin(treated_geos) & week_values.between(test_start, test_end)
    baseline_test_spend = panel.loc[treated_window, "spend"].to_numpy(dtype=float)
    incremental_spend = baseline_test_spend * (SPEND_MULTIPLIER - 1.0)
    incremental_revenue = incremental_spend * TRUE_ROAS

    panel.loc[treated_window, "spend"] = np.round(
        baseline_test_spend + incremental_spend, 2
    )
    panel.loc[treated_window, "revenue"] = np.round(
        panel.loc[treated_window, "revenue"].to_numpy(dtype=float) + incremental_revenue,
        2,
    )

    truth = panel.loc[treated_window, ["geo", "week"]].copy()
    truth["incremental_spend"] = np.round(incremental_spend, 2)
    truth["incremental_revenue"] = np.round(incremental_revenue, 2)
    total_revenue = float(truth["incremental_revenue"].sum())
    total_spend = float(truth["incremental_spend"].sum())
    truth["true_incremental_revenue_total"] = round(total_revenue, 2)
    truth["true_roas"] = total_revenue / total_spend
    truth = truth.sort_values(["geo", "week"]).reset_index(drop=True)

    return panel, truth, assignment


def write_dataset(output_dir: str | Path | None = None, seed: int = SEED) -> None:
    """Regenerate the canonical CSVs in their committed location."""
    target = Path(output_dir) if output_dir is not None else (
        DATA_DIR / "generated" / "part6" / "sec6.3-geo"
    )
    target.mkdir(parents=True, exist_ok=True)
    panel, truth, assignment = generate_dataset(seed=seed)
    panel.to_csv(target / "geo_panel.csv", index=False)
    truth.to_csv(target / "geo_ground_truth.csv", index=False)

    test_start = pd.to_datetime(panel["week"]).sort_values().unique()[-TEST_WEEKS]
    print(f"Wrote {len(panel):,} panel rows to {target / 'geo_panel.csv'}")
    print(f"Wrote {len(truth):,} planted-effect rows to {target / 'geo_ground_truth.csv'}")
    print(
        f"Frozen design: {len(assignment) // 2} pairs, test starts "
        f"{pd.Timestamp(test_start).date()}, true ROAS={truth['true_roas'].iloc[0]:.3f}"
    )


if __name__ == "__main__":
    write_dataset()
