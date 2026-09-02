"""Synthetic weekly panel for Chapter 3.4 (Causal Impact with Time Series).

Generates the committed seeds under ``data/generated/part3/sec3.4-causal-impact/``:

- ``weekly_sales.csv`` — what the analyst sees: 236 Mondays of brand revenue, a
  category demand index, a sibling product line that got no media, and a flag
  marking the eight campaign weeks.
- ``ground_truth.csv`` — the untreated path the campaign replaced and the true
  weekly effect, in dollars and as a share of the baseline. Used only to grade
  estimates and to build the failure scenarios; never to fit.

Why synthetic: the chapter needs a dated one-off intervention whose true effect
is known to the dollar, four years of pre-period so annual seasonality can be
learned and validated, control series the intervention provably could not have
touched, and a decaying effect with borrowed demand behind it. No public
marketing dataset carries a known causal answer, and injecting a lift into real
data (which the chapter's Dunnhumby section still does) cannot produce the
carryover-and-payback shape this chapter needs to teach reporting windows.

The story: a national CPG brand runs an eight-week TV and PR burst. Because the
flight is national there is no untreated market to compare against, so the
controls are series the campaign could not have reached: a syndicated category
demand index and a sibling product line that received no media.

Design notes (fixed on realism grounds BEFORE any model was run; do not tune
these to make chapter results look better):

- All three series share one AR(1) demand factor, one annual seasonal shape,
  and their own gentle growth. That shared structure is exactly why the
  controls are predictive, and it is what lets them absorb a market-wide shock.
- Each series carries its own AR(1) idiosyncratic noise, so the controls are
  informative but never sufficient.
- The effect ramps over two weeks to +12%, holds for the rest of the flight,
  decays over ``len(CARRYOVER_EFFECT)`` weeks after it ends, and is then partly
  paid back over ``len(PAYBACK_EFFECT)`` weeks of borrowed demand. Reading the
  campaign at the flight window, at flight-plus-carryover, and over the full
  post window therefore gives three different answers, and only the last one is
  the number to report.
- ``brand_sales`` equals ``brand_counterfactual`` plus ``true_effect`` exactly,
  so recovery can be graded without reconstructing the generator.

Regenerate by running this file directly. ``generate_dataset(seed=42)`` must
reproduce the committed CSVs exactly; any drift is an RNG or logic regression.

Author-generated data: no third-party rights attach, released under CC BY 4.0
with the rest of this repository's data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from msbook.paths import DATA_DIR

SEED = 42

START_DATE = "2021-09-06"  # a Monday; the flight then starts 2025-09-01
PRE_WEEKS = 208  # four years of pre-period
CAMPAIGN_WEEKS = 8
POST_WEEKS = 20  # observed weeks after the flight ends
N_WEEKS = PRE_WEEKS + CAMPAIGN_WEEKS + POST_WEEKS

# Levels and growth (per year, applied linearly)
BRAND_BASE = 1_150_000.0
BRAND_GROWTH = 0.05
CATEGORY_BASE = 100.0
CATEGORY_GROWTH = 0.02
SIBLING_BASE = 420_000.0
SIBLING_GROWTH = 0.01

# One annual shape, scaled per series
SEASON_AMPLITUDE = 0.11
SEASON_HARMONIC_2 = 0.35  # second harmonic as a share of the first
SEASON_PHASE = 0.019  # puts the annual peak in late November
SEASON_SCALE = {"brand": 1.00, "category": 0.95, "sibling": 0.85}

# Shared demand factor (macro, weather, category-wide promotion)
COMMON_RHO = 0.80
COMMON_SD = 0.050
COMMON_LOADING = {"brand": 1.00, "category": 1.00, "sibling": 0.90}

# Series-specific noise
IDIO_RHO = 0.30
IDIO_SD = {"brand": 0.022, "category": 0.012, "sibling": 0.035}

# Campaign effect as a share of the untreated baseline.
FLIGHT_EFFECT = (0.06, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12)
CARRYOVER_EFFECT = (0.075, 0.050, 0.030, 0.015)
PAYBACK_EFFECT = (-0.030,) * 10


def _ar1(rng: np.random.Generator, n: int, sd: float, rho: float) -> np.ndarray:
    """Return a stationary AR(1) path whose marginal standard deviation is ``sd``."""
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, sd)
    innovation_sd = sd * np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0.0, innovation_sd)
    return out


def _seasonal_shape(n: int) -> np.ndarray:
    """One annual profile, two harmonics, mean zero over a full year."""
    t = np.arange(n, dtype=float)
    angle = 2.0 * np.pi * (t / 52.0 + SEASON_PHASE)
    return SEASON_AMPLITUDE * (np.sin(angle) + SEASON_HARMONIC_2 * np.sin(2.0 * angle))


def effect_profile() -> np.ndarray:
    """Weekly effect as a share of the untreated baseline, length ``N_WEEKS``."""
    profile = np.zeros(N_WEEKS, dtype=float)
    start = PRE_WEEKS
    profile[start : start + CAMPAIGN_WEEKS] = FLIGHT_EFFECT
    after = start + CAMPAIGN_WEEKS
    tail = np.array(CARRYOVER_EFFECT + PAYBACK_EFFECT, dtype=float)
    profile[after : after + len(tail)] = tail
    return profile


def generate_dataset(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Build the observed panel and its ground truth. Pure: no file I/O."""
    rng = np.random.default_rng(seed)

    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    years = np.arange(N_WEEKS, dtype=float) / 52.0
    season = _seasonal_shape(N_WEEKS)
    common = _ar1(rng, N_WEEKS, COMMON_SD, COMMON_RHO)

    def series(name: str, base: float, growth: float) -> np.ndarray:
        trend = 1.0 + growth * years
        seasonal = 1.0 + SEASON_SCALE[name] * season
        idio = _ar1(rng, N_WEEKS, IDIO_SD[name], IDIO_RHO)
        return (
            base
            * trend
            * seasonal
            * (1.0 + idio)
            * (1.0 + COMMON_LOADING[name] * common)
        )

    brand_cf = series("brand", BRAND_BASE, BRAND_GROWTH)
    category = series("category", CATEGORY_BASE, CATEGORY_GROWTH)
    sibling = series("sibling", SIBLING_BASE, SIBLING_GROWTH)

    effect_pct = effect_profile()
    true_effect = brand_cf * effect_pct
    brand_observed = brand_cf + true_effect

    campaign = np.zeros(N_WEEKS, dtype=int)
    campaign[PRE_WEEKS : PRE_WEEKS + CAMPAIGN_WEEKS] = 1

    panel = pd.DataFrame(
        {
            "week": weeks,
            "brand_sales": np.round(brand_observed, 2),
            "category_index": np.round(category, 4),
            "sibling_line": np.round(sibling, 2),
            "campaign": campaign,
        }
    )
    truth = pd.DataFrame(
        {
            "week": weeks,
            "brand_counterfactual": np.round(brand_cf, 2),
            "true_effect": np.round(true_effect, 2),
            "effect_pct": np.round(effect_pct, 6),
        }
    )
    return {"panel": panel, "truth": truth}


def write_dataset(output_dir: str | Path | None = None, seed: int = SEED) -> None:
    """Regenerate the canonical CSVs in their committed location."""
    target = (
        Path(output_dir)
        if output_dir is not None
        else DATA_DIR / "generated" / "part3" / "sec3.4-causal-impact"
    )
    target.mkdir(parents=True, exist_ok=True)
    tables = generate_dataset(seed=seed)
    tables["panel"].to_csv(target / "weekly_sales.csv", index=False)
    tables["truth"].to_csv(target / "ground_truth.csv", index=False)

    truth = tables["truth"]
    flight = truth.loc[truth["effect_pct"] > 0].head(CAMPAIGN_WEEKS)
    full = truth.iloc[PRE_WEEKS:]
    print(f"Wrote {len(tables['panel']):,} weeks to {target / 'weekly_sales.csv'}")
    print(f"Wrote ground truth to {target / 'ground_truth.csv'}")
    print(
        f"True effect: flight window ${flight['true_effect'].sum():,.0f}, "
        f"full post window ${full['true_effect'].sum():,.0f}"
    )


if __name__ == "__main__":
    write_dataset()
