"""Synthetic weekly sales generator for Chapter 5.3 (Sales Forecasting and
Scenario Planning).

Generates the committed seeds under ``data/generated/part5/sec5.3-sales-forecast/``:

- ``weekly_sales.csv``       — what the analyst sees: six 52-week fiscal years
  of weekly e-commerce revenue with holiday/promotion flags.
- ``campaign_effect_estimate.csv`` — the "measured" incremental effects the
  scenario section consumes (a point estimate and an uncertainty, as a geo
  experiment or a calibrated MMM would deliver them).
- ``ground_truth.csv``       — the generator's noise-free expected sales and
  the true campaign counterfactual. Never used to fit or tune the chapter's
  models; only the companion notebook's closing section and the tests read it.

Why synthetic: the chapter needs several years of real-calendar history with a
repeatable holiday peak, multiple late-September forecast origins, AND a known
causal campaign effect, under a license that allows redistribution. We did not
find a public retail dataset that combines all of these (Dunnhumby has a
synthetic date anchor and no annual seasonality; Online Retail II ends 1 week
into its second December; Iowa Liquor is wholesale purchasing with a reporting
break in 2016).

Design notes (fixed on realism grounds BEFORE any model was run; do not tune
these to make chapter results look better):

- Calendar: 312 consecutive Monday-start weeks form six fixed 52-week
  planning years. Each block closes in late December and contains Christmas
  in one of its final two weeks (fiscal week 51 in the earlier years, 52 in
  the later ones, because a real year is slightly longer than 52 weeks).
  Event weeks (Thanksgiving, Cyber week, gift rush, the Christmas-week drop)
  are located from the REAL calendar, so they drift the way real holidays do.
- log-additive DGP: level + ~6.5%/year growth + smooth annual seasonality +
  calendar-event effects whose strength varies by year + recurring
  business-as-usual promotions + a slow random-walk level drift + AR(1) noise
  that is larger in the holiday stretch + two one-off shocks.
- The recurring holiday promotion (fiscal weeks 47-50) is part of business as
  usual and therefore part of the baseline a forecast learns. The chapter's
  "additional holiday campaign" scenario is an increment ON TOP of it, with
  true lift ``TRUE_CAMPAIGN_LIFT`` applied to fiscal weeks 47-50 of the final
  year. The shipped estimate (mean 0.08, se 0.025) deliberately differs from
  the truth (0.07): measurements are noisy.

Regenerate with:  python sales_forecast_data_generation.py
``generate_dataset(seed=42)`` must reproduce the committed CSVs exactly; the
test suite treats any drift as an RNG/logic regression (part6 MMM precedent).

Author-generated data: no third-party rights attach. Reuse freely with
attribution (CC BY 4.0); the generator itself is covered by the repo's MIT
license.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_WEEKS = 312                               # six fixed 52-week fiscal years
FIRST_WEEK = pd.Timestamp("2014-01-06")     # a Monday; blocks close late December
SEED = 42

BASE_WEEKLY_SALES = 150_000.0               # dollars, first-year level
ANNUAL_GROWTH = 0.065

# True causal effect of the FY2019 "additional holiday campaign" scenario:
# +7% on fiscal weeks 47-50 of the final year. The shipped estimate in
# campaign_effect_estimate.csv is 0.08 +/- 0.025 — noisy, like a real one.
TRUE_CAMPAIGN_LIFT = 0.07
CAMPAIGN_FISCAL_WEEKS = (47, 48, 49, 50)

# Recurring business-as-usual promotions (part of the baseline, every year).
BAU_PROMO_FISCAL_WEEKS = (19, 20, 38, 47, 48, 49, 50)


def _fiscal_calendar() -> pd.DataFrame:
    """312 Monday-start weeks tagged with fiscal year (2014..2019) and week 1..52."""
    weeks = pd.date_range(FIRST_WEEK, periods=N_WEEKS, freq="W-MON")
    return pd.DataFrame(
        {
            "week": weeks,
            "fiscal_year": 2014 + np.arange(N_WEEKS) // 52,
            "fiscal_week": np.arange(N_WEEKS) % 52 + 1,
        }
    )


def _week_containing(
    date: pd.Timestamp, weeks: pd.DatetimeIndex
) -> pd.Timestamp | None:
    """The Monday-start week that contains ``date``; None if out of range."""
    candidates = weeks[weeks <= date]
    if len(candidates) == 0:
        return None
    week = candidates[-1]
    if date - week > pd.Timedelta(days=6):
        return None
    return week


def _thanksgiving(year: int) -> pd.Timestamp:
    """Fourth Thursday of November."""
    november = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thursdays = november[november.dayofweek == 3]
    return thursdays[3]


def _event_table(cal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per-week calendar-event effects (log scale), located from real dates.

    Event strength varies by year (a shared holiday-strength factor plus
    event-level noise), so no two Q4s are alike — the part of Q4 uncertainty
    a model cannot learn away.
    """
    weeks = pd.DatetimeIndex(cal["week"])
    effect = pd.Series(0.0, index=weeks)
    holiday = pd.Series(0, index=weeks, dtype=int)

    for year in sorted(cal["week"].dt.year.unique()):
        year_strength = rng.normal(0.0, 0.07)   # shared Q4 strength for this year

        def bump(anchor: pd.Timestamp, mean: float, sd: float, scaled: bool = True):
            week = _week_containing(anchor, weeks)
            draw = rng.normal(mean, sd)   # always drawn: keeps the RNG stream stable
            if week is None:
                return
            effect[week] += draw + (year_strength if scaled else 0.0)
            holiday[week] = 1

        thanksgiving = _thanksgiving(year)
        bump(thanksgiving, 0.30, 0.04)                        # Black Friday week
        bump(thanksgiving + pd.Timedelta(days=7), 0.20, 0.04)  # Cyber week
        christmas_week = _week_containing(pd.Timestamp(f"{year}-12-25"), weeks)
        bump(christmas_week - pd.Timedelta(days=14), 0.16, 0.03)   # gift rush
        bump(christmas_week - pd.Timedelta(days=7), 0.24, 0.04)    # shipping cutoff
        bump(christmas_week, -0.28, 0.04, scaled=False)            # post-cutoff drop
        bump(pd.Timestamp(f"{year + 1}-01-01"), -0.10, 0.03, scaled=False)

    return pd.DataFrame(
        {"week": weeks, "event_effect": effect.to_numpy(), "holiday_week": holiday.to_numpy()}
    )


def generate_dataset(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Return the three seed tables. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    cal = _fiscal_calendar()
    t = np.arange(N_WEEKS)

    # Systematic components (log scale) -------------------------------------
    level = np.log(BASE_WEEKLY_SALES)
    trend = np.log1p(ANNUAL_GROWTH) / 52.0 * t
    fw = cal["fiscal_week"].to_numpy()
    annual = (
        -0.055 * np.cos(2 * np.pi * (fw - 2) / 52.0)   # winter low, autumn ramp
        - 0.035 * np.cos(4 * np.pi * (fw - 6) / 52.0)  # mild summer dip
    )

    events = _event_table(cal, rng)
    bau_promo = np.isin(fw, BAU_PROMO_FISCAL_WEEKS).astype(int)
    promo_effect = rng.normal(0.09, 0.02, size=N_WEEKS) * bau_promo

    # One-off shocks the analyst has to live with ---------------------------
    shock = np.zeros(N_WEEKS)
    shock[(cal["fiscal_year"] == 2016) & cal["fiscal_week"].isin([23, 24])] = (-0.20, -0.12)
    shock[(cal["fiscal_year"] == 2018) & (cal["fiscal_week"] == 10)] = 0.20

    expected_log = level + trend + annual + events["event_effect"].to_numpy() + promo_effect + shock

    # Noise: slow level drift + AR(1), larger in the holiday stretch --------
    drift = np.cumsum(rng.normal(0.0, 0.003, size=N_WEEKS))
    sigma = np.where((fw >= 46) | (fw <= 1), 0.060, 0.034)
    ar = np.zeros(N_WEEKS)
    for i in range(1, N_WEEKS):
        ar[i] = 0.5 * ar[i - 1] + rng.normal(0.0, sigma[i])

    sales = np.round(np.exp(expected_log + drift + ar), 0)

    weekly_sales = pd.DataFrame(
        {
            "week": cal["week"],
            "sales": sales,
            "fiscal_year": cal["fiscal_year"],
            "fiscal_week": cal["fiscal_week"],
            "holiday_week": events["holiday_week"].to_numpy(),
            "bau_promotion": bau_promo,
        }
    )

    campaign_week = (
        (cal["fiscal_year"] == 2019) & cal["fiscal_week"].isin(CAMPAIGN_FISCAL_WEEKS)
    ).astype(int)
    ground_truth = pd.DataFrame(
        {
            "week": cal["week"],
            "expected_sales": np.round(np.exp(expected_log), 0),
            "sales": sales,
            "campaign_week": campaign_week,
            "true_campaign_lift": TRUE_CAMPAIGN_LIFT * campaign_week.to_numpy(),
            "sales_with_campaign": np.round(
                sales * (1.0 + TRUE_CAMPAIGN_LIFT * campaign_week.to_numpy()), 0
            ),
        }
    )

    campaign_effect_estimate = pd.DataFrame(
        {
            "scenario": [
                "business_as_usual",
                "additional_holiday_campaign",
                "additional_media",
            ],
            "effect_type": ["relative", "relative", "absolute_dollars"],
            "effect_mean": [0.0, 0.08, 60_000.0],
            "effect_se": [0.0, 0.025, 25_000.0],
            "source": [
                "baseline forecast",
                "geo experiment on last year's holiday campaign (Section 6.3)",
                "calibrated MMM response curve (Sections 6.2 and 6.4)",
            ],
        }
    )

    return {
        "weekly_sales": weekly_sales,
        "campaign_effect_estimate": campaign_effect_estimate,
        "ground_truth": ground_truth,
    }


def main() -> None:
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "data" / "generated" / "part5" / "sec5.3-sales-forecast"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate_dataset(SEED)
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
