"""
MMM synthetic data generation with identifiability QC.

- Generates base data via PySiMMMulator.
- Applies a segment-mix style post-process to introduce OFF weeks,
  larger share variation, and lower channel correlations.
- Rebuilds sales through geometric adstock and Hill saturation, so the dataset
  has the response curves that Chapters 6.2 and 6.4 discuss. Each channel's
  curve is solved to hit a chosen true average ROI and a chosen true marginal
  ROI; the ground-truth CSV reports both.

Four settings keep the media coefficients identifiable. Saturation and carryover
smooth the media contribution, which makes it quiet relative to the baseline's own
movement, and a quiet signal cannot be separated from the baseline by any
estimator. BASELINE_SCALE raises media's share of sales,
BASELINE_RESIDUAL_KEPT limits the baseline movement no control explains,
OBSERVATION_NOISE_R2 puts the reported fit back where a synthetic panel
should sit, and SEARCH_SPEND_FLATNESS makes paid search the one channel the
model cannot learn.

Requires pysimmmulator==0.5.1, which is not part of the chapter environment
because only this generator needs it:

    pip install pysimmmulator==0.5.1

The committed dataset under data/generated/part6/sec6.2-mmm/ was produced with
that version. 0.6.x changed the AdstockParameters config API and rejects the
true_lambda_decay keyword below, so upgrading means porting this config and
then deliberately regenerating the dataset and re-running the sec6.2 notebook.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import brentq
from pysimmmulator import Simulate

# ---------------------------------------------------------------------------
# Response-curve ground truth.
#
# The media contribution is generated with geometric adstock on EXPOSURE and a
# Hill saturation with alpha = 1 (Meridian's `slope_m` default is
# Deterministic(1.0), so alpha = 1 keeps the generating process inside the
# model's family). Each channel has two free parameters, beta and the
# half-saturation point K, and two targets: the realised average ROI and a
# chosen marginal ROI. Beta scales both linearly, so the marginal/average ratio
# is a function of K alone -- a 1-D root find, then beta closes the average.
# ---------------------------------------------------------------------------

#: Adstock half-life in weeks. These are the values sec6.2 already states as
#: priors, so the prose and the data now agree.
TRUE_HALF_LIFE_WEEKS = {
    "TV_CTV": 6.0, "OOH": 6.0, "Google": 1.0, "Meta": 3.0, "TikTok": 3.0,
}

#: True average ROI (incremental revenue / spend over the whole period).
TRUE_AVERAGE_ROI = {
    "TV_CTV": 4.17, "OOH": 6.28, "Google": 0.35, "Meta": 3.75, "TikTok": 3.01,
}

#: True marginal ROI, defined as Meridian defines mROI: the return on a 1%
#: exposure increase. OOH and Meta sit above a 40%-margin break-even ROAS of
#: 2.50; TV/CTV, TikTok and Google sit below it. TV/CTV outranks Meta on the
#: average and is outranked by it on the margin, which is the whole point of
#: optimising on marginal rather than average return.
TRUE_MARGINAL_ROI = {
    "TV_CTV": 1.80,
    "OOH": 3.50,
    "Meta": 2.80,
    "TikTok": 0.80,
}

#: Google Search is pinned on the quantity Chapter 6.3's geo experiment
#: measures instead: the ROAS of a 30% spend cut sustained for eight weeks.
#: Its 1% mROI then follows from the curve rather than being chosen.
GOOGLE_PULLBACK_ROAS = 0.20
PULLBACK_MULTIPLIER = 0.70
PULLBACK_WEEKS = 8


#: Carryover window. Matches the `max_lag` the sec6.2 Meridian model uses, so the
#: generating process sits inside the model family rather than being a truncated
#: approximation of it.
TRUE_MAX_LAG = 12

#: Fraction of the baseline that no control variable explains. See the note where
#: it is applied: media contribution is smooth under saturation, so an unmodelled
#: baseline swing larger than the media swing makes the channel coefficients
#: unidentifiable.
BASELINE_RESIDUAL_KEPT = 0.60

#: Scale on the non-media baseline. Saturation and adstock smooth the media
#: contribution, so at the simulator's default baseline media is both small and
#: quiet and no estimator can separate it from the baseline's own movement.
#: Shrinking the baseline raises media's share of sales (and the advertiser's
#: ad-to-sales ratio) until the media coefficients are identified.
BASELINE_SCALE = 0.40

#: How strongly paid-search budget follows seasonal demand. Search budgets track
#: query volume, and query volume tracks demand, so search spend is partly a
#: consequence of the outcome it is credited with. That co-movement is the reason
#: paid search is the hardest channel to identify in an MMM: its weekly pattern is
#: close to a linear combination of the seasonality controls, so the likelihood has
#: little to say about its coefficient and the posterior falls back on the prior.
#: 0 reproduces the independent-budget behaviour of earlier versions.
SEARCH_DEMAND_COUPLING = 0.5

#: How close paid search runs to a flat always-on budget. 0 leaves the simulator's
#: week-to-week variation; 1 holds search spend at a constant weekly dollar amount.
#: Section 6.2 already warns that a channel whose spend never varies cannot be
#: learned from the data; this is that warning made true for one channel, and it is
#: why the Search posterior in the demo is the prior with barely any update.
SEARCH_SPEND_FLATNESS = 0.9

#: Target in-sample R-squared. Reported fit is restored with independent
#: observation noise rather than by leaving structure in the baseline: white
#: noise is orthogonal to the media basis, so it lowers R-squared without
#: handing the media coefficients someone else's movement.
OBSERVATION_NOISE_R2 = 0.980


def geometric_adstock(exposure, decay: float, max_lag: int = TRUE_MAX_LAG):
    """Return sum_{i=0..max_lag} x_{t-i} * decay**i -- Meridian's own definition."""
    exposure = np.asarray(exposure, dtype=float)
    out = np.zeros(len(exposure), dtype=float)
    for lag in range(max_lag + 1):
        if lag == 0:
            out += exposure
        else:
            out[lag:] += (decay ** lag) * exposure[:-lag]
    return out


def hill_response(exposure: np.ndarray, decay: float, beta: float, half_sat: float) -> np.ndarray:
    """Incremental revenue per week: beta * a / (a + K)."""
    adstocked = geometric_adstock(exposure, decay)
    return beta * adstocked / (adstocked + half_sat)


def _avg_roi(exposure, spend, decay, half_sat) -> float:
    return float(hill_response(exposure, decay, 1.0, half_sat).sum() / spend.sum())


def _mroi_1pct(exposure, spend, decay, half_sat) -> float:
    base = hill_response(exposure, decay, 1.0, half_sat).sum()
    up = hill_response(exposure * 1.01, decay, 1.0, half_sat).sum()
    return float((up - base) / (0.01 * spend.sum()))


def _pullback_roas(exposure, spend, decay, half_sat) -> float:
    """Revenue lost per dollar removed by a sustained cut in the last weeks."""
    base = hill_response(exposure, decay, 1.0, half_sat)
    cut = exposure.copy()
    cut[-PULLBACK_WEEKS:] *= PULLBACK_MULTIPLIER
    reduced = hill_response(cut, decay, 1.0, half_sat)
    lost = base[-PULLBACK_WEEKS:].sum() - reduced[-PULLBACK_WEEKS:].sum()
    removed = spend[-PULLBACK_WEEKS:].sum() * (1.0 - PULLBACK_MULTIPLIER)
    return float(lost / removed)


def solve_response_curve(exposure, spend, decay, target_avg, target_metric, metric_fn):
    """Return (beta, half_sat) hitting both the average and the marginal target."""
    exposure = np.asarray(exposure, dtype=float)
    spend = np.asarray(spend, dtype=float)
    target_ratio = target_metric / target_avg

    def gap(half_sat: float) -> float:
        return metric_fn(exposure, spend, decay, half_sat) / _avg_roi(
            exposure, spend, decay, half_sat
        ) - target_ratio

    scale = float(geometric_adstock(exposure, decay).mean())
    lo, hi = scale * 1e-8, scale * 1e8
    if gap(lo) * gap(hi) > 0:
        raise ValueError(
            f"marginal/average ratio {target_ratio:.4f} is outside the reachable "
            f"range [{target_ratio + gap(lo):.4f}, {target_ratio + gap(hi):.4f}]"
        )
    half_sat = brentq(gap, lo, hi, xtol=scale * 1e-12, rtol=1e-14, maxiter=500)
    shape = hill_response(exposure, decay, 1.0, half_sat)
    beta = target_avg * spend.sum() / shape.sum()
    return float(beta), float(half_sat)



CHANNELS = ["TV_CTV", "OOH", "Google", "Meta", "TikTok"]

# Display names for outputs
CHANNEL_DISPLAY = {
    "TV_CTV": "TV/CTV",
    "OOH": "OOH",
    "Google": "Google Search",
    "Meta": "Meta",
    "TikTok": "TikTok",
}


def compute_identifiability_metrics(spend_df: pd.DataFrame) -> dict:
    """Compute identifiability metrics for spend matrix."""
    metrics: dict[str, dict] = {"overall": {}}

    off_rates = {col: (spend_df[col] == 0).mean() for col in spend_df.columns}
    metrics["overall"]["off_rates"] = off_rates
    metrics["overall"]["off_rate_mean"] = float(np.mean(list(off_rates.values())))

    total_spend = spend_df.sum(axis=1).replace(0, np.nan)
    shares = spend_df.div(total_spend, axis=0).fillna(0)
    share_stds = {}
    for col in spend_df.columns:
        std_val = shares[col][spend_df[col] > 0].std()
        if pd.isna(std_val):
            std_val = 0.0
        share_stds[col] = float(std_val)
    metrics["overall"]["share_stds"] = share_stds
    metrics["overall"]["share_std_mean"] = float(np.mean(list(share_stds.values())))

    corr = spend_df.corr().abs().fillna(0.0)
    mask = ~np.eye(len(spend_df.columns), dtype=bool)
    corr_vals = corr.values[mask]
    metrics["overall"]["channel_corr_max"] = float(np.nanmax(corr_vals))
    metrics["overall"]["channel_corr_matrix"] = corr

    return metrics


def check_identifiability_qc(
    metrics: dict,
    off_rate_min: float = 0.10,
    off_rate_max: float = 0.25,
    share_std_min: float = 0.05,
    corr_max: float = 0.85,
) -> tuple[bool, list[str]]:
    """Return pass/fail and warnings list."""
    warnings = []
    overall = metrics["overall"]

    off_rate = overall["off_rate_mean"]
    share_std = overall["share_std_mean"]
    corr = overall["channel_corr_max"]

    if off_rate < off_rate_min:
        warnings.append(f"OFF rate too low: {off_rate:.1%} < {off_rate_min:.0%}")
    if off_rate > off_rate_max:
        warnings.append(f"OFF rate too high: {off_rate:.1%} > {off_rate_max:.0%}")
    if share_std < share_std_min:
        warnings.append(f"Share std too low: {share_std:.3f} < {share_std_min}")
    if corr > corr_max:
        warnings.append(f"Channel corr too high: {corr:.3f} > {corr_max}")

    return (len(warnings) == 0), warnings


def _generate_segment_mix_weights(
    n_weeks: int,
    n_channels: int,
    rng: np.random.Generator,
    segment_len_min: int = 4,
    segment_len_max: int = 8,
    off_prob_min: float = 0.05,
    off_prob_max: float = 0.15,
    dirichlet_alpha_base: float = 8.0,
    channel_noise_std: float = 0.08,
    min_on_channels: int = 3,
) -> np.ndarray:
    """Generate per-week channel weights with OFF weeks and share variation."""
    weights = np.zeros((n_weeks, n_channels), dtype=float)

    idx = 0
    while idx < n_weeks:
        seg_len = int(rng.integers(segment_len_min, segment_len_max + 1))
        seg_end = min(idx + seg_len, n_weeks)

        off_prob = rng.uniform(off_prob_min, off_prob_max, size=n_channels)
        on_mask = rng.random(n_channels) > off_prob

        if on_mask.sum() < min_on_channels:
            forced = rng.choice(n_channels, size=min_on_channels, replace=False)
            on_mask[:] = False
            on_mask[forced] = True

        base_share = rng.dirichlet(np.full(on_mask.sum(), dirichlet_alpha_base))
        base = np.zeros(n_channels, dtype=float)
        base[on_mask] = base_share

        for t in range(idx, seg_end):
            noise = rng.lognormal(mean=0.0, sigma=channel_noise_std, size=n_channels)
            w = base * noise
            w[~on_mask] = 0.0
            total = w.sum()
            if total <= 0:
                w = base.copy()
                total = w.sum()
            weights[t] = w / total

        idx = seg_end

    return weights


def _median_ratio(df: pd.DataFrame, numer_col: str, denom_col: str) -> float:
    ratio = df[numer_col] / df[denom_col].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0
    return float(ratio.median())


def generate_dataset(
    base_seed: int = 42,
    max_retries: int = 5,
    off_rate_min: float = 0.10,
    off_rate_max: float = 0.25,
    share_std_min: float = 0.05,
    corr_max: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate data with QC. Returns (df, truth_df, qc_metrics)."""

    cfg = {
        "basic_params": {
            "years": 3,
            "channels_impressions": ["TV_CTV", "OOH", "Meta", "TikTok"],
            "channels_clicks": ["Google"],
            "frequency_of_campaigns": 1,
            "start_date": "2021/1/1",
            "true_cvr": {
                # Conversion rates chosen so most channels land at a true ROI
                # of roughly 3-6, with Google deliberately low (~0.2) to give
                # the §6.4 calibration demo a channel worth calibrating.
                "TV_CTV": 0.0020,
                "OOH": 0.0015,
                "Google": 0.15,
                "Meta": 0.0015,
                "TikTok": 0.0010,
            },
            "revenue_per_conv": 50.0,
        },
        "baseline_params": {
            "base_p": 200000,
            "trend_p": 150000,
            "temp_var": 10,
            "temp_coef_mean": 25,
            "temp_coef_sd": 5,
            "error_std": 150,
        },
        "ad_spend_params": {
            "campaign_spend_mean": 500000,
            "campaign_spend_std": 150000,
            "max_min_proportion_on_each_channel": {
                "TV_CTV": {"min": 0.20, "max": 0.40},
                "OOH": {"min": 0.03, "max": 0.10},
                "Google": {"min": 0.15, "max": 0.30},
                "Meta": {"min": 0.10, "max": 0.25},
                "TikTok": {"min": 0.03, "max": 0.12},
            },
        },
        "media_params": {
            "true_cpm": {
                "TV_CTV": 12.0,
                "OOH": 5.0,
                "Meta": 8.0,
                "TikTok": 6.0,
            },
            "true_cpc": {"Google": 2.50},
            "noisy_cpm_cpc": {
                "TV_CTV": {"loc": 0.0, "scale": 1.0},
                "OOH": {"loc": 0.0, "scale": 0.5},
                "Google": {"loc": 0.0, "scale": 0.25},
                "Meta": {"loc": 0.0, "scale": 0.8},
                "TikTok": {"loc": 0.0, "scale": 0.6},
            },
        },
        "cvr_params": {
            "noisy_cvr": {
                "TV_CTV": {"loc": 0.0, "scale": 0.0010},
                "OOH": {"loc": 0.0, "scale": 0.0008},
                "Google": {"loc": 0.0, "scale": 0.02},
                "Meta": {"loc": 0.0, "scale": 0.0008},
                "TikTok": {"loc": 0.0, "scale": 0.0006},
            },
        },
        "adstock_params": {
            "true_lambda_decay": {
                "TV_CTV": 0.25,
                "OOH": 0.10,
                "Google": 0.05,
                "Meta": 0.15,
                "TikTok": 0.12,
            },
            "alpha_saturation": {
                "TV_CTV": 3.0,
                "OOH": 2.5,
                "Google": 1.5,
                "Meta": 2.0,
                "TikTok": 2.0,
            },
            "gamma_saturation": {
                "TV_CTV": 0.30,
                "OOH": 0.35,
                "Google": 0.50,
                "Meta": 0.40,
                "TikTok": 0.40,
            },
        },
        "output_params": {"aggregation_level": "weekly"},
    }

    sim = Simulate(random_seed=base_seed)
    final_df, channel_roi = sim.run_with_config(cfg)

    # Trim the final week if terminal sales drop too sharply
    if len(final_df) >= 2:
        last_sales = float(final_df["total_revenue"].iloc[-1])
        prev_sales = float(final_df["total_revenue"].iloc[-2])
        if prev_sales > 0 and last_sales < 0.5 * prev_sales:
            final_df = final_df.iloc[:-1].copy()
            print("Trimmed the final week due to >50% week-over-week sales drop.")

    # Normalize to snake_case
    df = final_df.reset_index()
    rename_map = {
        "TV_CTV_impressions": "tv_ctv_impressions",
        "OOH_impressions": "ooh_impressions",
        "Meta_impressions": "meta_impressions",
        "TikTok_impressions": "tiktok_impressions",
        "Google_clicks": "google_clicks",
        "TV_CTV_spend": "tv_ctv_spend",
        "OOH_spend": "ooh_spend",
        "Meta_spend": "meta_spend",
        "TikTok_spend": "tiktok_spend",
        "Google_spend": "google_spend",
        "total_revenue": "sales",
    }
    df = df.rename(columns=rename_map)
    df["sales"] = df["sales"].clip(lower=1e-6)

    # Time and geo
    df["time"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df["geo"] = "national"

    spend_cols = [
        "tv_ctv_spend",
        "ooh_spend",
        "google_spend",
        "meta_spend",
        "tiktok_spend",
    ]
    exposure_cols = [
        "tv_ctv_impressions",
        "ooh_impressions",
        "google_clicks",
        "meta_impressions",
        "tiktok_impressions",
    ]

    # Baseline estimation from original sales and ROI
    roi_map = {
        "tv_ctv_spend": channel_roi["TV_CTV"],
        "ooh_spend": channel_roi["OOH"],
        "google_spend": channel_roi["Google"],
        "meta_spend": channel_roi["Meta"],
        "tiktok_spend": channel_roi["TikTok"],
    }
    incremental = sum(df[col] * roi_map[col] for col in spend_cols)
    baseline = df["sales"] - incremental
    baseline = baseline.clip(lower=0.0)

    # Add annual seasonality to baseline
    week_index = np.arange(len(df))
    sin_term = np.sin(2 * np.pi * week_index / 52.0)
    cos_term = np.cos(2 * np.pi * week_index / 52.0)
    seasonality = 1.0 + 0.08 * sin_term + 0.04 * cos_term
    baseline = (baseline * seasonality * BASELINE_SCALE).clip(lower=0.0)

    # Keep the baseline mostly explainable by the controls the notebook supplies.
    #
    # Media contribution is smooth once adstock and saturation are applied, so its
    # week-to-week variation is small. If the baseline carries a large movement
    # that no control explains, that movement is the loudest thing in the series
    # and any estimator -- OLS with the true basis included -- loads it onto media.
    # On this panel the unmodelled baseline swing is about twice the media swing,
    # which inflates every channel's ROI roughly threefold. Shrinking the residual
    # to a quarter puts the demo back in the regime where the media coefficients
    # are identified, and matches the identifiability the linear version happened
    # to have. Production data is noisier; sec6.2 says so.
    _t = np.arange(len(df), dtype=float)
    _doy = df["week_start"].dt.dayofyear.to_numpy(dtype=float)
    _control_basis = np.column_stack([
        np.ones(len(df)),
        _t, _t ** 2,
        np.sin(2 * np.pi * _doy / 365.25), np.cos(2 * np.pi * _doy / 365.25),
        ((df["week_start"].dt.month == 11) & df["week_start"].dt.day.between(22, 28)).to_numpy(float),
        ((df["week_start"].dt.month == 12) & df["week_start"].dt.day.between(22, 28)).to_numpy(float),
    ])
    _base = baseline.to_numpy(dtype=float)
    _coef, *_ = np.linalg.lstsq(_control_basis, _base, rcond=None)
    _explained = _control_basis @ _coef
    baseline = pd.Series(
        np.clip(_explained + BASELINE_RESIDUAL_KEPT * (_base - _explained), 0.0, None),
        index=baseline.index,
    )

    # Exposure per spend ratios
    ratios = {
        "tv_ctv_impressions": _median_ratio(df, "tv_ctv_impressions", "tv_ctv_spend"),
        "ooh_impressions": _median_ratio(df, "ooh_impressions", "ooh_spend"),
        "google_clicks": _median_ratio(df, "google_clicks", "google_spend"),
        "meta_impressions": _median_ratio(df, "meta_impressions", "meta_spend"),
        "tiktok_impressions": _median_ratio(df, "tiktok_impressions", "tiktok_spend"),
    }

    total_spend = df[spend_cols].sum(axis=1).to_numpy()

    # Break the shared trend between the media budget and the baseline. The
    # simulator drives both from the same upward trend, which leaves
    # corr(baseline, total spend) around 0.8: media and baseline then explain the
    # same movement and the model cannot tell them apart. Orthogonalising total
    # spend against the baseline's own basis (level, trend, trend^2, annual
    # seasonality) keeps the total budget and the week-to-week lumpiness while
    # removing the part that is collinear with the baseline.
    _t = np.arange(len(df), dtype=float) / max(len(df) - 1, 1)
    _basis = np.column_stack([
        np.ones_like(_t), _t, _t ** 2,
        np.sin(2 * np.pi * _t * len(df) / 52.0), np.cos(2 * np.pi * _t * len(df) / 52.0),
    ])
    _coef, *_ = np.linalg.lstsq(_basis, total_spend, rcond=None)
    _resid = total_spend - _basis @ _coef
    total_spend = np.clip(total_spend.mean() + _resid, total_spend.min() * 0.5, None)
    total_spend = total_spend * (df[spend_cols].sum(axis=1).to_numpy().sum() / total_spend.sum())


    qc_metrics = None
    last_warnings: list[str] = []

    for attempt in range(max_retries):
        rng = np.random.default_rng(base_seed + attempt * 1000)
        weights = _generate_segment_mix_weights(
            n_weeks=len(df),
            n_channels=len(spend_cols),
            rng=rng,
        )
        if SEARCH_DEMAND_COUPLING > 0.0:
            # Search budget follows demand: more queries in season, more clicks, more
            # spend. Tilt the search share by the baseline itself, then renormalise so
            # the weekly total is untouched -- money moves between channels, not in.
            _google = spend_cols.index("google_spend")
            _demand = baseline.to_numpy(dtype=float)
            _demand = (_demand - _demand.mean()) / _demand.std()
            _tilt = np.exp(SEARCH_DEMAND_COUPLING * _demand)
            weights = weights.copy()
            weights[:, _google] *= _tilt
            _row_totals = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, _row_totals, out=np.zeros_like(weights),
                                where=_row_totals > 0)
        new_spend = weights * total_spend[:, None]
        spend_df = pd.DataFrame(new_spend, columns=spend_cols)
        qc_metrics = compute_identifiability_metrics(spend_df)
        passed, warnings = check_identifiability_qc(
            qc_metrics,
            off_rate_min=off_rate_min,
            off_rate_max=off_rate_max,
            share_std_min=share_std_min,
            corr_max=corr_max,
        )
        last_warnings = warnings
        if passed:
            print(f"QC passed on attempt {attempt + 1}/{max_retries}.")
            break
        print(f"QC failed on attempt {attempt + 1}/{max_retries}: {warnings}")

    if qc_metrics is None:
        raise RuntimeError("QC metrics not computed.")

    if last_warnings:
        print("WARNING: QC thresholds not fully met after retries:")
        for w in last_warnings:
            print(f"  - {w}")

    if SEARCH_SPEND_FLATNESS > 0.0:
        # Flatten the search budget toward a constant weekly amount and give the
        # difference to the other channels in proportion, so each week's total
        # budget is unchanged.
        _google = spend_cols.index("google_spend")
        _search = new_spend[:, _google]
        _flat = (1.0 - SEARCH_SPEND_FLATNESS) * _search + SEARCH_SPEND_FLATNESS * _search.mean()
        _delta = _search - _flat
        _others = np.delete(np.arange(new_spend.shape[1]), _google)
        _other_totals = new_spend[:, _others].sum(axis=1)
        _shares = np.divide(new_spend[:, _others], _other_totals[:, None],
                            out=np.zeros_like(new_spend[:, _others]), where=_other_totals[:, None] > 0)
        new_spend = new_spend.copy()
        new_spend[:, _google] = _flat
        new_spend[:, _others] += _shares * _delta[:, None]
        new_spend = np.clip(new_spend, 0.0, None)

    # Recompute the identifiability metrics on the spend matrix that actually
    # ships. The flattening above changes every channel's OFF rate and share
    # variance, so the metrics computed before it describe a matrix no consumer
    # ever sees. Google Search is expected to fail the variation check by design;
    # the QC print says so rather than hiding it behind a stale average.
    qc_metrics = compute_identifiability_metrics(
        pd.DataFrame(new_spend, columns=spend_cols)
    )

    # Apply new spend
    for i, col in enumerate(spend_cols):
        df[col] = new_spend[:, i]

    # Update exposures based on spend ratios
    exposure_noise_std = 0.05
    rng = np.random.default_rng(base_seed + 999)
    for exp_col, spend_col in zip(exposure_cols, spend_cols):
        ratio = ratios[exp_col]
        noise = rng.lognormal(mean=0.0, sigma=exposure_noise_std, size=len(df))
        df[exp_col] = df[spend_col] * ratio * noise

    # Recompute sales through adstock + Hill saturation, so the dataset has the
    # response curves that Chapters 6.2 and 6.4 talk about. The previous version
    # of this block added `spend * true_roi` directly, which made the true
    # marginal ROI identical to the true average ROI and left the data with no
    # carryover at all.
    curve_params: dict[str, tuple[float, float, float]] = {}
    incremental_new = np.zeros(len(df), dtype=float)
    for ch, spend_col, exp_col in zip(CHANNELS, spend_cols, exposure_cols):
        decay = 0.5 ** (1.0 / TRUE_HALF_LIFE_WEEKS[ch])
        exposure = df[exp_col].to_numpy(dtype=float)
        spend_series = df[spend_col].to_numpy(dtype=float)
        if ch == "Google":
            beta, half_sat = solve_response_curve(
                exposure, spend_series, decay, TRUE_AVERAGE_ROI[ch],
                GOOGLE_PULLBACK_ROAS, _pullback_roas,
            )
        else:
            beta, half_sat = solve_response_curve(
                exposure, spend_series, decay, TRUE_AVERAGE_ROI[ch],
                TRUE_MARGINAL_ROI[ch], _mroi_1pct,
            )
        curve_params[ch] = (decay, beta, half_sat)
        incremental_new = incremental_new + hill_response(exposure, decay, beta, half_sat)

    clean_sales = (baseline + incremental_new).to_numpy(dtype=float)
    noise_var = 0.0 if OBSERVATION_NOISE_R2 >= 1.0 else (
        clean_sales.var() * (1.0 - OBSERVATION_NOISE_R2) / OBSERVATION_NOISE_R2)
    noise = np.random.default_rng(base_seed + 4242).normal(
        0.0, float(np.sqrt(noise_var)), size=len(clean_sales)
    )
    df["sales"] = np.clip(clean_sales + noise, 1e-6, None)

    # Synthetic controls
    day_of_year = df["week_start"].dt.dayofyear
    df["seas_sin52"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["seas_cos52"] = np.cos(2 * np.pi * day_of_year / 365.25)

    df["hldy_thanksgiving"] = (
        (df["week_start"].dt.month == 11)
        & (df["week_start"].dt.day >= 22)
        & (df["week_start"].dt.day <= 28)
    ).astype(int)
    df["hldy_christmas"] = (
        (df["week_start"].dt.month == 12)
        & (df["week_start"].dt.day >= 22)
        & (df["week_start"].dt.day <= 28)
    ).astype(int)

    # Trend controls
    df["trend"] = np.arange(len(df), dtype=float)
    df["trend_sq"] = df["trend"] ** 2

    output_cols = [
        "time",
        "geo",
        "tv_ctv_impressions",
        "ooh_impressions",
        "meta_impressions",
        "tiktok_impressions",
        "google_clicks",
        "tv_ctv_spend",
        "ooh_spend",
        "meta_spend",
        "tiktok_spend",
        "google_spend",
        "sales",
        "seas_sin52",
        "seas_cos52",
        "hldy_thanksgiving",
        "hldy_christmas",
        "trend",
        "trend_sq",
    ]

    df = df[output_cols]

    # Ground truth response-curve table. `true_roi` stays the average ROI so the
    # existing data contract holds; the marginal columns are what Chapter 6.4
    # calibrates against.
    roi_rows = []
    for ch, spend_col, exp_col in zip(CHANNELS, spend_cols, exposure_cols):
        decay, beta, half_sat = curve_params[ch]
        exposure = df[exp_col].to_numpy(dtype=float)
        spend_series = df[spend_col].to_numpy(dtype=float)
        revenue = hill_response(exposure, decay, beta, half_sat)
        roi_rows.append({
            "channel": CHANNEL_DISPLAY[ch],
            "true_roi": round(float(revenue.sum() / spend_series.sum()), 4),
            "true_marginal_roi": round(_mroi_1pct(exposure, spend_series, decay, half_sat) * beta, 4),
            "true_pullback_roas_8w": round(_pullback_roas(exposure, spend_series, decay, half_sat) * beta, 4),
            "adstock_half_life_weeks": TRUE_HALF_LIFE_WEEKS[ch],
            "adstock_decay": round(decay, 4),
            "hill_half_saturation": round(half_sat, 2),
        })
    truth_df = pd.DataFrame(roi_rows)
    # ROI sanity check: require positive ROI for all channels
    if (truth_df["true_roi"] <= 0).any():
        bad = truth_df.loc[truth_df["true_roi"] <= 0, "channel"].tolist()
        raise ValueError(f"Negative ROI detected for channels: {bad}")


    return df, truth_df, qc_metrics


def main() -> None:
    df, truth_df, qc_metrics = generate_dataset()

    from msbook.paths import DATA_DIR

    data_dir = DATA_DIR / "generated" / "part6" / "sec6.2-mmm"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "mmm_synthetic_data.csv"
    truth_path = data_dir / "mmm_ground_truth_roi.csv"

    df.to_csv(data_path, index=False)
    truth_df.to_csv(truth_path, index=False)

    print(f"Saved {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"Saved {truth_path} ({truth_df.shape[0]} rows)")

    overall = qc_metrics["overall"]
    print("\n[QC Metrics on the final spend matrix]")
    print(f"OFF rate mean: {overall['off_rate_mean']:.1%}")
    print(f"Share std mean: {overall['share_std_mean']:.3f}")
    print(f"Channel corr max: {overall['channel_corr_max']:.3f}")
    search_share_std = overall["share_stds"]["google_spend"]
    search_off_rate = overall["off_rates"]["google_spend"]
    print(
        f"Google Search, by design: OFF rate {search_off_rate:.1%}, "
        f"share std {search_share_std:.3f} — an always-on flat budget, which is "
        f"the channel sec6.3 and sec6.4 exist to measure. The other four carry "
        f"the variation the model learns from."
    )


if __name__ == "__main__":
    main()
