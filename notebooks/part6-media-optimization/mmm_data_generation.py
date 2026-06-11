"""
MMM synthetic data generation with identifiability QC.

- Generates base data via PySiMMMulator.
- Applies a segment-mix style post-process to introduce OFF weeks,
  larger share variation, and lower channel correlations.
- Recomputes exposures and sales to keep spend/ROI consistency.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pysimmmulator import Simulate


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
                # Adjusted to a realistic ROI range (1-10)
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
    baseline = (baseline * seasonality).clip(lower=0.0)

    # Exposure per spend ratios
    ratios = {
        "tv_ctv_impressions": _median_ratio(df, "tv_ctv_impressions", "tv_ctv_spend"),
        "ooh_impressions": _median_ratio(df, "ooh_impressions", "ooh_spend"),
        "google_clicks": _median_ratio(df, "google_clicks", "google_spend"),
        "meta_impressions": _median_ratio(df, "meta_impressions", "meta_spend"),
        "tiktok_impressions": _median_ratio(df, "tiktok_impressions", "tiktok_spend"),
    }

    total_spend = df[spend_cols].sum(axis=1).to_numpy()

    qc_metrics = None
    last_warnings: list[str] = []

    for attempt in range(max_retries):
        rng = np.random.default_rng(base_seed + attempt * 1000)
        weights = _generate_segment_mix_weights(
            n_weeks=len(df),
            n_channels=len(spend_cols),
            rng=rng,
        )
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

    # Recompute sales
    incremental_new = sum(df[col] * roi_map[col] for col in spend_cols)
    df["sales"] = (baseline + incremental_new).clip(lower=1e-6)

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

    # Ground truth ROI table
    roi_rows = []
    for ch in CHANNELS:
        roi_rows.append({"channel": CHANNEL_DISPLAY[ch], "true_roi": round(channel_roi[ch], 4)})
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
    print("\n[QC Metrics]")
    print(f"OFF rate mean: {overall['off_rate_mean']:.1%}")
    print(f"Share std mean: {overall['share_std_mean']:.3f}")
    print(f"Channel corr max: {overall['channel_corr_max']:.3f}")


if __name__ == "__main__":
    main()
