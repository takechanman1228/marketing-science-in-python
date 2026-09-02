"""Shared helpers for the Chapter 5.3 sales-forecasting workflow.

Numeric building blocks used by the sec5.3 notebook and its tests:
point-forecast and interval metrics plus rolling forecast origins.
Chapter-specific data assembly, simulation, and plotting stay in the notebook.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |y - yhat| / sum |y|."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denominator = np.abs(y).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y - yhat).sum() / denominator)


def normalized_bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed forecast bias: sum (yhat - y) / sum |y|.

    Positive values mean persistent over-forecast. For the non-negative
    sales series used in the book, the |y| denominator equals sum(y).
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denominator = np.abs(y).sum()
    if denominator == 0:
        return float("nan")
    return float((yhat - y).sum() / denominator)


def total_window_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed error of the WINDOW TOTAL: (sum yhat - sum y) / sum y.

    The metric the year-end question depends on — weekly misses partly
    cancel inside a multi-week sum, so this can be far smaller (or, with a
    biased model, exactly as bad) as the weekly error.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denominator = y.sum()
    if denominator == 0:
        return float("nan")
    return float((yhat.sum() - y.sum()) / denominator)


def pinball_loss(y: np.ndarray, yhat: np.ndarray, quantile: float) -> float:
    """Mean pinball (quantile) loss for one quantile forecast."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    error = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def interval_coverage(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Share of actuals inside [lower, upper]."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))


def quantile_crossing_rate(
    pred: pd.DataFrame,
    cols: Sequence[str] = ("p10", "p50", "p90"),
) -> float:
    """Share of rows where the quantile columns are not monotone."""
    lo, mid, hi = (pred[c].to_numpy(dtype=float) for c in cols)
    crossing = (lo > mid) | (mid > hi)
    return float(np.mean(crossing))


def rearrange_quantiles(
    pred: pd.DataFrame,
    cols: Sequence[str] = ("p10", "p50", "p90"),
) -> pd.DataFrame:
    """Row-wise sort of the quantile columns (display fix, not calibration)."""
    out = pred.copy()
    ordered = np.sort(out[list(cols)].to_numpy(dtype=float), axis=1)
    out[list(cols)] = ordered
    return out


# ---------------------------------------------------------------------------
# Rolling forecast origins
# ---------------------------------------------------------------------------

def make_forecast_origins(
    dates: pd.Series,
    *,
    min_train_weeks: int,
    horizon: int,
    step: int,
) -> list[pd.Timestamp]:
    """Evenly spaced forecast origins over the unique dates in ``dates``.

    Each returned origin has at least ``min_train_weeks`` observations up to
    and including itself, and at least ``horizon`` observations after it.
    ``step == horizon`` yields non-overlapping evaluation windows.
    """
    unique_dates = pd.Index(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    first_index = min_train_weeks - 1
    last_index = len(unique_dates) - horizon - 1
    if last_index < first_index:
        raise ValueError(
            "Not enough history for the requested training window and "
            "forecast horizon."
        )
    return [
        pd.Timestamp(unique_dates[i])
        for i in range(first_index, last_index + 1, step)
    ]
