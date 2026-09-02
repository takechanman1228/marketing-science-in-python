"""Shared data loaders for the two public retail datasets the book uses.

Usage from any notebook under notebooks/<topic>/:

    import sys; sys.path.insert(0, "..")
    from _retail_data import load_dunnhumby_transactions, load_online_retail_ii
"""

from __future__ import annotations

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Default paths — resolved via msbook so they work from any cwd. All raw
# datasets live under data/raw/ (gitignored; see data/raw/README.md).
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

try:
    from msbook.paths import DATA_DIR as _DATA_DIR
    _RAW = _DATA_DIR / "raw"
except ImportError:
    # msbook not installed — cwd-relative fallback for orphaned scripts.
    _RAW = _Path("../../data/raw")

_DUNNHUMBY_DIR = str(_RAW / "Dunnhumby_kaggle")
_ONLINE_RETAIL_PATH = str(_RAW / "Online_Retail_II_UCI" / "online_retail_II.csv")


# ===================================================================
# Dunnhumby — "The Complete Journey" (8 CSVs, ~810 MB)
# ===================================================================

def load_dunnhumby_transactions(
    data_dir: str | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load transaction_data.csv (2.6M rows, 12 columns).

    Parameters
    ----------
    data_dir : directory holding the Dunnhumby CSVs.
    usecols : subset of columns to read (saves memory).

    Returns
    -------
    DataFrame with original columns; no rows removed.
    """
    d = data_dir or _DUNNHUMBY_DIR
    return pd.read_csv(os.path.join(d, "transaction_data.csv"), usecols=usecols)


def load_dunnhumby_products(data_dir: str | None = None) -> pd.DataFrame:
    """Load product.csv (~92K rows)."""
    d = data_dir or _DUNNHUMBY_DIR
    return pd.read_csv(os.path.join(d, "product.csv"))


def load_dunnhumby_demographics(data_dir: str | None = None) -> pd.DataFrame:
    """Load hh_demographic.csv (801 households)."""
    d = data_dir or _DUNNHUMBY_DIR
    return pd.read_csv(os.path.join(d, "hh_demographic.csv"))


def load_dunnhumby_causal(
    data_dir: str | None = None,
    product_ids: list[int] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Load causal_data.csv (~39M rows, 664 MB) in chunks.

    Parameters
    ----------
    product_ids : if given, keep only rows matching these PRODUCT_IDs.
    chunksize : rows per chunk (default 1M).
    """
    d = data_dir or _DUNNHUMBY_DIR
    path = os.path.join(d, "causal_data.csv")
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        dtype={"display": str},
        chunksize=chunksize,
    ):
        if product_ids is not None:
            chunk = chunk[chunk["PRODUCT_ID"].isin(product_ids)]
        if len(chunk):
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_dunnhumby_campaigns(
    data_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load campaign & coupon tables as a dict.

    Returns
    -------
    dict with keys: ``campaign_desc``, ``campaign_table``, ``coupon``,
    ``coupon_redempt``.
    """
    d = data_dir or _DUNNHUMBY_DIR
    return {
        "campaign_desc": pd.read_csv(os.path.join(d, "campaign_desc.csv")),
        "campaign_table": pd.read_csv(os.path.join(d, "campaign_table.csv")),
        "coupon": pd.read_csv(os.path.join(d, "coupon.csv")),
        "coupon_redempt": pd.read_csv(os.path.join(d, "coupon_redempt.csv")),
    }


# ===================================================================
# Online Retail II (UCI / Kaggle, 1 CSV, ~90 MB)
# ===================================================================

def load_online_retail_ii(
    data_path: str | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Load online_retail_II.csv (~1.07M rows).

    Parameters
    ----------
    clean : if True (default), drop cancellations (Invoice starts with 'C'),
        drop rows with null Customer ID, cast Customer ID to int→str,
        drop rows with non-positive Quantity or Price, and add a Revenue column.
    """
    p = data_path or _ONLINE_RETAIL_PATH
    df = pd.read_csv(p, parse_dates=["InvoiceDate"])

    if clean:
        df = df[
            ~df["Invoice"].astype(str).str.startswith("C")
            & df["Customer ID"].notna()
        ].copy()
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
        df["Revenue"] = df["Quantity"] * df["Price"]

    return df
