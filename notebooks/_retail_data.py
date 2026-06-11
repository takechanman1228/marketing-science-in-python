"""Shared data loaders for the 3 retail open datasets.

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
_IOWA_PATH = str(_RAW / "Iowa_Liquor_Sales_Data_kaggle_cc0" / "Iowa_Liquor_Sales.csv")


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
    data_dir : path to the ``archive/`` folder.
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


# ===================================================================
# Iowa Liquor Sales (Kaggle CC0, 1 CSV, ~3.2 GB)
# ===================================================================

def load_iowa_liquor(
    data_path: str | None = None,
    category_filter: str | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load Iowa_Liquor_Sales.csv (~37.8M rows, 3.2 GB) in chunks.

    Parameters
    ----------
    category_filter : e.g. ``"VODKA 80 PROOF"`` — keep only matching Category Name.
    usecols : columns to read; defaults to a useful subset.
    nrows : if set, stop after this many rows (before filtering).
    chunksize : rows per chunk.
    """
    p = data_path or _IOWA_PATH

    if usecols is None:
        usecols = [
            "Invoice/Item Number", "Date", "Store Number", "Store Name",
            "City", "Zip Code", "County", "Category", "Category Name",
            "Vendor Name", "Item Number", "Item Description",
            "Pack", "Bottle Volume (ml)",
            "State Bottle Cost", "State Bottle Retail",
            "Bottles Sold", "Sale (Dollars)",
            "Volume Sold (Liters)", "Volume Sold (Gallons)",
        ]

    frames: list[pd.DataFrame] = []
    rows_read = 0

    for chunk in pd.read_csv(
        p,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        rows_read += len(chunk)

        # Parse dollar-sign columns
        for col in ("State Bottle Cost", "State Bottle Retail", "Sale (Dollars)"):
            if col in chunk.columns:
                chunk[col] = (
                    chunk[col]
                    .astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .astype(float)
                )

        # Parse date
        if "Date" in chunk.columns:
            chunk["Date"] = pd.to_datetime(chunk["Date"], format="mixed")

        if category_filter is not None:
            chunk = chunk[chunk["Category Name"] == category_filter]

        if len(chunk):
            frames.append(chunk)

        if nrows is not None and rows_read >= nrows:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Drop non-positive sales
    if "Bottles Sold" in df.columns:
        df = df[(df["Bottles Sold"] > 0) & (df["Sale (Dollars)"] > 0)].copy()

    return df
