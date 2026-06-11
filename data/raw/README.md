# `data/raw/` — raw external datasets (gitignored)

Large datasets that are **not** redistributed with this repo. Each loader in
`msbook.data` (or `notebooks/_retail_data.py`) reads from here; a loader either
fetches on first call or you place the files manually per the table below.

This directory is committed (with `.gitkeep` and this README); the data files
inside are gitignored.

## Datasets used by the book

| Dataset | Loader / source | Expected path (under `data/raw/`) | Notes |
|---|---|---|---|
| Dunnhumby — *The Complete Journey* | `msbook.data.load_dunnhumby()` (kagglehub) | `Dunnhumby_kaggle/transaction_data.csv` (+ the other CSVs) | Requires `~/.kaggle/kaggle.json`; sec4.1/4.2/5.1/5.2/5.3 |
| Online Retail II (UCI) | `msbook.data.load_online_retail_ii()` (manual) | `Online_Retail_II_UCI/online_retail_II.csv` | ~95 MB, CC0; sec4.1 BERTopic |
| Iowa Liquor Sales | `_retail_data.load_iowa_liquor()` (manual) | `Iowa_Liquor_Sales_Data_kaggle_cc0/Iowa_Liquor_Sales.csv` | ~3.2 GB, CC0; only needed to regenerate sec5.3's committed parquet |
| MineThatData (Hillstrom) | `msbook.data.load_minethatdata()` (auto, HTTP) | `minethatdata/hillstrom_email.csv` | fetched from minethatdata.com; sec3.4/3.5 |

The Dunnhumby CSVs live **directly** under `Dunnhumby_kaggle/` (there is no
`archive/` sub-folder; that was just Kaggle's download artifact name).

## Kaggle credentials (one-time setup)

Loaders that pull from Kaggle (Dunnhumby) need an API token:

```bash
# 1. Create one at https://www.kaggle.com/settings → Account → API → Create New Token
# 2. Save the downloaded kaggle.json as:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Manual placement

For datasets without an auto-fetch loader, download from the source and place
the files at the **Expected path** above (relative to this `data/raw/` directory).
