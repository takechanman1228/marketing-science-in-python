# Data Profiles

Machine-readable summaries of the retail open datasets used across this book.

## Purpose

- **Quick reference** for humans and AI assistants when writing new chapters
- **Schema documentation** so notebooks don't need to rediscover column types
- **Cross-chapter index** showing which datasets are used where

## Files

| Profile | Dataset | Size | Primary chapters |
|---------|---------|------|-----------------|
| `dunnhumby.md` | Dunnhumby "The Complete Journey" | ~810 MB (8 CSVs) | sec4.1 (traditional), 4.2, 5.1, 5.2, 5.3 |
| `online_retail_ii.md` | Online Retail II (UCI/Kaggle) | ~90 MB | sec4.1 (semantic) |
| `iowa_liquor_sales.md` | Iowa Liquor Sales (Kaggle CC0) | ~3.2 GB | sec5.3 |

## Loading Data

Use the shared loader rather than writing raw `pd.read_csv` calls. From any
notebook (the `mdsbook` package resolves the repo root regardless of cwd):

```python
from msbook.data import load_dunnhumby, load_online_retail_ii
```

For chapters that call the `notebooks/_retail_data.py` module directly
(`load_dunnhumby_transactions`, `load_online_retail_ii`, etc.), the loader
reads from `data/raw/<dataset>/`.

See `src/msbook/data.py` and `notebooks/_retail_data.py` for all available
loader functions.

## Companion EDA Notebooks

Exploratory EDA notebooks for these datasets were retired to
`notebooks/.archive/retail-open-data-eda/`.
