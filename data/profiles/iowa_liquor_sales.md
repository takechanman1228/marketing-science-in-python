---
name: Iowa Liquor Sales
source: Iowa Alcoholic Beverages Division (via Kaggle)
url: https://www.kaggle.com/datasets/residentmario/iowa-liquor-sales
license: CC0 (Public Domain)
domain: Government liquor sales records (State of Iowa)
format: CSV
total_size: 3.47 GB
date_range: "~2012 to 2024 (ongoing public dataset)"
chapters_used:
  - sec6.5  # Demand Forecasting — VODKA 80 PROOF subset for Nixtla/TimeGPT
columns:
  - name: Invoice/Item Number
    dtype: string
    description: Unique identifier for each invoice line item.
  - name: Date
    dtype: string
    description: Transaction date. Requires pd.to_datetime(format="mixed") to parse.
  - name: Store Number
    dtype: int
    description: Numeric identifier for the retail store.
  - name: Store Name
    dtype: string
    description: Name of the retail store.
  - name: Address
    dtype: string
    description: Street address of the store. Dropped during cleaning.
  - name: City
    dtype: string
    description: City where the store is located.
  - name: Zip Code
    dtype: string
    description: ZIP code of the store location.
  - name: Store Location
    dtype: string
    description: Multi-line string containing full address with embedded newlines. Dropped during cleaning.
  - name: County Number
    dtype: float
    description: Numeric code for the county. Dropped during cleaning.
  - name: County
    dtype: string
    description: Name of the county.
  - name: Category
    dtype: float
    description: Numeric category code for the product type.
  - name: Category Name
    dtype: string
    description: Human-readable label for the product category (e.g. "VODKA 80 PROOF").
  - name: Vendor Number
    dtype: int
    description: Numeric identifier for the vendor/supplier. Dropped during cleaning.
  - name: Vendor Name
    dtype: string
    description: Name of the vendor/supplier.
  - name: Item Number
    dtype: int
    description: Numeric product identifier (SKU-level).
  - name: Item Description
    dtype: string
    description: Human-readable product name/description.
  - name: Pack
    dtype: int
    description: Number of bottles per pack/case.
  - name: Bottle Volume (ml)
    dtype: int
    description: Volume of a single bottle in milliliters.
  - name: State Bottle Cost
    dtype: string
    description: Wholesale cost per bottle. String with "$" prefix — requires parsing.
  - name: State Bottle Retail
    dtype: string
    description: Suggested retail price per bottle. String with "$" prefix — requires parsing.
  - name: Bottles Sold
    dtype: int
    description: Number of bottles sold in the transaction. Can be negative for returns.
  - name: Sale (Dollars)
    dtype: string
    description: Total sale amount. String with "$" prefix — requires parsing.
  - name: Volume Sold (Liters)
    dtype: float
    description: Total volume sold in liters.
  - name: Volume Sold (Gallons)
    dtype: float
    description: Total volume sold in gallons.
---

# Iowa Liquor Sales

## Overview

The Iowa Liquor Sales dataset contains every wholesale purchase of liquor in the
state of Iowa by retailers, as recorded by the Iowa Alcoholic Beverages Division.
The dataset is published on Kaggle under a CC0 (public domain) license and spans
roughly 2012 to 2024, with new data added on an ongoing basis.

The single CSV file (`Iowa_Liquor_Sales.csv`) resides at:

```
data/raw/Iowa_Liquor_Sales_Data_kaggle_cc0/Iowa_Liquor_Sales.csv
```

At 3.47 GB, it is the largest single file used in the book and must be handled
with chunked-reading strategies.


## Schema

The dataset has **24 columns** covering five groups of information:

| Group              | Columns                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Transaction        | Invoice/Item Number, Date, Bottles Sold, Sale (Dollars)                 |
| Store              | Store Number, Store Name, Address, City, Zip Code, Store Location       |
| Geography          | County Number, County                                                   |
| Product            | Category, Category Name, Vendor Number, Vendor Name, Item Number, Item Description, Pack, Bottle Volume (ml) |
| Pricing / Volume   | State Bottle Cost, State Bottle Retail, Volume Sold (Liters), Volume Sold (Gallons) |

**Type caveats:**

- **Date** is stored as a string. Parse with `pd.to_datetime(format="mixed")`.
- **State Bottle Cost**, **State Bottle Retail**, and **Sale (Dollars)** are
  strings prefixed with `"$"` and may contain commas. Strip `"$,"` with a regex
  before casting to `float`.
- **Store Location** contains multi-line strings (embedded `\n` in the CSV),
  which can break naive parsers. This column is dropped during cleaning.
- **Bottles Sold** is an integer that can be negative (returns/adjustments).


## Performance Notes

Because the file is 3.47 GB, loading it into memory in one pass is impractical
on many machines. The recommended strategy is **chunked reading**:

```python
import pandas as pd

chunks = pd.read_csv(
    path,
    chunksize=500_000,
    usecols=[...],          # select only needed columns
    dtype={"Zip Code": str} # keep leading zeros
)

frames = []
for chunk in chunks:
    # apply filters and transforms per chunk
    frames.append(chunk)

df = pd.concat(frames, ignore_index=True)
```

Selecting only the columns you need via `usecols` dramatically reduces memory
pressure. For the demand-forecasting use case (sec6.5), the working subset after
filtering to VODKA 80 PROOF and aggregating weekly is orders of magnitude smaller
than the raw file.


## Cleaning Pipeline

The loader performs the following steps in order:

1. **Column selection** -- Drop `Address`, `Store Location`, `County Number`, and
   `Vendor Number` (not needed for analysis, and `Store Location` causes parsing
   issues).

2. **Chunked reading** -- Read with `chunksize=500_000` to limit peak memory.

3. **Dollar-column parsing** -- For `State Bottle Cost`, `State Bottle Retail`,
   and `Sale (Dollars)`:
   ```python
   col.str.replace(r"[$,]", "", regex=True).astype(float)
   ```

4. **Date parsing** -- Convert the `Date` column:
   ```python
   pd.to_datetime(df["Date"], format="mixed")
   ```

5. **Category filtering** (optional) -- Subset to a single category such as
   `"VODKA 80 PROOF"` to create a manageable analytical dataset.

6. **Row filtering** -- Drop rows where `Bottles Sold <= 0` or
   `Sale (Dollars) <= 0` to remove returns and zero-value records.


## Key Statistics

| Metric                           | Value                    |
|----------------------------------|--------------------------|
| Raw file size                    | 3.47 GB                  |
| Total columns                    | 24                       |
| Columns retained after cleaning  | 20                       |
| Date range                       | ~2012 -- 2024            |
| Chunk size used for reading      | 500,000 rows             |
| VODKA 80 PROOF weekly time series| ~1,900 (Item x Store)    |


## Data Quality Notes

- **Dollar formatting**: The three dollar-valued columns are stored as strings
  with a leading `$` sign. Forgetting to strip and cast will produce downstream
  errors in any numeric operation.
- **Multi-line Store Location**: The `Store Location` field contains embedded
  newlines, which can corrupt row boundaries in parsers that do not handle quoted
  multi-line fields. Safest to drop this column early.
- **Negative quantities**: Some rows have negative `Bottles Sold`, representing
  returns or adjustments. These are filtered out for forecasting tasks.
- **File size**: At 3.47 GB the dataset cannot be loaded naively on machines with
  limited RAM. Always use chunked reading or column subsetting.
- **Ongoing updates**: The public dataset is periodically refreshed, so row
  counts may differ across downloads.


## Insights for Downstream Tasks

### sec6.5 -- Demand Forecasting

The primary use of this dataset in the book is for demand forecasting with
Nixtla / TimeGPT (sec6.5). The pipeline:

1. Filter to a single product category (`"VODKA 80 PROOF"`).
2. Aggregate to **weekly** granularity by `(Item Number, Store Number)`.
3. This produces approximately **1,900 individual time series**, each
   representing weekly sales of a specific vodka SKU at a specific store.
4. The resulting panel dataset feeds directly into Nixtla's TimeGPT API for
   zero-shot and fine-tuned probabilistic forecasts.

The weekly aggregation smooths out day-of-week effects while preserving enough
temporal resolution to capture seasonal patterns (holidays, year-end spikes).
Practitioners should watch for sparse series (stores that sell a given SKU
infrequently) and consider minimum-history thresholds before forecasting.
