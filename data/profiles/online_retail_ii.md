---
name: "Online Retail II"
source: UCI Machine Learning Repository / Kaggle
url: https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset
license: CC0 (Public Domain)
domain: UK-based online gift/novelty retailer
format: CSV (1 file)
total_size: "~94.9 MB"
rows: 1067371
columns:
  - name: Invoice
    dtype: int/str
    nullable: false
    description: "Unique invoice number; prefix 'C' indicates a cancellation"
  - name: StockCode
    dtype: str
    nullable: false
    description: "Product code (5-digit with optional letter variants)"
  - name: Description
    dtype: str
    nullable: true
    description: "Product name; some nulls"
  - name: Quantity
    dtype: int
    nullable: false
    description: "Quantity per line item; negative values indicate cancellations"
  - name: InvoiceDate
    dtype: datetime
    nullable: false
    description: "Transaction timestamp"
  - name: Price
    dtype: float
    nullable: false
    description: "Unit price in GBP"
  - name: Customer ID
    dtype: float
    nullable: true
    description: "Customer identifier; nullable — cast to int then str after dropping nulls"
  - name: Country
    dtype: str
    nullable: false
    description: "Customer's country of residence"

date_range: "Dec 2009 – Dec 2011"

chapters_used:
  - sec4.2  # BERTopic Segmentation (product descriptions for topic modeling)
  - sec5.4  # Hybrid Recommendations (user-item purchase history + product text features)
---

# Online Retail II

## Overview

The Online Retail II dataset contains all transactions for a UK-based,
non-store online retailer specializing in unique all-occasion gifts and
novelty items. The single CSV file (`online_retail_II.csv`) resides in
`data/raw/Online_Retail_II_UCI/` and weighs
approximately 94.9 MB. It covers the period from December 2009 to
December 2011 and includes 1,067,371 rows across 8 columns.

Approximately 85% of transactions originate from the United Kingdom, with
the remaining 15% spread across 40+ countries. A significant portion of
records (roughly 25%) lack a Customer ID, and about 2% of invoices
represent cancellations (identified by an 'C' prefix on the Invoice
field).

## Schema

| Column | Type | Nullable | Description |
|---|---|---|---|
| Invoice | int/str | No | Unique invoice number; prefix 'C' = cancellation |
| StockCode | str | No | Product code (5-digit + letter variants) |
| Description | str | Yes | Product name; some nulls |
| Quantity | int | No | Quantity per line item; negative for cancellations |
| InvoiceDate | datetime | No | Transaction timestamp |
| Price | float | No | Unit price in GBP |
| Customer ID | float | Yes | Customer identifier; cast to int then str after cleaning |
| Country | str | No | Customer's country of residence |

## Cleaning Pipeline

The loader function supports a `clean=True` flag that applies the
following steps in order:

1. **Drop cancellations.** Remove rows where `Invoice` starts with 'C'.
2. **Drop null Customer IDs.** Remove rows where `Customer ID` is NaN.
3. **Cast Customer ID.** Convert from float to int, then to str for
   consistent categorical handling.
4. **Drop non-positive quantities.** Remove rows where `Quantity <= 0`.
5. **Drop non-positive prices.** Remove rows where `Price <= 0`.
6. **Add Revenue column.** Compute `Revenue = Quantity * Price`.

After cleaning, approximately 800,000 rows remain.

## Key Statistics

| Metric | Raw | After cleaning |
|---|---:|---:|
| Rows | 1,067,371 | ~800,000 |
| Unique invoices | — | ~22,000 |
| Unique customers | — | ~4,300 |
| Unique products (StockCode) | — | ~3,900 |
| Countries | 43 | ~38 |
| Date range | Dec 2009 – Dec 2011 | Dec 2009 – Dec 2011 |

- **Geography:** ~85% of transactions from the United Kingdom.
- **File size:** 94.9 MB (single CSV).
- **Currency:** All prices in GBP.

## Data Quality Notes

1. **Missing Customer IDs.** Approximately 25% of rows lack a Customer
   ID. These must be dropped before any customer-level analysis
   (segmentation, CLV, recommendation). The loader's `clean=True` flag
   handles this automatically.

2. **Cancellation rows.** About 2% of invoices carry a 'C' prefix,
   indicating cancelled orders. These have negative Quantity values and
   must be removed for revenue and purchase-frequency analyses.

3. **Extreme Quantity values.** Some rows contain very large Quantity
   values representing bulk or wholesale orders. Consider capping or
   flagging outliers depending on the analysis context.

4. **Zero-price entries.** A small number of rows have `Price = 0`,
   corresponding to internal adjustments, bad-debt entries, or sample
   items. These are removed by the cleaning pipeline.

5. **Non-UK transactions.** While the retailer is UK-based, roughly 15%
   of revenue comes from international customers across 40+ countries.
   Country-level filtering may be needed for analyses that assume a
   single market.

## Insights for Downstream Tasks

### sec4.2 — BERTopic Segmentation

The `Description` column provides free-text product names suitable for
topic modeling with BERTopic. After deduplication at the `StockCode`
level, product descriptions can be embedded and clustered to discover
latent product categories that go beyond the retailer's internal stock
code hierarchy. These topics serve as features for customer segmentation
when combined with purchase history.

### sec5.4 — Hybrid Recommendations

The cleaned dataset provides a natural user-item interaction matrix
(`Customer ID` x `StockCode`) based on purchase events. Implicit
feedback signals include purchase frequency and total revenue per
customer-product pair. The `Description` text features enable
content-based filtering that complements collaborative signals, making
this dataset well-suited for hybrid recommendation approaches that fuse
behavioral and textual information.
