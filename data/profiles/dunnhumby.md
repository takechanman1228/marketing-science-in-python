---
name: "Dunnhumby — The Complete Journey"
source: Dunnhumby via Kaggle
url: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey
license: Open (Kaggle)
domain: Grocery retail loyalty card
format: CSV (8 files)
total_size: "~847 MB"

tables:
  - name: transaction_data
    file: transaction_data.csv
    rows: 2595732
    columns:
      - household_key
      - BASKET_ID
      - DAY
      - PRODUCT_ID
      - QUANTITY
      - SALES_VALUE
      - STORE_ID
      - RETAIL_DISC
      - TRANS_TIME
      - WEEK_NO
      - COUPON_DISC
      - COUPON_MATCH_DISC
    join_keys: [household_key, PRODUCT_ID, STORE_ID]

  - name: hh_demographic
    file: hh_demographic.csv
    rows: 801
    columns:
      - AGE_DESC
      - MARITAL_STATUS_CODE
      - INCOME_DESC
      - HOMEOWNER_DESC
      - HH_COMP_DESC
      - HOUSEHOLD_SIZE_DESC
      - KID_CATEGORY_DESC
      - household_key
    join_keys: [household_key]

  - name: product
    file: product.csv
    rows: 92353
    columns:
      - PRODUCT_ID
      - MANUFACTURER
      - DEPARTMENT
      - BRAND
      - COMMODITY_DESC
      - SUB_COMMODITY_DESC
      - CURR_SIZE_OF_PRODUCT
    join_keys: [PRODUCT_ID]

  - name: causal_data
    file: causal_data.csv
    rows: 36786524
    columns:
      - PRODUCT_ID
      - STORE_ID
      - WEEK_NO
      - display
      - mailer
    join_keys: [PRODUCT_ID, STORE_ID, WEEK_NO]

  - name: campaign_desc
    file: campaign_desc.csv
    rows: 30
    columns:
      - DESCRIPTION
      - CAMPAIGN
      - START_DAY
      - END_DAY
    join_keys: [CAMPAIGN]

  - name: campaign_table
    file: campaign_table.csv
    rows: 7208
    columns:
      - DESCRIPTION
      - household_key
      - CAMPAIGN
    join_keys: [household_key, CAMPAIGN]

  - name: coupon
    file: coupon.csv
    rows: 124548
    columns:
      - COUPON_UPC
      - PRODUCT_ID
      - CAMPAIGN
    join_keys: [COUPON_UPC, PRODUCT_ID, CAMPAIGN]

  - name: coupon_redempt
    file: coupon_redempt.csv
    rows: 2318
    columns:
      - household_key
      - DAY
      - COUPON_UPC
      - CAMPAIGN
    join_keys: [household_key, COUPON_UPC, CAMPAIGN]

temporal_range:
  weeks: "1–102 (2 years)"
  days: "1–711"

chapters_used:
  - sec4.2  # Customer Segmentation
  - sec4.3  # CLV
  - sec5.3  # Collaborative Filtering
  - sec5.4  # Content-Based / Hybrid Recommendations
  - sec5.7  # Uplift-Based Personalization
  - sec6.2  # Price Elasticity
  - sec6.3  # Product Assortment
  - sec6.4  # Promotion Optimization
---

# Dunnhumby — The Complete Journey

## Overview

The Dunnhumby "Complete Journey" dataset captures two years of household-level
grocery transactions from a group of 2,500 loyalty-card holders across multiple
stores. It includes transaction line items, household demographics, product
attributes, in-store display and mailer promotions (causal data), and a full
campaign/coupon lifecycle (campaign descriptions, household targeting, coupon
definitions, and redemptions). The eight CSV files total approximately 847 MB and
reside in `data/raw/Dunnhumby_kaggle/`.

## Schema & Relationships

The dataset is organized around four entity types joined by natural keys:

| Join key | Links |
|---|---|
| `household_key` | transaction_data, hh_demographic, campaign_table, coupon_redempt |
| `PRODUCT_ID` | transaction_data, product, causal_data, coupon |
| `STORE_ID` | transaction_data, causal_data |
| `WEEK_NO` | transaction_data, causal_data |
| `CAMPAIGN` | campaign_desc, campaign_table, coupon, coupon_redempt |
| `COUPON_UPC` | coupon, coupon_redempt |

**Core transactional join:**
`transaction_data` is the fact table. It joins to `hh_demographic` on
`household_key`, to `product` on `PRODUCT_ID`, and to `causal_data` on the
composite key (`PRODUCT_ID`, `STORE_ID`, `WEEK_NO`).

**Campaign/coupon lifecycle:**
`campaign_desc` defines 30 campaigns with start/end days. `campaign_table`
records which households were targeted by each campaign. `coupon` maps coupon
UPCs to products and campaigns. `coupon_redempt` records actual redemption
events by household.

**Product hierarchy:**
Within `product`, the columns form a three-level hierarchy:
`DEPARTMENT` > `COMMODITY_DESC` > `SUB_COMMODITY_DESC`.

## Key Statistics

| Table | Rows | Columns | File size |
|---|---:|---:|---|
| transaction_data | 2,595,732 | 12 | 141.7 MB |
| hh_demographic | 801 | 8 | — |
| product | 92,353 | 7 | 6.4 MB |
| causal_data | 36,786,524 | 5 | 695.9 MB |
| campaign_desc | 30 | 4 | — |
| campaign_table | 7,208 | 3 | — |
| coupon | 124,548 | 3 | 2.8 MB |
| coupon_redempt | 2,318 | 4 | — |

- **Households:** 2,500 total; 801 with demographic records (32% coverage).
- **Products:** 92,353 unique PRODUCT_IDs.
- **Stores:** Multiple STORE_IDs (exact count varies by analysis).
- **Temporal span:** WEEK_NO 1–102 (2 years), DAY 1–711.

## Data Quality Notes

1. **Demographic coverage gap.** Only 801 of the 2,500 households have
   demographic records in `hh_demographic` (32%). Any analysis that requires
   demographics must handle the missing 68% explicitly (inner join or
   imputation).

2. **Mixed types in `causal_data.display`.** The `display` column contains mixed
   types and must be read with `dtype=str` (or `dtype={"display": str}`) to
   avoid pandas type-inference errors.

3. **Negative discount values.** `RETAIL_DISC` in `transaction_data` stores
   discount amounts as negative numbers (e.g., -1.50 means $1.50 off). Take the
   absolute value when computing discount depth.

4. **Missing commodity descriptions.** Some products carry the placeholder
   strings `"NO COMMODITY DESCRIPTION"` and/or `"NO SUBCOMMODITY DESCRIPTION"`.
   Filter or recode these before using the product hierarchy.

5. **TRANS_TIME encoding.** Transaction time is stored as a military-time
   integer (e.g., 1631 = 4:31 PM). Convert with
   `hour = TRANS_TIME // 100; minute = TRANS_TIME % 100`.

## Insights for Downstream Tasks

### sec4.2 — Customer Segmentation

Build RFM (Recency, Frequency, Monetary) features from `transaction_data`
aggregated at the `household_key` level. Enrich segments with
`hh_demographic` for households where demographics are available. Use
`WEEK_NO` and `DAY` to compute recency relative to the observation window
end.

### sec4.3 — Customer Lifetime Value

The two-year transaction history supports BG/NBD and Gamma-Gamma CLV models.
Split on a calibration/holdout boundary (e.g., week 52) and evaluate
predicted spend against actual holdout transactions. Demographic features can
serve as covariates in extended models.

### sec5.3 — Collaborative Filtering

Construct a user-item purchase matrix from `transaction_data`
(`household_key` x `PRODUCT_ID`). Use purchase counts or binary indicators as
implicit feedback signals. The dataset's density is low (many products per
household), making it a realistic test for sparse collaborative filtering.

### sec5.4 — Content-Based / Hybrid Recommendations

Combine product features from `product` (DEPARTMENT, BRAND, COMMODITY_DESC)
with transaction signals to build content-based or hybrid recommenders.
The product hierarchy provides natural feature groupings at multiple
granularity levels.

### sec5.7 — Uplift-Based Personalization

Use `campaign_table` as the treatment assignment indicator and
`coupon_redempt` (or post-campaign purchase lift) as the outcome. The 30
campaigns in `campaign_desc` with explicit start/end days allow before-after
and treatment-control comparisons for uplift modeling.

### sec6.2 — Price Elasticity

Derive unit prices from `SALES_VALUE / QUANTITY` and discount depth from
`abs(RETAIL_DISC) / (SALES_VALUE - RETAIL_DISC)`. Combine with `causal_data`
display/mailer indicators to control for promotional context when estimating
price elasticity curves.

### sec6.3 — Product Assortment Optimization

Leverage the three-level product hierarchy (`DEPARTMENT` > `COMMODITY_DESC` >
`SUB_COMMODITY_DESC`) to analyze category structure, identify substitution
patterns via cross-elasticities, and evaluate assortment breadth versus depth
trade-offs at the store level.

### sec6.4 — Promotion Optimization

Join `transaction_data` discount columns (`RETAIL_DISC`, `COUPON_DISC`,
`COUPON_MATCH_DISC`) with `causal_data` (`display`, `mailer`) to measure
incremental lift from each promotional vehicle. Model promotion ROI by
comparing baseline sales (non-promoted weeks) against promoted periods.
