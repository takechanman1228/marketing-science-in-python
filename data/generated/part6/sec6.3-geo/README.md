# `data/generated/part6/sec6.3-geo/` — Geo experiment panel

Synthetic geo-week data used by **Chapter 6.3 (Incrementality Experiments)**.
The panel is small enough to run the full design, power, estimation, synthetic
control, and placebo workflow in seconds.

## Files

| File | Description |
|---|---|
| `geo_panel.csv` | 40 anonymous geographies × 104 Mondays with observed Google Search spend and revenue |
| `geo_ground_truth.csv` | The 64 planted treated geo-weeks, their incremental spend and revenue, and the experiment-level true ROAS |

## Frozen design

- 8 matched pairs selected from 40 candidate geographies
- 1 geography randomized to treatment within each pair
- 8-week test from 2024-02-05 through 2024-03-25
- 30% spend cut in treated geographies
- the return on that cut (revenue given up per dollar removed) planted at
  1.50, the same value the MMM generator pins for Google Search

The observed panel carries no treatment flag and no lift column (the 30%
spend cut itself is visible, as in any real experiment). The companion
notebook reconstructs the assignment from pre-period data alone and opens
the ground-truth file only after estimation.

## Provenance

Generated from scratch by:

```
notebooks/part6-media-optimization/geo_experiment_data_generation.py
```

The generator is fully seeded and is the source of truth. The CSVs are
committed so readers can run the experiment without a preparation step.

## License

Synthetic / author-generated. No third-party data or confidential project
material is included.
