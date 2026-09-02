# `data/` — raw & generated data

Repo-canonical location for all datasets used by the book's notebooks.

| Subdirectory | Tier | Bundled in git? | Typical contents |
|---|---|---|---|
| `raw/` | Raw external datasets (fetched on demand or placed manually) | ❌ gitignored | Dunnhumby, Online Retail II, MineThatData |
| `generated/` | Generated — derived-from-raw and synthetic | ❌ by default; canonical/expensive outputs committed | `segment_assignments.csv`, weekly parquet, MMM synthetic |
| `profiles/` | Dataset documentation (source, license, schema) | ✅ | one `.md` per dataset |

`generated/` mirrors the chapter layout: `generated/part<N>/<chapter>/...`.

## Path access from notebooks

Notebooks resolve data paths via `msbook`, never via fragile relative paths:

```python
from msbook.paths import chapter_generated          # -> data/generated/partN/<chapter>/
from msbook.data import (
    load_online_retail_ii, load_dunnhumby, load_minethatdata, load_mmm_synthetic,
)

df = load_online_retail_ii()                          # reads data/raw/...
```

Raw datasets are gitignored (large / license-restricted) — see
[`raw/README.md`](raw/README.md) for how to obtain each. Generated data is
gitignored by default; canonical or expensive-to-regenerate outputs are
re-included explicitly in `.gitignore`.
