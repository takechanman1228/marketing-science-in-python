# Marketing Science in Python: Code Notebooks

Executable notebooks and helper code for the book
**Marketing Science in Python: A Practitioner's Guide** by Hajime Takeda.

📖 **Read the book:** https://marketing-science-in-python.com/

## What this repository is

This repository holds the runnable code for the book: the chapter notebooks, local
helper modules, and the small datasets needed to reproduce the examples. Use it to run
the notebooks and follow along with the book.

## What's included

- `notebooks/` — chapter notebooks (`.ipynb`), grouped by part
- `src/msbook/` — local helper modules used by the notebooks for paths and dataset loading
- `data/` — small committed and synthetic datasets, dataset profiles, and a guide to obtaining
  the larger public datasets (`data/raw/README.md`)
- `environment.yml` and per-part `requirements.txt` — environment setup
- `pyproject.toml` — installs the `msbook` local helper modules (`pip install -e .`)

The book manuscript and its build system live elsewhere; this repository is code only.

## Getting started

```bash
# 1. Clone
git clone https://github.com/takechanman1228/marketing-science-in-python.git
cd marketing-science-in-python

# 2. Create the environment and install the local helper modules
conda env create -f environment.yml
conda activate marketing-science
pip install -e .

# 3. Open a notebook
jupyter lab notebooks/part3-causal-inference/sec3.2_ab_test_figures.ipynb
```

Prefer the browser? Every notebook has an **Open in Colab** badge at the top. If you use pip
instead of conda, install the per-part `requirements.txt` for the chapter you are running, then
`pip install -e .`.

## Notebooks by part

- **Part 3 — Causal Inference:** A/B test analysis, quasi-experiments, meta-learners (CATE),
  uplift modeling
- **Part 4 — Customer Analytics:** segmentation (traditional and embedding-based), customer
  lifetime value with PyMC-Marketing (plus a LightGBM companion pipeline)
- **Part 5 — Commercial Analytics:** price elasticity, assortment optimization, demand forecasting
- **Part 6 — Media Optimization:** marketing mix modeling, end to end

## Data

Small synthetic and derived datasets are committed under `data/generated/`. Larger public datasets
(Dunnhumby, Iowa liquor sales, Online Retail II, MineThatData) are not redistributed here;
`data/raw/README.md` explains how to obtain each one, and the `msbook.data` loaders read or fetch
them from `data/raw/`.

## License

- **Code** (notebooks, Python files, the `msbook` helper modules): MIT — see [`LICENSE-CODE`](LICENSE-CODE).
- **Book content and text:** © 2025–2026 Hajime Takeda, all rights reserved — see [`LICENSE`](LICENSE).

## Citation

If you reference this work, please cite it using [`CITATION.cff`](CITATION.cff).
