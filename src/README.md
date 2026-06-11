# `msbook` — local notebook helpers

`msbook` contains small local helper modules used by the notebooks in this repository.

Readers do not need to learn it as a separate library. It is installed with
`pip install -e .` so the notebooks can reliably find data files and shared dataset loaders.
It is not published to PyPI.

It provides:

- `msbook.paths` — stable paths to repository folders such as `data/`, `images/`, and `artifacts/`
- `msbook.data` — dataset loaders used by selected notebooks

Install the local helpers once after cloning:

```bash
conda env create -f environment.yml
conda activate marketing-science
pip install -e .
```
