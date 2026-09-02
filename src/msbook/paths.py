"""Resolve repo-root-anchored paths.

All notebooks should import from ``msbook.paths`` instead of using
``../`` relative paths. The package is installed via ``pip install -e .``
from the repo root, so these helpers work from any cwd or notebook
subdirectory (local JupyterLab, VSCode, Colab, nbconvert).

Repo-root detection walks up from this file until a ``pyproject.toml`` or
``_quarto.yml`` marker is found, so it is independent of the working
directory. It resolves to whichever checkout this file lives in, so with more
than one clone or git worktree, put that checkout's ``src/`` on ``PYTHONPATH``
rather than relying on wherever ``pip install -e .`` was last run.
"""
from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file until a repo-root marker is found.

    Markers are ``pyproject.toml`` (always at the repo root) or ``_quarto.yml``
    (present when running inside the Quarto book project). Works for both
    editable installs (``src/msbook/``) and direct path use. Raises
    RuntimeError if no ancestor contains a marker.
    """
    here = Path(__file__).resolve()
    markers = ("pyproject.toml", "_quarto.yml")
    for parent in [here, *here.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError(
        "Could not locate repo root (no pyproject.toml or _quarto.yml "
        f"ancestor of {here}). msbook resolves paths from its own location, "
        "so install it from a clone of the book repo with `pip install -e .` "
        "rather than as a standalone package."
    )


REPO_ROOT = _find_repo_root()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
IMAGES_DIR = REPO_ROOT / "images"
TABLES_DIR = REPO_ROOT / "tables"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def chapter_images(part: str, chapter: str) -> Path:
    """Directory for final figures embedded in qmd.

    Example
    -------
    >>> chapter_images("5", "sec5.1")
    .../images/part5/sec5.1
    """
    out = IMAGES_DIR / f"part{part}" / chapter
    out.mkdir(parents=True, exist_ok=True)
    return out


def chapter_tables(part: str, chapter: str) -> Path:
    """Directory for final summary tables embedded in qmd (small, human-readable)."""
    out = TABLES_DIR / f"part{part}" / chapter
    out.mkdir(parents=True, exist_ok=True)
    return out


def chapter_generated(part: str, chapter: str) -> Path:
    """Directory for generated datasets (derived-from-raw or synthetic).

    Contents are gitignored by default; individual files can be un-ignored
    in .gitignore when re-generation is expensive or the data is canonical
    (e.g. the bundled synthetic MMM dataset).
    """
    out = DATA_DIR / "generated" / f"part{part}" / chapter
    out.mkdir(parents=True, exist_ok=True)
    return out


def chapter_artifacts(part: str, chapter: str) -> Path:
    """Directory for model artifacts, traces, scratch (gitignored by default)."""
    out = ARTIFACTS_DIR / f"part{part}" / chapter
    out.mkdir(parents=True, exist_ok=True)
    return out
