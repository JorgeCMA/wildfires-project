"""Generic CSV helpers anchored to the project root.

Paths are given project-root-relative, e.g.
``"data/processed/enriched/firms_spain_full_enrichment.csv"``.
Absolute paths are used as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wildfire.config import PROJECT_ROOT


def resolve_from_root(relative_path: str | Path) -> Path:
    """Resolve ``relative_path`` against the project root.

    Absolute paths are returned unchanged. Relative paths that would
    escape the project root (``..``) are rejected.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    resolved = (PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT.resolve() and (
        PROJECT_ROOT.resolve() not in resolved.parents
    ):
        raise ValueError(f"Path escapes the project root: {relative_path}")
    return resolved


def save_csv(df: pd.DataFrame, relative_path: str | Path) -> Path:
    """Save ``df`` to ``relative_path`` (project-root-relative).

    Parent directories are created as needed.

    Returns
    -------
    Path
        The resolved path the CSV was written to.
    """
    path = resolve_from_root(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_csv(relative_path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load a CSV from ``relative_path`` (project-root-relative).

    Extra keyword arguments are forwarded to :func:`pandas.read_csv`.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = resolve_from_root(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **read_kwargs)
