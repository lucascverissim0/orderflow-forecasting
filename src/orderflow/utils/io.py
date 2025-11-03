"""
io.py
------
Tiny I/O helpers to standardize how we read/write Parquet/CSV across the project.
- Resolves paths relative to the configured storage root.
- Ensures parent folders exist before writes.
"""

from __future__ import annotations
import os
from typing import Optional, Union, Dict, Any
import pandas as pd

from orderflow.utils.config import get_config


def _abs_path(rel_path: str) -> str:
    """
    Join a repo-relative path to the configured storage root if the path is under
    data folders; otherwise return as-is. This lets you pass paths like:
      "data/processed/features.parquet" or just "processed/features.parquet"
    """
    cfg = get_config()
    root = cfg.storage.get("root", ".")
    # Normalize and detect if path already absolute
    if os.path.isabs(rel_path):
        return rel_path

    # If caller passes "processed/xxx.parquet", map to "<root>/processed/xxx.parquet"
    if rel_path.split(os.sep)[0] in {"data", "raw", "interim", "processed"}:
        # If they already included "data/", keep it, else prefix with root
        if rel_path.startswith("data" + os.sep):
            abs_p = os.path.join(rel_path)  # keep "data/..."
        else:
            abs_p = os.path.join(root, rel_path)
    else:
        abs_p = os.path.join(root, rel_path)

    return os.path.normpath(abs_p)


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------- Parquet ---------------- #

def read_parquet(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a Parquet file with pyarrow/fastparquet (pandas decides).
    """
    ap = _abs_path(path)
    return pd.read_parquet(ap, **kwargs)


def write_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Write a DataFrame to Parquet, ensuring parent directories exist.
    """
    ap = _abs_path(path)
    _ensure_parent(ap)
    df.to_parquet(ap, index=True, **kwargs)


# ---------------- CSV (for quick imports/exports) ---------------- #

def read_csv(path: str, **kwargs) -> pd.DataFrame:
    ap = _abs_path(path)
    return pd.read_csv(ap, **kwargs)


def write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    ap = _abs_path(path)
    _ensure_parent(ap)
    df.to_csv(ap, index=True, **kwargs)


# ---------------- Convenience: safe load-or-create ---------------- #

def load_or_empty_parquet(path: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Try reading a Parquet file; if it doesn't exist, return an empty DataFrame
    with optional columns.
    """
    ap = _abs_path(path)
    if not os.path.exists(ap):
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(ap)


# Example usage (manual test)
if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})
    write_parquet(df, "processed/example.parquet")
    print(read_parquet("processed/example.parquet"))
