"""BForTFin dataset loader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _is_bfortfin_dataset_dir(path: Path) -> bool:
    """Heuristic check for a valid BForTFin dataset directory."""
    if not path.is_dir():
        return False

    csv_paths = sorted(path.rglob("*.csv"))
    if not csv_paths:
        return False

    # Validate schema on a few files.
    checked = 0
    for csv_path in csv_paths:
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        if {"Date", "Price"}.issubset(set(cols)):
            checked += 1
        if checked >= 3:
            return True
    return checked > 0


def _resolve_bfortfin_dataset_dir(root_dir: Optional[str | Path] = None) -> Path:
    """Resolve the directory containing BForTFin CSV files."""
    if root_dir is None:
        root = Path("datasets")
    else:
        root = Path(root_dir)

    candidates = [
        root / "raw" / "Finance" / "BForTFin" / "Dataset",
        root / "Finance" / "BForTFin" / "Dataset",
        root,
    ]
    for candidate in candidates:
        if _is_bfortfin_dataset_dir(candidate):
            return candidate

    return root / "raw" / "Finance" / "BForTFin" / "Dataset"


def _parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse mixed date formats seen in BForTFin files."""
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        mask = parsed.isna()
        parsed.loc[mask] = pd.to_datetime(values.loc[mask], errors="coerce", dayfirst=True)
    return parsed


def load_bfortfin_dataframe(
    root_dir: Optional[str | Path] = None,
    time_col: str = "date",
    target_col: str = "target",
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """Load and validate all BForTFin CSV files as one multi-series DataFrame."""
    dataset_dir = _resolve_bfortfin_dataset_dir(root_dir)
    csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        required = ["Date", "Price"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}. Expected at least {required}.")

        part = pd.DataFrame()
        part[time_col] = _parse_mixed_dates(df["Date"])
        part[target_col] = pd.to_numeric(
            df["Price"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

        rel = path.relative_to(dataset_dir).with_suffix("")
        item_id = "__".join(rel.parts).replace(" ", "_").replace("&", "and")
        part[item_id_col] = item_id

        part = part.dropna(subset=[time_col, target_col]).sort_values(time_col)
        if not part.empty:
            frames.append(part)

    if not frames:
        raise ValueError(f"All BForTFin series under {dataset_dir} are empty after cleaning.")

    return pd.concat(frames, ignore_index=True)
