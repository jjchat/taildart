"""ETT dataset loader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


VALID_ETT_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


def _resolve_ett_dataset_dir(root_dir: Optional[str | Path] = None) -> Path:
    """Resolve the directory containing ETT CSV files."""
    if root_dir is None:
        root = Path("datasets")
    else:
        root = Path(root_dir)

    candidates = [
        root / "raw" / "Energy" / "ETT" / "Dataset",
        root / "Energy" / "ETT" / "Dataset",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return root / "raw" / "Energy" / "ETT" / "Dataset"


def load_ett_dataframe(
    name: str,
    root_dir: Optional[str | Path] = None,
    time_col: str = "date",
    target_col: str = "OT",
) -> pd.DataFrame:
    """Load and validate one ETT CSV as a DataFrame."""
    if name not in VALID_ETT_DATASETS:
        raise ValueError(
            f"Unknown ETT dataset '{name}'. "
            f"Valid names: {sorted(VALID_ETT_DATASETS)}"
        )

    dataset_dir = _resolve_ett_dataset_dir(root_dir)
    csv_path = dataset_dir / f"{name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"ETT CSV not found: {csv_path}. "
            "Pass root_dir pointing to datasets root or ETT Dataset directory."
        )

    df = pd.read_csv(csv_path)
    required = [time_col, target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} missing columns: {missing}. "
            f"Expected at least {required}."
        )

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col)
    return df
