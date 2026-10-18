"""CMAPSS dataset loader helpers for TailDart pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


VALID_FD_SPLITS = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_BASE_TIMESTAMP = pd.Timestamp("2000-01-01 00:00:00")


def _is_cmapss_dataset_dir(path: Path) -> bool:
    """Return True when a directory contains all CMAPSS train split files."""
    if not path.is_dir():
        return False

    for split in VALID_FD_SPLITS:
        if not (path / f"train_{split}.txt").is_file():
            return False
    return True


def _resolve_cmapss_dataset_dir(root_dir: Optional[str | Path] = None) -> Path:
    """Resolve the directory containing CMAPSS split files."""
    if root_dir is None:
        root = Path("datasets")
    else:
        root = Path(root_dir)

    candidates = [
        root / "raw" / "Industrial" / "CMAPSS" / "CMAPSS",
        root / "Industrial" / "CMAPSS" / "CMAPSS",
        root / "raw" / "Industrial" / "CMAPSS",
        root / "Industrial" / "CMAPSS",
        root,
    ]
    for candidate in candidates:
        if _is_cmapss_dataset_dir(candidate):
            return candidate
        nested = candidate / "CMAPSS"
        if _is_cmapss_dataset_dir(nested):
            return nested

    return root / "raw" / "Industrial" / "CMAPSS" / "CMAPSS"


def _read_cmapss_train_split(path: Path) -> pd.DataFrame:
    """Read one CMAPSS train split file and assign stable column names."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 26:
        raise ValueError(f"{path} has {df.shape[1]} columns, expected at least 26.")

    columns = [
        "unit",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        *[f"sensor_{idx:02d}" for idx in range(1, 22)],
    ]
    df = df.iloc[:, :26].copy()
    df.columns = columns
    return df


def load_cmapss_dataframe(
    root_dir: Optional[str | Path] = None,
    time_col: str = "date",
    target_col: str = "target",
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """Load all CMAPSS train splits as a single sensors-only long DataFrame.

    Output schema:
    - ``time_col``: synthetic hourly timestamp derived from cycle count.
    - ``target_col``: univariate sensor value.
    - ``item_id_col``: stable series id ``cmapss__FDxxx__unit_xxx__sensor_xx``.
    """
    dataset_dir = _resolve_cmapss_dataset_dir(root_dir)
    parts: list[pd.DataFrame] = []

    sensor_columns = [f"sensor_{idx:02d}" for idx in range(1, 22)]

    for split in VALID_FD_SPLITS:
        split_path = dataset_dir / f"train_{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"CMAPSS train split not found: {split_path}. "
                "Pass root_dir pointing to datasets root or CMAPSS data directory."
            )

        split_df = _read_cmapss_train_split(split_path)
        long_df = split_df.melt(
            id_vars=["unit", "cycle"],
            value_vars=sensor_columns,
            var_name="sensor_name",
            value_name=target_col,
        )
        long_df[target_col] = pd.to_numeric(long_df[target_col], errors="coerce")
        long_df = long_df.dropna(subset=[target_col])

        long_df["unit"] = pd.to_numeric(long_df["unit"], errors="coerce").astype("Int64")
        long_df["cycle"] = pd.to_numeric(long_df["cycle"], errors="coerce").astype("Int64")
        long_df = long_df.dropna(subset=["unit", "cycle"])

        long_df[time_col] = CMAPSS_BASE_TIMESTAMP + pd.to_timedelta(
            long_df["cycle"].astype(int) - 1,
            unit="h",
        )
        long_df[item_id_col] = (
            "cmapss__"
            + split
            + "__unit_"
            + long_df["unit"].astype(int).map(lambda val: f"{val:03d}")
            + "__"
            + long_df["sensor_name"]
        )
        parts.append(long_df[[time_col, target_col, item_id_col]])

    if not parts:
        raise ValueError(f"No CMAPSS train rows loaded from {dataset_dir}.")

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values([item_id_col, time_col], kind="stable").reset_index(drop=True)
    return result
