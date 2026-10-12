# Copyright 2026 Chiang Qin Zhi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Barebones generalized dataset loader for Lag-Gemma + GluonTS.

This module is intentionally minimal and easy to customize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset, ListDataset, MetaData, TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from data.dataloaders.bfortfin_loader import load_bfortfin_dataframe
from data.dataloaders.cmapss_loader import load_cmapss_dataframe
from data.dataloaders.ett_loader import load_ett_dataframe

# Keep names simple and explicit; adjust as your local file names evolve.
ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
CUSTOM_DATASETS = ETT_DATASETS + ["bfortfin"]  # "cmapss"
LEFT_PAD_FINETUNE_GLUONTS_DATASETS = [
    "fred_md",
    "hospital",
    "tourism_monthly",
    "m4_hourly",
    # Excluded: ~24k/~23k tiny series that would dominate series-weighted
    # sampling on near-all-padding windows once left-padding is enabled.
    # "m4_quarterly",
    # "m4_yearly",
    "nn5_daily_without_missing",
    "nn5_weekly",
]
LEFT_PAD_FINETUNE_DATASETS = list(LEFT_PAD_FINETUNE_GLUONTS_DATASETS)
GLUONTS_DATASETS = [
    "australian_electricity_demand",
    "electricity_hourly",
    "london_smart_meters_without_missing",
    "solar_10_minutes",
    "wind_farms_without_missing",
    "pedestrian_counts",
    "uber_tlc_hourly",
    "traffic",
    "kdd_cup_2018_without_missing",
    "saugeenday",
    "sunspot_without_missing",
    "exchange_rate",
    "weather",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "traffic_nips",
]
ALL_DATASETS = CUSTOM_DATASETS + GLUONTS_DATASETS


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset loading rules.

    - file_name: CSV file name under `root_dir`.
    - freq: pandas/GluonTS frequency string.
    - prediction_length: default horizon for this dataset.
    - time_col: timestamp column in CSV.
    - target_col: target value column in CSV.
    - item_id_col: optional id column; if missing, file is treated as single series.
    """

    file_name: str
    freq: str
    prediction_length: int
    time_col: str = "date"
    target_col: str = "target"
    item_id_col: Optional[str] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec(file_name="ETTh1.csv", freq="H", prediction_length=24, target_col="OT"),
    "ETTh2": DatasetSpec(file_name="ETTh2.csv", freq="H", prediction_length=24, target_col="OT"),
    "ETTm1": DatasetSpec(file_name="ETTm1.csv", freq="15min", prediction_length=96, target_col="OT"),
    "ETTm2": DatasetSpec(file_name="ETTm2.csv", freq="15min", prediction_length=96, target_col="OT"),
    "bfortfin": DatasetSpec(
        file_name="",
        freq="D",
        prediction_length=96,
        time_col="date",
        target_col="target",
        item_id_col="item_id",
    ),
    "cmapss": DatasetSpec(
        file_name="",
        freq="H",
        prediction_length=1,
        time_col="date",
        target_col="target",
        item_id_col="item_id",
    ),
}


class SingleInstanceSampler(InstanceSampler):
    """Randomly pick a single valid window for each time series."""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


class CombinedDatasetIterator:
    """Sample entries from pooled datasets until every dataset is exhausted.

    Each draw randomly selects one active dataset according to ``weights``.
    If the selected dataset has no items left, it is removed from the sampling
    pool and iteration continues with the remaining datasets in the same pass.

    This is deliberate for multi-dataset pretraining: one mixed pass should
    consume all available datasets rather than terminate early when sampling
    happens to hit a shorter dataset that is already exhausted.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        seed: Optional[int],
        weights: Sequence[float],
    ) -> None:
        self._datasets = [iter(dataset) for dataset in datasets]
        self._weights = list(weights)
        self._rng = random.Random(seed)

    def __iter__(self) -> CombinedDatasetIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        while self._datasets:
            active = [(idx, weight) for idx, weight in enumerate(self._weights) if weight > 0]
            if not active:
                # No sampleable datasets remain (all non-positive weights).
                # End the iterator gracefully instead of raising from random.choices.
                raise StopIteration

            active_indices = [idx for idx, _ in active]
            active_weights = [weight for _, weight in active]
            picked_active = self._rng.choices(
                range(len(active_indices)),
                weights=active_weights,
                k=1,
            )[0]
            index = active_indices[picked_active]
            dataset = self._datasets[index]
            try:
                return next(dataset)
            except StopIteration:
                del self._datasets[index]
                del self._weights[index]

        raise StopIteration


class CombinedDataset:
    """Lightweight pooled dataset wrapper for multi-dataset pretraining."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        seed: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not datasets:
            raise ValueError("`datasets` must contain at least one dataset.")

        self._seed = seed
        self._datasets = list(datasets)
        if weights is None:
            n_datasets = len(self._datasets)
            self._weights = [1.0 / n_datasets] * n_datasets
        else:
            if len(weights) != len(self._datasets):
                raise ValueError("`weights` must match the number of datasets.")
            self._weights = list(weights)

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    def __iter__(self) -> CombinedDatasetIterator:
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)


@dataclass(frozen=True)
class PooledPretrainingDatasets:
    """Container for pooled multi-dataset pretraining inputs and metadata."""

    train_data: Dataset
    val_data: Dataset
    dataset_stats: list[dict[str, Any]]
    dropped_dataset_stats: list[dict[str, Any]]
    train_weights: list[float]
    val_weights: list[float]
    dataset_timesteps_requested: int | str
    dataset_timesteps_available: int
    dataset_timesteps_actual: int
    # Datasets that had one or more too-short series left-padded with NaN so
    # they meet the history_length requirement. Defaulted for backward
    # compatibility with callers that construct this dataclass directly.
    padded_dataset_stats: list[dict[str, Any]] = field(default_factory=list)


def _resolve_spec(name: str) -> DatasetSpec:
    if name not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset '{name}'. Add it to DATASET_SPECS. "
            f"Available: {sorted(DATASET_SPECS.keys())}"
        )
    return DATASET_SPECS[name]


def _read_csv(spec: DatasetSpec, root_dir: str | Path) -> pd.DataFrame:
    path = Path(root_dir) / spec.file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Update DATASET_SPECS[file_name] or pass a different root_dir."
        )

    df = pd.read_csv(path)
    required = [spec.time_col, spec.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}. "
            f"Expected at least {required}."
        )

    df = df.copy()
    df[spec.time_col] = pd.to_datetime(df[spec.time_col])
    df[spec.target_col] = pd.to_numeric(df[spec.target_col], errors="coerce")
    df = df.dropna(subset=[spec.time_col, spec.target_col]).sort_values(spec.time_col)
    return df


def _read_ett_csv(name: str, spec: DatasetSpec, root_dir: str | Path) -> pd.DataFrame:
    """Read ETT data from the dedicated loader."""
    return load_ett_dataframe(
        name=name,
        root_dir=root_dir,
        time_col=spec.time_col,
        target_col=spec.target_col,
    )


def _read_bfortfin_csv(spec: DatasetSpec, root_dir: str | Path) -> pd.DataFrame:
    """Read BForTFin data from the dedicated loader."""
    return load_bfortfin_dataframe(
        root_dir=root_dir,
        time_col=spec.time_col,
        target_col=spec.target_col,
        item_id_col=spec.item_id_col or "item_id",
    )


def _read_cmapss_csv(spec: DatasetSpec, root_dir: str | Path) -> pd.DataFrame:
    """Read CMAPSS data from the dedicated loader."""
    return load_cmapss_dataframe(
        root_dir=root_dir,
        time_col=spec.time_col,
        target_col=spec.target_col,
        item_id_col=spec.item_id_col or "item_id",
    )


def _iter_series(df: pd.DataFrame, spec: DatasetSpec) -> Iterable[Tuple[str, pd.DataFrame]]:
    if spec.item_id_col and spec.item_id_col in df.columns:
        for item_id, part in df.groupby(spec.item_id_col):
            yield str(item_id), part.sort_values(spec.time_col)
        return
    yield "0", df.sort_values(spec.time_col)


def _to_entries(
    df: pd.DataFrame,
    spec: DatasetSpec,
    data_id: Optional[int],
) -> List[dict]:
    entries: List[dict] = []
    for idx, (item_id, part) in enumerate(_iter_series(df, spec)):
        target = part[spec.target_col].to_numpy(dtype=np.float32)
        if target.size == 0:
            continue
        entry = {
            "target": target,
            "start": part[spec.time_col].iloc[0],
            "item_id": item_id,
        }
        if data_id is not None:
            entry["data_id"] = data_id
        else:
            entry["data_id"] = idx
        entries.append(entry)
    return entries


def load_raw_dataset(
    name: str,
    dataset_path: str | Path,
    data_id: Optional[int] = None,
    prediction_length: Optional[int] = None,
    freq: Optional[str] = None,
) -> TrainDatasets:
    """Load a CSV dataset and expose a GluonTS TrainDatasets object.

    This returns both train/test as full series. Split helpers below create
    train/validation/test windows for model training/evaluation.
    """

    if name in GLUONTS_DATASETS or name in LEFT_PAD_FINETUNE_GLUONTS_DATASETS:
        return get_dataset(name, path=Path(dataset_path))

    spec = _resolve_spec(name)
    if name in ETT_DATASETS:
        df = _read_ett_csv(name, spec, dataset_path)
    elif name == "bfortfin":
        df = _read_bfortfin_csv(spec, dataset_path)
    elif name == "cmapss":
        df = _read_cmapss_csv(spec, dataset_path)
    else:
        df = _read_csv(spec, dataset_path)
    entries = _to_entries(df, spec, data_id=data_id)

    pl = int(prediction_length or spec.prediction_length)
    f = str(freq or spec.freq)
    metadata = MetaData(freq=f, prediction_length=pl)

    full = ListDataset(entries, freq=f)
    return TrainDatasets(metadata=metadata, train=full, test=full)


def _resolve_sampling_weights(
    dataset_stats: Sequence[dict[str, Any]],
    weighting: str,
    field_name: str,
) -> list[float]:
    if weighting == "uniform":
        return [1.0] * len(dataset_stats)
    if weighting == "series":
        return [float(stat[field_name]) for stat in dataset_stats]
    if weighting == "timesteps":
        points_field = "total_train_points" if field_name == "num_train_series" else "total_val_points"
        return [float(stat[points_field]) for stat in dataset_stats]
    raise ValueError(
        f"Unknown weighting strategy '{weighting}'. Expected one of: uniform, series, timesteps."
    )


def _proportional_timestep_caps(
    capacities: Sequence[int],
    budget: int,
    minimum_points: int,
) -> list[int]:
    """Allocate a deterministic point budget proportionally across valid series."""
    if budget < 0:
        raise ValueError("dataset timestep budget must be >= 0")
    total = sum(capacities)
    if budget >= total:
        return list(capacities)

    eligible = [index for index, capacity in enumerate(capacities) if capacity >= minimum_points]
    if not eligible or budget < minimum_points:
        return [0] * len(capacities)

    caps = [0] * len(capacities)
    active = set(eligible)
    remaining = budget
    while active and remaining > 0:
        active_capacity = sum(capacities[index] - caps[index] for index in active)
        if active_capacity <= 0:
            break
        allocations: list[tuple[float, int, int]] = []
        for index in sorted(active):
            available = capacities[index] - caps[index]
            exact = remaining * available / active_capacity
            allocation = min(available, int(exact))
            allocations.append((exact - allocation, index, allocation))
        assigned = 0
        for _, index, allocation in allocations:
            caps[index] += allocation
            assigned += allocation
        leftover = remaining - assigned
        for _, index, _ in sorted(allocations, key=lambda item: (-item[0], item[1])):
            if leftover <= 0:
                break
            if caps[index] < capacities[index]:
                caps[index] += 1
                leftover -= 1
        remaining = budget - sum(caps)
        active = {index for index in active if caps[index] < capacities[index]}
        if assigned == 0 and leftover == remaining:
            break

    undersized = [
        index for index in eligible if 0 < caps[index] < minimum_points
    ]
    reclaimed = sum(caps[index] for index in undersized)
    for index in undersized:
        caps[index] = 0

    for index in sorted(eligible, key=lambda current: (-capacities[current], current)):
        if reclaimed <= 0:
            break
        if caps[index] == 0 and reclaimed >= minimum_points:
            allocation = min(capacities[index], reclaimed)
            caps[index] = allocation
            reclaimed -= allocation
        elif caps[index] >= minimum_points:
            allocation = min(capacities[index] - caps[index], reclaimed)
            caps[index] += allocation
            reclaimed -= allocation
    return caps


def _cap_training_datasets(
    train_datasets: Sequence[Dataset],
    *,
    dataset_timesteps: int | str,
    minimum_points: int,
) -> tuple[list[list[dict[str, Any]]], list[int], int, int]:
    entries_by_dataset = [[dict(entry) for entry in dataset] for dataset in train_datasets]
    capacities = [
        int(np.asarray(entry["target"]).size)
        for entries in entries_by_dataset
        for entry in entries
    ]
    available = sum(capacities)
    if dataset_timesteps == "all":
        budget = available
    else:
        budget = min(int(dataset_timesteps), available)

    caps = _proportional_timestep_caps(capacities, budget, minimum_points)
    capped_datasets: list[list[dict[str, Any]]] = []
    dataset_totals: list[int] = []
    offset = 0
    for entries in entries_by_dataset:
        capped_entries: list[dict[str, Any]] = []
        dataset_total = 0
        for entry in entries:
            cap = caps[offset]
            offset += 1
            if cap < minimum_points:
                continue
            capped_entry = dict(entry)
            capped_entry["target"] = np.asarray(entry["target"], dtype=np.float32)[:cap].copy()
            capped_entries.append(capped_entry)
            dataset_total += cap
        capped_datasets.append(capped_entries)
        dataset_totals.append(dataset_total)
    return capped_datasets, dataset_totals, available, sum(dataset_totals)


def build_pretraining_datasets(
    dataset_names: Sequence[str],
    dataset_path: str | Path,
    history_length: int,
    prediction_length: Optional[int] = None,
    num_val_windows: Optional[int] = 1,
    freq: Optional[str] = None,
    last_k_percentage: Optional[float] = None,
    weighting: str = "series",
    seed: Optional[int] = None,
    dataset_timesteps: int | str = "all",
    pad_short_series: bool = False,
) -> PooledPretrainingDatasets:
    """Build single- or multi-dataset train/validation inputs for pretraining.

    `pad_short_series` (default False): drop series shorter than the full history
    requirement (pretraining behavior). Set True to left-pad them with NaN
    instead (finetuning behavior).
    """

    if not dataset_names:
        raise ValueError("`dataset_names` must contain at least one dataset.")
    if dataset_timesteps != "all" and int(dataset_timesteps) < 1:
        raise ValueError("`dataset_timesteps` must be a positive integer or 'all'.")

    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    dataset_stats: list[dict[str, Any]] = []
    pad_stats_by_dataset: list[dict[str, Any]] = []

    for data_id, name in enumerate(dataset_names):
        (
            train_dataset,
            val_dataset,
            total_train_points,
            total_val_points,
            total_val_windows,
            max_train_end_date,
            total_points,
            pad_stats,
        ) = create_train_and_val_datasets_with_dates(
            name=name,
            dataset_path=dataset_path,
            data_id=data_id,
            history_length=history_length,
            prediction_length=prediction_length,
            num_val_windows=num_val_windows,
            freq=freq,
            last_k_percentage=last_k_percentage,
            pad_short_series=pad_short_series,
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        pad_stats_by_dataset.append(pad_stats)
        dataset_stats.append(
            {
                "dataset": name,
                "data_id": data_id,
                "num_train_series": len(train_dataset),
                # Despite its existing name, num_val_series means the number of
                # validation source series. Validation has one entry per rolling
                # window, so len(val_dataset) counts windows, not source series
                # (for example, 2 series with 3 windows each produces 6 validation
                # entries). Each accepted source series has one training entry, so
                # use len(train_dataset) for source-series weighting and reporting.
                "num_val_series": len(train_dataset),
                "total_train_points": total_train_points,
                "total_val_points": total_val_points,
                "total_val_windows": total_val_windows,
                "max_train_end_date": (
                    max_train_end_date.to_timestamp().isoformat()
                    if isinstance(max_train_end_date, pd.Period)
                    else max_train_end_date.isoformat()
                )
                if max_train_end_date is not None
                else None,
                "total_points": total_points,
            }
        )

    non_empty_indices = [
        idx
        for idx, stat in enumerate(dataset_stats)
        if int(stat["num_train_series"]) > 0 and int(stat["num_val_series"]) > 0
    ]
    dropped_dataset_stats = [
        {
            "dataset": dataset_stats[idx]["dataset"],
            "data_id": dataset_stats[idx]["data_id"],
            "num_train_series": dataset_stats[idx]["num_train_series"],
            "num_val_series": dataset_stats[idx]["num_val_series"],
            "reason": "no_valid_train_or_val_series",
        }
        for idx, stat in enumerate(dataset_stats)
        if idx not in non_empty_indices
    ]
    dropped_datasets = [stat["dataset"] for stat in dropped_dataset_stats]
    if dropped_datasets:
        train_datasets = [train_datasets[idx] for idx in non_empty_indices]
        val_datasets = [val_datasets[idx] for idx in non_empty_indices]
        dataset_stats = [dataset_stats[idx] for idx in non_empty_indices]
        pad_stats_by_dataset = [pad_stats_by_dataset[idx] for idx in non_empty_indices]

    # Report every retained dataset that had one or more series left-padded.
    padded_dataset_stats = [
        {
            "dataset": stat["dataset"],
            "data_id": stat["data_id"],
            "num_series_padded": pad_stats["num_series_padded"],
            "pad_length_min": pad_stats["pad_length_min"],
            "pad_length_max": pad_stats["pad_length_max"],
            "total_pad_points": pad_stats["total_pad_points"],
        }
        for stat, pad_stats in zip(dataset_stats, pad_stats_by_dataset)
        if pad_stats["num_series_padded"] > 0
    ]

    if not dataset_stats:
        raise ValueError(
            "No datasets produced non-empty train/validation series after applying "
            "history_length/prediction_length constraints. "
            f"Dropped datasets: {sorted(dropped_datasets)}"
        )

    available_timesteps = sum(int(stat["total_train_points"]) for stat in dataset_stats)
    actual_timesteps = available_timesteps
    for stat in dataset_stats:
        stat["total_train_points_available"] = stat["total_train_points"]
        stat["num_train_series_available"] = stat["num_train_series"]

    if dataset_timesteps != "all":
        if padded_dataset_stats:
            padded_names = [stat["dataset"] for stat in padded_dataset_stats]
            raise ValueError(
                "dataset_timesteps capping is not supported with NaN left-padded "
                f"datasets {padded_names}. Use dataset_timesteps='all' when "
                "pad_short_series=True."
            )
        minimum_train_points = int(history_length) + int(prediction_length or 1)
        capped_train_datasets, capped_totals, available_timesteps, actual_timesteps = (
            _cap_training_datasets(
                train_datasets,
                dataset_timesteps=dataset_timesteps,
                minimum_points=minimum_train_points,
            )
        )
        for stat, capped_entries, capped_total in zip(
            dataset_stats,
            capped_train_datasets,
            capped_totals,
        ):
            stat["total_train_points"] = capped_total
            stat["num_train_series"] = len(capped_entries)
        if not any(capped_train_datasets):
            raise ValueError(
                "Dataset timestep budget is too small to retain a valid training series. "
                f"Need at least {minimum_train_points} timesteps."
            )
        train_datasets = capped_train_datasets

    train_weights = _resolve_sampling_weights(
        dataset_stats=dataset_stats,
        weighting=weighting,
        field_name="num_train_series",
    )
    val_weights = _resolve_sampling_weights(
        dataset_stats=dataset_stats,
        weighting=weighting,
        field_name="num_val_series",
    )

    if len(train_datasets) == 1:
        train_data: Dataset = train_datasets[0]
        val_data: Dataset = val_datasets[0]
    else:
        train_data = CombinedDataset(train_datasets, seed=seed, weights=train_weights)
        val_data = CombinedDataset(val_datasets, seed=seed, weights=val_weights)

    return PooledPretrainingDatasets(
        train_data=train_data,
        val_data=val_data,
        dataset_stats=dataset_stats,
        dropped_dataset_stats=dropped_dataset_stats,
        padded_dataset_stats=padded_dataset_stats,
        train_weights=train_weights,
        val_weights=val_weights,
        dataset_timesteps_requested=dataset_timesteps,
        dataset_timesteps_available=available_timesteps,
        dataset_timesteps_actual=actual_timesteps,
    )


def create_train_and_val_datasets_with_dates(
    name: str,
    dataset_path: str | Path,
    data_id: int,
    history_length: int,
    prediction_length: Optional[int] = None,
    num_val_windows: Optional[int] = 1,
    val_start_date: Optional[pd.Timestamp] = None,
    train_start_date: Optional[pd.Timestamp] = None,
    freq: Optional[str] = None,
    last_k_percentage: Optional[float] = None,
    pad_short_series: bool = False,
):
    """Create train/val ListDataset in the same field format as previous utilities.

    Notes:
    - `val_start_date`/`train_start_date` are accepted for compatibility.
    - This barebones version primarily uses index-based splitting.
    - `pad_short_series` (default False): when False, series shorter than the
      full history requirement are dropped (pretraining behavior). When True,
      such series are left-padded with NaN instead (finetuning behavior).
    """

    del val_start_date, train_start_date  # kept only for API compatibility

    raw = load_raw_dataset(
        name=name,
        dataset_path=dataset_path,
        data_id=data_id,
        prediction_length=prediction_length,
        freq=freq,
    )

    pred_len = int(prediction_length or raw.metadata.prediction_length)
    f = str(freq or raw.metadata.freq)
    timestep_delta = pd.tseries.frequencies.to_offset(f)
    n_val = int(num_val_windows or 1)

    train_entries: List[dict] = []
    val_entries: List[dict] = []
    total_train_points = 0
    total_val_points = 0
    total_val_windows = 0
    max_train_end_date = None
    num_series_padded = 0
    total_pad_points = 0
    pad_length_min: Optional[int] = None
    pad_length_max: Optional[int] = None

    for i, series in enumerate(raw.train):
        target = np.asarray(series["target"], dtype=np.float32)
        real_size = int(target.size)
        validation_size = pred_len + n_val - 1
        full_requirement = history_length + pred_len + validation_size

        if not pad_short_series:
            # Drop-only behavior (pretraining default): series shorter than the
            # full history requirement are skipped, never padded.
            if real_size < full_requirement:
                continue
            series_start = series["start"]
            pad_len = 0
        else:
            # A series needs at least `prediction_length + validation_size` REAL
            # points to form even one (possibly left-padded) window. Anything
            # shorter cannot be rescued by padding and is dropped.
            if real_size < pred_len + validation_size:
                continue

            series_start = series["start"]
            pad_len = 0
            if real_size < full_requirement:
                # Left-pad too-short series with NaN so they satisfy the history
                # requirement. The estimator's AddObservedValuesIndicator marks
                # NaN as observed=0 and imputes 0.0, so this prefix becomes a
                # genuine left-pad: excluded from scaling and the loss. Padding by
                # `history_length` lets every real point serve as a forecast target
                # with a padded history (not just the last 1-2 points).
                pad_len = int(history_length)
                target = np.concatenate(
                    [np.full(pad_len, np.nan, dtype=np.float32), target]
                )
                series_start = series["start"] - pad_len * timestep_delta
                num_series_padded += 1
                total_pad_points += pad_len
                pad_length_min = pad_len if pad_length_min is None else min(pad_length_min, pad_len)
                pad_length_max = pad_len if pad_length_max is None else max(pad_length_max, pad_len)

        train_end = target.size - validation_size
        if last_k_percentage is not None:
            keep = max(1, int(train_end * float(last_k_percentage) / 100.0))
            train_start = max(0, train_end - keep)
        else:
            train_start = 0

        train_target = target[train_start:train_end]
        train_start_ts = series_start + train_start * timestep_delta
        train_entry = {
            "target": train_target,
            "start": train_start_ts,
            "item_id": series.get("item_id", str(i)),
            "data_id": data_id,
        }
        train_entries.append(train_entry)
        if pad_len > 0:
            # Do not let synthetic NaN left-padding inflate weighting stats.
            total_train_points += int(np.count_nonzero(~np.isnan(train_target)))
        else:
            total_train_points += int(train_target.size)

        if train_target.size > 0:
            end_ts = train_start_ts + (train_target.size - 1) * timestep_delta
            if max_train_end_date is None or end_ts > max_train_end_date:
                max_train_end_date = end_ts

        for window_idx in range(n_val):
            forecast_start = train_end + window_idx
            # Padded series meet history_length so val_start >= 0; clamp
            # defensively for any residual edge case.
            val_start = max(0, forecast_start - history_length)
            val_end = forecast_start + pred_len
            val_target = target[val_start:val_end]
            val_entry = {
                "target": val_target,
                "start": series_start + val_start * timestep_delta,
                "item_id": series.get("item_id", str(i)),
                "data_id": data_id,
            }
            val_entries.append(val_entry)
            if pad_len > 0:
                total_val_points += int(np.count_nonzero(~np.isnan(val_target)))
            else:
                total_val_points += int(val_target.size)
            total_val_windows += 1

    train_ds = ListDataset(train_entries, freq=f)
    val_ds = ListDataset(val_entries, freq=f)
    total_points = total_train_points + len(train_entries) * (pred_len + n_val - 1)
    pad_stats = {
        "num_series_padded": num_series_padded,
        "total_pad_points": total_pad_points,
        "pad_length_min": pad_length_min,
        "pad_length_max": pad_length_max,
    }

    return (
        train_ds,
        val_ds,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
        pad_stats,
    )


def create_test_dataset(
    name: str,
    dataset_path: str | Path,
    history_length: int,
    freq: Optional[str] = None,
    data_id: Optional[int] = None,
):
    """Create test ListDataset with last (history + prediction) span per series."""

    raw = load_raw_dataset(name=name, dataset_path=dataset_path, data_id=data_id, freq=freq)
    pred_len = int(raw.metadata.prediction_length)
    f = str(freq or raw.metadata.freq)
    timestep_delta = pd.tseries.frequencies.to_offset(f)

    data: List[dict] = []
    total_points = 0
    for i, series in enumerate(raw.test):
        target = np.asarray(series["target"], dtype=np.float32)
        span = history_length + pred_len
        if target.size > span:
            offset = target.size - span
            target = target[-span:]
            start = series["start"] + offset * timestep_delta
        else:
            start = series["start"]

        entry = {
            "target": target,
            "start": start,
            "item_id": series.get("item_id", str(i)),
            "data_id": series.get("data_id", i if data_id is None else data_id),
        }
        data.append(entry)
        total_points += int(target.size)

    return ListDataset(data, freq=f), pred_len, total_points
