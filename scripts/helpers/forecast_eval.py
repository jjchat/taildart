"""Reusable helpers for evaluations."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation.metrics import calculate_seasonal_error
from gluonts.evaluation.metrics import mase as gluonts_mase

TAIL_DART_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TAIL_DART_ROOT.parent
for p in (TAIL_DART_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.training import build_estimator_kwargs

DEFAULT_METRIC_COLUMNS = ["mae", "mse", "rmse", "mape", "smape", "mase", "tilde_q"]
METRIC_EPS = np.finfo(float).eps

@dataclass
class ForecastResult:
    """Per-series holdout forecast and aligned ground truth."""

    item_id: str
    forecast_start: pd.Timestamp
    history: np.ndarray
    actual: np.ndarray
    prediction: np.ndarray


def ensure_paths() -> tuple[Path, Path]:
    """Return (tail_dart_root, project_root) and inject into sys.path."""
    return TAIL_DART_ROOT, PROJECT_ROOT


def _dataset_frequency(name: str) -> str:
    """Return the canonical GluonTS frequency string for a supported dataset."""
    if name in {"ETTm1", "ETTm2"}:
        return "15min"
    if name in {"ETTh1", "ETTh2"}:
        return "H"
    if name == "bfortfin":
        return "D"
    raise ValueError(f"Unsupported dataset name: {name}")


def load_dataset_frame(name: str, dataset_root: Path) -> pd.DataFrame:
    """Load one supported dataset as a normalized frame with date/target/item_id."""
    if name == "bfortfin":
        from data.dataloaders.bfortfin_loader import load_bfortfin_dataframe

        return load_bfortfin_dataframe(dataset_root)

    from data.dataloaders.ett_loader import load_ett_dataframe

    df = load_ett_dataframe(name=name, root_dir=dataset_root, time_col="date", target_col="OT")
    return df.assign(item_id=name)[["date", "target", "item_id"]]


def returns_frame(
    df: pd.DataFrame,
    *,
    time_col: str = "date",
    target_col: str = "target",
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """Convert a multi-series level DataFrame into per-series returns."""
    frames: list[pd.DataFrame] = []

    for item_id, part in df.groupby(item_id_col):
        out = part.sort_values(time_col).copy()
        out[target_col] = out[target_col].pct_change(fill_method=None)
        out = out.dropna(subset=[time_col, target_col])
        out = out[np.isfinite(out[target_col])].reset_index(drop=True)
        if out.empty:
            continue
        out[item_id_col] = item_id
        frames.append(out[[time_col, target_col, item_id_col]])

    if not frames:
        raise ValueError("No valid return series remain after pct_change transformation.")

    return pd.concat(frames, ignore_index=True)


def prepare_temporal_series_cache(
    df: pd.DataFrame,
    *,
    time_col: str = "date",
    target_col: str = "target",
    item_id_col: str = "item_id",
) -> list[dict]:
    """Precompute sorted per-series targets/timestamps for repeated split calls."""

    cache: list[dict] = []
    for item_id, part in df.groupby(item_id_col):
        part = part.sort_values(time_col).dropna(subset=[target_col]).reset_index(drop=True)
        if part.empty:
            continue
        cache.append(
            {
                "item_id": str(item_id),
                "target": part[target_col].to_numpy(dtype=np.float32),
                "timestamps": pd.to_datetime(part[time_col]).reset_index(drop=True),
            }
        )
    return cache


def temporal_split_for_holdout(
    df: pd.DataFrame,
    prediction_length: int,
    train_ratio: float = 0.8,
    min_context: int = 64,
    *,
    series_cache: list[dict] | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Split each series into train/holdout windows for one-shot forecast eval."""
    train_entries: list[dict] = []
    holdout_entries: list[dict] = []
    rows: list[dict] = []

    prepared = (
        series_cache if series_cache is not None else prepare_temporal_series_cache(df)
    )
    for series in prepared:
        item_id = str(series["item_id"])
        target = np.asarray(series["target"], dtype=np.float32)
        timestamps = pd.to_datetime(series["timestamps"]).reset_index(drop=True)

        min_train_len = max(min_context, prediction_length + 1)
        split_idx = max(min_train_len, int(len(target) * train_ratio))
        if split_idx + prediction_length > len(target):
            continue

        train_target = target[:split_idx]
        holdout_target = target[: split_idx + prediction_length]

        start = pd.Timestamp(timestamps.iloc[0])
        forecast_start = pd.Timestamp(timestamps.iloc[split_idx])

        train_entries.append(
            {
                "target": train_target,
                "start": start,
                "item_id": str(item_id),
            }
        )
        holdout_entries.append(
            {
                "target": holdout_target,
                "start": start,
                "item_id": str(item_id),
            }
        )

        rows.append(
            {
                "item_id": str(item_id),
                "n_total": int(len(target)),
                "n_train": int(split_idx),
                "n_test": int(prediction_length),
                "forecast_start": forecast_start,
            }
        )

    if not train_entries:
        raise ValueError("No valid series available after temporal split.")

    return train_entries, holdout_entries, pd.DataFrame(rows)


def temporal_train_val_holdout_split(
    df: pd.DataFrame,
    prediction_length: int,
    train_ratio: float = 0.8,
    min_context: int = 64,
    *,
    series_cache: list[dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame]:
    """Split each series into train, validation, and final holdout windows.

    The returned datasets are intended for fine-tuning with a trainer validation
    window that is distinct from the final reported holdout test window.
    """
    train_entries: list[dict] = []
    val_entries: list[dict] = []
    holdout_entries: list[dict] = []
    rows: list[dict] = []

    min_train_len = max(min_context, prediction_length + 1)
    min_required = min_train_len + (2 * prediction_length)

    prepared = (
        series_cache if series_cache is not None else prepare_temporal_series_cache(df)
    )
    for series in prepared:
        item_id = str(series["item_id"])
        target = np.asarray(series["target"], dtype=np.float32)
        timestamps = pd.to_datetime(series["timestamps"]).reset_index(drop=True)
        if len(target) < min_required:
            continue

        pre_holdout_idx = max(min_train_len + prediction_length, int(len(target) * train_ratio))
        holdout_end_idx = pre_holdout_idx + prediction_length
        if holdout_end_idx > len(target):
            continue

        val_start_idx = pre_holdout_idx - prediction_length
        train_end_idx = val_start_idx
        if train_end_idx < min_train_len:
            continue

        start = pd.Timestamp(timestamps.iloc[0])
        val_start = pd.Timestamp(timestamps.iloc[val_start_idx])
        forecast_start = pd.Timestamp(timestamps.iloc[pre_holdout_idx])

        train_entries.append(
            {
                "target": target[:train_end_idx],
                "start": start,
                "item_id": str(item_id),
            }
        )
        val_entries.append(
            {
                "target": target[:pre_holdout_idx],
                "start": start,
                "item_id": str(item_id),
            }
        )
        holdout_entries.append(
            {
                "target": target[:holdout_end_idx],
                "start": start,
                "item_id": str(item_id),
            }
        )

        rows.append(
            {
                "item_id": str(item_id),
                "n_total": int(len(target)),
                "n_train": int(train_end_idx),
                "n_val": int(prediction_length),
                "n_test": int(prediction_length),
                "val_start": val_start,
                "forecast_start": forecast_start,
            }
        )

    if not train_entries:
        raise ValueError("No valid series available after temporal train/val/holdout split.")

    return train_entries, val_entries, holdout_entries, pd.DataFrame(rows)


def build_chunk_holdout_entries(
    df: pd.DataFrame,
    *,
    block_size: int | None = None,
    context_length: int | None = None,
    prediction_length: int,
    freq_label: str,
    freq_alias: str,
    time_col: str = "date",
    target_col: str = "target",
    item_id_col: str = "item_id",
) -> tuple[list[dict], pd.DataFrame]:
    """Split each series into non-overlapping fixed blocks with a trailing holdout.

    Pass ``context_length`` when the named window is the number of observations
    visible to the model. The resulting block contains ``context_length`` plus
    ``prediction_length`` values. ``block_size`` retains the legacy behavior in
    which the supplied size includes the trailing holdout.
    """
    if (block_size is None) == (context_length is None):
        raise ValueError("Provide exactly one of block_size or context_length")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")

    if context_length is not None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        window_size = int(context_length)
        rows_per_block = window_size + int(prediction_length)
    else:
        rows_per_block = int(block_size)
        if rows_per_block <= prediction_length:
            raise ValueError("block_size must be greater than prediction_length")
        window_size = rows_per_block

    holdout_entries = []
    split_rows = []

    for item_id, part in df.groupby(item_id_col):
        ordered = part.sort_values(time_col).reset_index(drop=True)
        ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
        ordered = ordered.dropna(subset=[time_col, target_col]).reset_index(drop=True)
        if ordered.empty:
            continue
        if ordered[target_col].nunique() <= 1:
            continue

        n_blocks = len(ordered) // rows_per_block
        if n_blocks <= 0:
            continue

        trimmed = ordered.iloc[: n_blocks * rows_per_block].copy().reset_index(drop=True)
        if len(trimmed) == 0 or trimmed[target_col].isnull().all() or trimmed[target_col].nunique() <= 1:
            continue

        for block_idx in range(n_blocks):
            lo = block_idx * rows_per_block
            hi = lo + rows_per_block
            block = trimmed.iloc[lo:hi].copy().reset_index(drop=True)
            block_target = block[target_col].to_numpy(dtype=np.float32)

            finite_mask = np.isfinite(block_target)
            if not finite_mask.all():
                continue
            if block_target.size < (prediction_length + 1):
                continue
            if np.unique(block_target).size <= 1:
                continue

            block_item_id = f"{item_id}__B{block_idx+1:04d}_{freq_alias}{window_size}"
            start = pd.Timestamp(block[time_col].iloc[0])
            forecast_start = pd.Timestamp(block[time_col].iloc[-prediction_length])

            holdout_entries.append(
                {
                    "target": block_target,
                    "start": start,
                    "item_id": block_item_id,
                }
            )

            split_rows.append(
                {
                    "source_item_id": str(item_id),
                    "item_id": block_item_id,
                    "frequency": freq_label,
                    "window": window_size,
                    "window_label": f"{window_size}{freq_alias}",
                    "block_index": int(block_idx + 1),
                    "n_total": int(block_target.size),
                    "n_train": int(block_target.size - prediction_length),
                    "n_test": int(prediction_length),
                    "forecast_start": forecast_start,
                }
            )

    if not holdout_entries:
        raise ValueError(f"No valid chunk entries for {freq_label} window={window_size}.")

    return holdout_entries, pd.DataFrame(split_rows)


def temporal_fixed_train_rolling_split(
    df: pd.DataFrame,
    prediction_length: int,
    train_ratio: float = 0.8,
    min_context: int = 32,
    *,
    step_size: int = 1,
    series_cache: list[dict] | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Build fixed-train rolling test datasets for one-step evaluation."""

    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0.")
    if step_size <= 0:
        raise ValueError("step_size must be > 0.")

    train_entries: list[dict] = []
    rolling_entries: list[dict] = []
    rows: list[dict] = []

    min_train_len = max(min_context, prediction_length + 1)
    prepared = (
        series_cache if series_cache is not None else prepare_temporal_series_cache(df)
    )
    for series in prepared:
        item_id = str(series["item_id"])
        target = np.asarray(series["target"], dtype=np.float32)
        timestamps = pd.to_datetime(series["timestamps"]).reset_index(drop=True)
        n_total = int(len(target))
        if n_total < (min_train_len + prediction_length):
            continue

        train_end_idx = max(min_train_len, int(n_total * train_ratio))
        if train_end_idx + prediction_length > n_total:
            continue

        start = pd.Timestamp(timestamps.iloc[0])
        train_entries.append(
            {
                "target": target[:train_end_idx],
                "start": start,
                "item_id": item_id,
            }
        )

        first_forecast_idx = train_end_idx
        last_forecast_idx = n_total - prediction_length
        step_count = 0
        for forecast_idx in range(first_forecast_idx, last_forecast_idx + 1, step_size):
            history = target[:forecast_idx]
            context = history[-min_context:]
            if context.size != min_context:
                continue
            holdout_end = forecast_idx + prediction_length
            if holdout_end > n_total:
                continue
            rolling_entries.append(
                {
                    "target": target[:holdout_end],
                    "start": start,
                    "item_id": item_id,
                }
            )
            step_count += 1

        if step_count == 0:
            continue

        rows.append(
            {
                "item_id": item_id,
                "n_total": int(n_total),
                "n_train": int(train_end_idx),
                "n_val": 0,
                "n_test": int(step_count * prediction_length),
                "val_start": pd.NaT,
                "forecast_start": pd.Timestamp(timestamps.iloc[first_forecast_idx]),
                "initial_cutoff_timestamp": pd.Timestamp(timestamps.iloc[train_end_idx - 1]),
                "n_steps": int(step_count),
                "prediction_length": int(prediction_length),
                "step_size": int(step_size),
                "walk_style": "fixed_train_rolling",
            }
        )

    if not train_entries:
        raise ValueError("No valid series available after fixed-train rolling split.")

    return train_entries, rolling_entries, pd.DataFrame(rows)


def few_shot_split_for_holdout(
    df: pd.DataFrame,
    prediction_length: int,
    shots: int,
    min_context: int = 64,
    *,
    series_cache: list[dict] | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Build few-shot train windows and one holdout window per series.

    `shots` is the number of contiguous `prediction_length` chunks used for
    supervised adaptation before the final holdout chunk.
    """
    if shots <= 0:
        raise ValueError("`shots` must be >= 1.")

    train_entries: list[dict] = []
    holdout_entries: list[dict] = []
    rows: list[dict] = []

    min_required = max(min_context, prediction_length) + (shots + 1) * prediction_length

    prepared = (
        series_cache if series_cache is not None else prepare_temporal_series_cache(df)
    )
    for series in prepared:
        item_id = str(series["item_id"])
        target = np.asarray(series["target"], dtype=np.float32)
        timestamps = pd.to_datetime(series["timestamps"]).reset_index(drop=True)
        if len(target) < min_required:
            continue

        holdout_start = len(target) - prediction_length
        few_shot_end = holdout_start
        few_shot_start = max(min_context, few_shot_end - shots * prediction_length)

        train_target = target[:few_shot_end]
        holdout_target = target[: holdout_start + prediction_length]

        start = pd.Timestamp(timestamps.iloc[0])
        forecast_start = pd.Timestamp(timestamps.iloc[holdout_start])

        train_entries.append(
            {
                "target": train_target,
                "start": start,
                "item_id": str(item_id),
            }
        )
        holdout_entries.append(
            {
                "target": holdout_target,
                "start": start,
                "item_id": str(item_id),
            }
        )

        rows.append(
            {
                "item_id": str(item_id),
                "n_total": int(len(target)),
                "n_few_shot_points": int(few_shot_end - few_shot_start),
                "shots": int(shots),
                "n_test": int(prediction_length),
                "forecast_start": forecast_start,
            }
        )

    if not train_entries:
        raise ValueError("No valid series available for few-shot split.")

    return train_entries, holdout_entries, pd.DataFrame(rows)


def make_train_and_val_from_train_entries(
    train_entries: Iterable[dict],
    prediction_length: int,
    freq: str,
) -> tuple[ListDataset, ListDataset]:
    """Create train/validation datasets by holding out the tail of each training series."""
    fit_entries: list[dict] = []
    val_entries: list[dict] = []

    for entry in train_entries:
        target = np.asarray(entry["target"], dtype=np.float32)
        if len(target) <= prediction_length + 1:
            continue

        fit_entries.append(
            {
                "target": target[:-prediction_length],
                "start": entry["start"],
                "item_id": entry["item_id"],
            }
        )
        val_entries.append(
            {
                "target": target,
                "start": entry["start"],
                "item_id": entry["item_id"],
            }
        )

    if not fit_entries:
        raise ValueError("No valid series remained for train/validation datasets.")

    return ListDataset(fit_entries, freq=freq), ListDataset(val_entries, freq=freq)


def listdataset(entries: Iterable[dict], freq: str) -> ListDataset:
    """Materialize a GluonTS ListDataset from iterable entry dictionaries."""
    return ListDataset(list(entries), freq=freq)


def load_tail_dart_from_ckpt(
    ckpt_path: str | Path,
    prediction_length: int,
    context_length: int,
    device: str = "cpu",
    batch_size: int = 8,
    num_parallel_samples: int = 100,
    use_single_pass_sampling: bool = False,
    model_config: dict | None = None,
):
    """Load a predictor from a pretrained checkpoint for inference-only eval."""
    import torch

    from src.gluon.estimator import LagGemmaEstimator

    ckpt = Path(ckpt_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    
    # Use build_estimator_kwargs here instead of build_estimator(): checkpoint eval needs config-to-estimator translation without training-only defaults like run_dir/max_epochs.
    estimator = LagGemmaEstimator(
        **build_estimator_kwargs(
            model_config or {},
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            num_parallel_samples=num_parallel_samples,
            use_single_pass_sampling=use_single_pass_sampling,
            device=torch.device(device),
            ckpt_path=str(ckpt),
        )
    )
    module = estimator.create_module(use_kv_cache=True)
    predictor = build_predictor(estimator, module)
    return estimator, module, predictor


def finetune_tail_dart_from_ckpt(
    ckpt_path: str | Path,
    train_ds: ListDataset,
    val_ds: ListDataset,
    prediction_length: int,
    context_length: int,
    run_dir: Path,
    device: str = "cpu",
    max_epochs: int = 5,
    batch_size: int = 8,
    num_batches_per_epoch: int = 50,
    use_single_pass_sampling: bool = False,
    model_config: dict | None = None,
):
    """Fine-tune a model from pretrained checkpoint for few-shot adaptation."""
    import torch

    from src.gluon.estimator import LagGemmaEstimator

    ckpt = Path(ckpt_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    run_dir.mkdir(parents=True, exist_ok=True)

    estimator = LagGemmaEstimator(
        **build_estimator_kwargs(
            model_config or {},
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            device=torch.device(device),
            num_batches_per_epoch=num_batches_per_epoch,
            use_single_pass_sampling=use_single_pass_sampling,
            trainer_kwargs={"max_epochs": int(max_epochs), "default_root_dir": str(run_dir)},
            early_stopping_patience=3,
            ckpt_path=str(ckpt),
        ),
        allow_short_series_fallback=True,
    )
    fit = estimator.train_model(
        train_ds,
        val_ds,
        ckpt_path=str(ckpt),
        reset_training_state=True,
    )
    predictor = build_predictor(estimator, fit.model)
    return estimator, fit, predictor


def build_predictor(estimator, fitted_model):
    """Create a PyTorchPredictor from a trained estimator/module pair."""
    transformation = estimator.create_transformation()
    return estimator.create_predictor(transformation=transformation, module=fitted_model)


def forecast_to_point_array(forecast, point_forecast: str = "mean") -> np.ndarray:
    """Convert GluonTS forecast object to a 1D point forecast array."""
    point_forecast = str(point_forecast).strip().lower()
    if point_forecast not in {"mean", "median"}:
        raise ValueError("point_forecast must be either 'mean' or 'median'.")

    if point_forecast == "mean":
        mean = getattr(forecast, "mean", None)
        if mean is not None and not callable(mean):
            return np.asarray(mean, dtype=float).reshape(-1)

    if hasattr(forecast, "samples"):
        samples = np.asarray(forecast.samples, dtype=float)
        if samples.ndim == 1:
            return samples.reshape(-1)
        reducer = np.mean if point_forecast == "mean" else np.median
        return reducer(samples, axis=0).reshape(-1)

    if hasattr(forecast, "quantile"):
        return np.asarray(forecast.quantile(0.5), dtype=float).reshape(-1)

    return np.asarray(forecast, dtype=float).reshape(-1)


def forecast_to_mean_array(forecast) -> np.ndarray:
    """Convert GluonTS forecast object to 1D mean array."""
    return forecast_to_point_array(forecast, point_forecast="mean")


def _coerce_forecast_start(value) -> pd.Timestamp:
    """Normalize forecast start values into pandas timestamps."""
    if value is None or value is pd.NaT:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Period):
        return value.to_timestamp(how="start")
    return pd.Timestamp(value)


def _compute_tilde_q_metric(actual: np.ndarray, pred: np.ndarray) -> float:
    """Import and compute TILDE-Q lazily so notebook path setup can happen first."""
    try:
        from src.utils.tilde_q import compute_tilde_q
    except ImportError:
        from utils.tilde_q import compute_tilde_q

    return float(compute_tilde_q(y_true=actual, y_pred=pred))


def run_forecasts_raw(
    predictor,
    test_ds: ListDataset,
    num_samples: int = 100,
) -> tuple[list, list]:
    """Raw inference only: call make_evaluation_predictions and drain iterators.

    Returns (forecasts, tss). Time only this function to get fair inference
    timings comparable to Lag-Llama's run_lag_llama_predictions baseline.
    Post-processing (ForecastResult building) belongs outside the timed block.
    """
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=int(num_samples),
    )
    return list(forecast_it), list(ts_it)


def distribution_statistics(distribution, observed: np.ndarray) -> pd.DataFrame:
    """Compute exact one-step scores for GluonTS affine-wrapped distributions.

    ``distribution`` must be the GluonTS affine transform around a Student-t,
    Johnson SU, or gated ``Mixture`` forecast head (with Student-t/Johnson SU
    components), as returned by ``one_step_distribution``. This function
    evaluates the corresponding SciPy CDF/PPF analytically and uses the Torch
    distribution itself for log probability, so mixture NLL is the exact
    ``logsumexp`` mixture density. For mixture heads the per-row gate weights
    are exported as ``mixture_weight_{k}`` columns so downstream analysis can
    track weights by frequency, asset group, or volatility regime and test for
    expert collapse.
    """
    import torch
    from gluonts.torch.distributions import AffineTransformed
    from scipy import stats

    from src.model.distributions import JohnsonSU, Mixture 

    if not isinstance(distribution, AffineTransformed):
        raise TypeError(
            "distribution_statistics expects the GluonTS AffineTransformed "
            "distribution returned by one_step_distribution; "
            f"received {type(distribution).__name__}."
        )

    base = distribution.base_dist
    outer_loc = distribution.loc
    outer_scale = distribution.scale

    def as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float).reshape(-1)

    y = np.asarray(observed, dtype=float).reshape(-1)
    affine_loc = as_numpy(outer_loc)
    affine_scale = as_numpy(outer_scale)
    standardized = (y - affine_loc) / affine_scale

    def scipy_marginal(component):
        if isinstance(component, torch.distributions.StudentT):
            return "StudentT", stats.t(
                df=as_numpy(component.df),
                loc=as_numpy(component.loc),
                scale=as_numpy(component.scale),
            )
        if isinstance(component, JohnsonSU):
            return "JohnsonSU", stats.johnsonsu(
                a=as_numpy(component.gamma),
                b=as_numpy(component.delta),
                loc=as_numpy(component.loc),
                scale=as_numpy(component.scale),
            )
        raise TypeError(
            "Probabilistic evaluation supports StudentT and JohnsonSU "
            "(optionally as gated Mixture components); "
            f"received {type(component).__name__}."
        )

    mixture_weights: np.ndarray | None = None
    if isinstance(base, Mixture):
        component_names, scipy_components = zip(
            *(scipy_marginal(component) for component in base.components)
        )
        head = "Mixture(" + "+".join(component_names) + ")"
        mixture_weights = (
            base.mixture_probs.detach()
            .cpu()
            .numpy()
            .reshape(-1, len(scipy_components))
            .astype(float)
        )

        def mixture_cdf(value: np.ndarray) -> np.ndarray:
            return sum(
                mixture_weights[:, k]
                * np.asarray(component.cdf(value), dtype=float).reshape(-1)
                for k, component in enumerate(scipy_components)
            )

        pit = mixture_cdf(standardized)

        def quantile(probability: float) -> np.ndarray:
            # The mixture CDF has no closed-form inverse; bisect between the
            # extreme component quantiles, which always bracket the answer.
            component_ppfs = np.stack(
                [
                    np.broadcast_to(
                        np.asarray(component.ppf(probability), dtype=float).reshape(-1),
                        y.shape,
                    )
                    for component in scipy_components
                ],
                axis=0,
            )
            lower = component_ppfs.min(axis=0)
            upper = component_ppfs.max(axis=0)
            for _ in range(80):
                mid = 0.5 * (lower + upper)
                below = mixture_cdf(mid) < probability
                lower = np.where(below, mid, lower)
                upper = np.where(below, upper, mid)
            return affine_loc + affine_scale * 0.5 * (lower + upper)

    else:
        head, scipy_dist = scipy_marginal(base)
        pit = np.asarray(scipy_dist.cdf(standardized), dtype=float).reshape(-1)

        def quantile(probability: float) -> np.ndarray:
            return affine_loc + affine_scale * np.asarray(
                scipy_dist.ppf(probability), dtype=float
            ).reshape(-1)

    observed_tensor = torch.as_tensor(
        y,
        dtype=distribution.mean.dtype,
        device=distribution.mean.device,
    ).reshape(distribution.batch_shape)
    nll = -as_numpy(distribution.log_prob(observed_tensor))

    frame = pd.DataFrame(
        {
            "distribution_head": head,
            "observed_return": y,
            "pit": np.asarray(pit, dtype=float).reshape(-1),
            "test_nll": nll,
            "pi80_lower": quantile(0.10),
            "pi80_upper": quantile(0.90),
            "pi95_lower": quantile(0.025),
            "pi95_upper": quantile(0.975),
        }
    )
    if mixture_weights is not None:
        for k in range(mixture_weights.shape[1]):
            frame[f"mixture_weight_{k + 1}"] = mixture_weights[:, k]
    return frame


def run_one_step_probabilistic_forecasts(
    *,
    predictor,
    test_ds: ListDataset,
    observed_by_item: dict[str, float],
) -> pd.DataFrame:
    """Evaluate exact distributions for the same held-out inputs as GluonTS.

    ``observed_by_item`` must map each unique test-window item id to its held-out
    value. The returned frame preserves those identifiers and forecast starts.
    """
    import torch
    from gluonts.dataset.loader import InferenceDataLoader
    from gluonts.dataset.split import split
    from gluonts.torch.batchify import batchify
    from gluonts.transform import SelectFields

    if predictor.prediction_length != 1:
        raise ValueError("Probabilistic forecast export currently requires one-step forecasts.")

    _, test_template = split(test_ds, offset=-1)
    test_data = test_template.generate_instances(1)
    transform = predictor.input_transform + SelectFields(
        predictor.input_names + predictor.required_fields,
        allow_missing=True,
    )
    loader = InferenceDataLoader(
        test_data.input,
        transform=transform,
        batch_size=predictor.batch_size,
        stack_fn=lambda data: batchify(data, predictor.device),
    )

    rows: list[pd.DataFrame] = []
    predictor.prediction_net.eval()
    with torch.no_grad():
        for batch in loader:
            kwargs = {name: batch[name] for name in predictor.input_names}
            distribution = predictor.prediction_net.one_step_distribution(**kwargs)
            item_ids = [str(value) for value in batch["item_id"]]
            observed = np.asarray([observed_by_item[item_id] for item_id in item_ids])
            frame = distribution_statistics(distribution, observed)
            frame.insert(0, "forecast_start", [
                _coerce_forecast_start(value) for value in batch["forecast_start"]
            ])
            frame.insert(0, "item_id", item_ids)
            rows.append(frame)

    if not rows:
        return pd.DataFrame(
            columns=[
                "item_id", "forecast_start", "distribution_head", "observed_return",
                "pit", "test_nll", "pi80_lower", "pi80_upper", "pi95_lower", "pi95_upper",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def materialize_forecast_results(
    forecasts: list,
    tss: list,
    series_list: list[dict],
    prediction_length: int,
    point_forecast: str = "mean",
) -> list[ForecastResult]:
    """Convert raw GluonTS forecast outputs into ForecastResult objects.

    Call this outside the timed block, after run_forecasts_raw, so that
    Python post-processing does not inflate inference timing measurements.
    """
    results: list[ForecastResult] = []
    for entry, forecast, ts in zip(series_list, forecasts, tss):
        pred = forecast_to_point_array(forecast, point_forecast=point_forecast)[:prediction_length]
        series = np.asarray(ts, dtype=float).reshape(-1)
        actual = series[-prediction_length:].reshape(-1)
        history = series[:-prediction_length].reshape(-1)
        results.append(
            ForecastResult(
                item_id=str(entry.get("item_id", "unknown")),
                forecast_start=_coerce_forecast_start(getattr(forecast, "start_date", pd.NaT)),
                history=history,
                actual=actual,
                prediction=pred,
            )
        )
    return results


def run_forecasts(
    predictor,
    test_ds: ListDataset,
    prediction_length: int,
    num_samples: int = 100,
    point_forecast: str = "mean",
) -> list[ForecastResult]:
    """Run one holdout forecast for each series in test_ds.

    Convenience wrapper around run_forecasts_raw + materialize_forecast_results.
    For timing-sensitive code, call the two phases separately.
    """
    forecasts, tss = run_forecasts_raw(predictor, test_ds, num_samples=num_samples)
    series_list = list(test_ds)
    return materialize_forecast_results(
        forecasts,
        tss,
        series_list,
        prediction_length,
        point_forecast=point_forecast,
    )


def evaluate_forecasts(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> tuple[float, float, float, float, pd.DataFrame]:
    """Compute error metrics after requiring complete finite forecasts."""
    if actual_df["unique_id"].duplicated().any():
        raise ValueError("Expected exactly one actual observation per unique_id.")

    expected_ids = set(actual_df["unique_id"])
    forecast_counts = forecast_df["unique_id"].value_counts(dropna=False)
    missing_ids = expected_ids - set(forecast_counts.index)
    duplicate_ids = set(forecast_counts[forecast_counts != 1].index)
    unexpected_ids = set(forecast_counts.index) - expected_ids
    if missing_ids or duplicate_ids or unexpected_ids:
        raise ValueError(
            f"Forecast coverage mismatch: missing={sorted(map(str, missing_ids))}, "
            f"duplicate={sorted(map(str, duplicate_ids))}, "
            f"unexpected={sorted(map(str, unexpected_ids))}"
        )

    forecast_values = pd.to_numeric(forecast_df["Forecast"], errors="coerce")
    if not np.isfinite(forecast_values).all():
        raise ValueError("Expected one finite forecast per unique_id.")

    merged = pd.merge(
        actual_df,
        forecast_df,
        on="unique_id",
        how="inner",
        validate="one_to_one",
    )
    merged["Residuals"] = merged["Actual"] - merged["Forecast"]
    mse = float((merged["Residuals"] ** 2).mean())
    mae = float(merged["Residuals"].abs().mean())
    rmse = float(np.sqrt(mse))
    merged["RMSE"] = rmse
    merged["MAE"] = mae
    actual_abs = merged["Actual"].abs()
    mape_valid = actual_abs > 0
    mape = (
        float(
            (
                merged.loc[mape_valid, "Residuals"].abs()
                / actual_abs[mape_valid]
            ).mean()
            * 100.0
        )
        if mape_valid.any()
        else np.nan
    )
    merged["MAPE"] = np.where(
        mape_valid,
        (merged["Residuals"].abs() / actual_abs) * 100.0,
        np.nan,
    )
    return mse, mae, rmse, mape, merged


def compute_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    history: np.ndarray | None = None,
    seasonal_error: float | None = None,  # naive scale: mean(|y[t] - y[t-m]|) for seasonality m; at m=1 collapses to mean(|diff(history)|)
) -> dict[str, float]:
    """Compute point-forecast metrics after filtering non-finite values.

    All returned metrics are error or loss quantities, so lower values are
    better for ``mae``, ``mse``, ``rmse``, ``mape``, ``smape``, ``mase``, and
    ``tilde_q``. ``tilde_q`` is lower-is-better because it is computed from
    ``tildeq_loss(...)`` in ``src.utils.tilde_q``.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)

    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if actual.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "mase": np.nan,
            "tilde_q": np.nan,
        }

    err = actual - pred
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape_valid = np.abs(actual) > 0
    mape = float((abs_err[mape_valid] / np.abs(actual[mape_valid])).mean() * 100.0) if mape_valid.any() else np.nan
    smape_denom = np.abs(actual) + np.abs(pred)
    smape_valid = smape_denom > 0
    smape = float((2.0 * abs_err[smape_valid] / smape_denom[smape_valid]).mean() * 100.0) if smape_valid.any() else np.nan
    if seasonal_error is None:
        history_arr = np.asarray([]) if history is None else np.asarray(history, dtype=float).reshape(-1)
        history_arr = history_arr[np.isfinite(history_arr)]
        seasonal_error = (
            float(calculate_seasonal_error(history_arr, seasonality=1))
            if history_arr.size > 1
            else np.nan
        )
    mase = (
        float(gluonts_mase(target=actual, forecast=pred, seasonal_error=seasonal_error))
        if np.isfinite(seasonal_error) and abs(seasonal_error) > METRIC_EPS
        else np.nan
    )
    try:
        tilde_q = _compute_tilde_q_metric(actual=actual, pred=pred)
    except Exception:
        tilde_q = np.nan

    return {
        "n": int(actual.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "mase": mase,
        "tilde_q": tilde_q,
    }


def metrics_frame(results: list[ForecastResult]) -> pd.DataFrame:
    """Convert forecast results to a per-series metrics DataFrame."""
    rows: list[dict] = []
    for r in results:
        row = {"item_id": r.item_id, "forecast_start": r.forecast_start}
        row.update(compute_metrics(r.actual, r.prediction, history=r.history))
        rows.append(row)
    return pd.DataFrame(rows)


def _finite_history(result: ForecastResult) -> np.ndarray:
    history = np.asarray(result.history, dtype=float).reshape(-1)
    return history[np.isfinite(history)]


def _history_naive_scale(history: np.ndarray) -> float:
    if history.size <= 1:
        return np.nan
    return float(np.mean(np.abs(np.diff(history))))


def _fixed_train_seasonal_error(ordered: list[ForecastResult]) -> float:
    return _history_naive_scale(_finite_history(ordered[0]))


def _blockwise_mase_stats(ordered: list[ForecastResult]) -> tuple[float, int]:
    block_mase_values = []
    for result in ordered:
        scale = _history_naive_scale(_finite_history(result))
        if not np.isfinite(scale) or scale <= METRIC_EPS:
            continue

        actual = np.asarray(result.actual, dtype=float).reshape(-1)
        pred = np.asarray(result.prediction, dtype=float).reshape(-1)
        mask = np.isfinite(actual) & np.isfinite(pred)
        if not mask.any():
            continue

        block_mae = float(np.mean(np.abs(actual[mask] - pred[mask])))
        block_mase_values.append(block_mae / scale)

    mase = float(np.mean(block_mase_values)) if block_mase_values else np.nan
    return mase, len(block_mase_values)


def _aggregate_results_by_item(
    results: list[ForecastResult],
    seasonal_error_fn: Callable[[list[ForecastResult]], float] | None,
    blockwise_mase: bool = False,
) -> pd.DataFrame:
    """Aggregate one-step result groups into one metric row per series."""
    if not results:
        columns = ["item_id", "forecast_start", "n", *DEFAULT_METRIC_COLUMNS]
        if blockwise_mase:
            columns.extend(["mase_n", "mase_coverage"])
        return pd.DataFrame(columns=columns)

    grouped: dict[str, list[ForecastResult]] = {}
    for result in results:
        grouped.setdefault(str(result.item_id), []).append(result)

    rows: list[dict] = []
    for item_id, series_results in grouped.items():
        ordered = sorted(series_results, key=lambda r: pd.Timestamp(r.forecast_start))
        actual = np.concatenate([np.asarray(r.actual, dtype=float).reshape(-1) for r in ordered])
        pred = np.concatenate([np.asarray(r.prediction, dtype=float).reshape(-1) for r in ordered])
        seasonal_error = seasonal_error_fn(ordered) if seasonal_error_fn is not None else np.nan
        row = {
            "item_id": item_id,
            "forecast_start": pd.Timestamp(ordered[0].forecast_start),
        }
        row.update(compute_metrics(actual, pred, seasonal_error=seasonal_error))
        if blockwise_mase:
            row["mase"], row["mase_n"] = _blockwise_mase_stats(ordered)
            row["mase_coverage"] = row["mase_n"] / len(ordered)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["item_id", "forecast_start"]).reset_index(drop=True)


def aggregate_fixed_train_rolling_results_by_item(results: list[ForecastResult]) -> pd.DataFrame:
    """Aggregate fixed-train rolling results using the initial train history scale."""
    return _aggregate_results_by_item(results, _fixed_train_seasonal_error)


def aggregate_chunked_block_results_by_item(results: list[ForecastResult]) -> pd.DataFrame:
    """Aggregate chunks and disclose valid-block MASE count and coverage."""
    return _aggregate_results_by_item(results, None, blockwise_mase=True)


def aggregate_metrics_by_asset_type(
    metrics_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...] | None = None,
    *,
    item_id_col: str = "item_id",
    asset_type_col: str = "asset_type",
    separator: str = "__",
    groupby_columns: list[str] | tuple[str, ...] | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-series metrics into per-asset-type metrics."""
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; compute metrics before aggregating.")

    if item_id_col not in metrics_df.columns:
        raise ValueError(f"metrics_df must include '{item_id_col}'.")

    metrics = list(metric_columns or DEFAULT_METRIC_COLUMNS)
    missing_metrics = [column for column in metrics if column not in metrics_df.columns]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"metrics_df is missing required metric columns: {missing}.")

    grouping = list(groupby_columns or [])
    missing_groupby = [column for column in grouping if column not in metrics_df.columns]
    if missing_groupby:
        missing = ", ".join(missing_groupby)
        raise ValueError(f"metrics_df is missing required grouping columns: {missing}.")

    grouped = metrics_df[[item_id_col, *grouping, *metrics]].copy()
    grouped[asset_type_col] = grouped[item_id_col].astype(str).str.split(separator, n=1).str[0]
    grouped = (
        grouped.groupby([*grouping, asset_type_col], as_index=False)[metrics]
        .agg(agg)
        .sort_values([*grouping, asset_type_col])
        .reset_index(drop=True)
    )
    return grouped


def macro_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro-mean metrics and aggregate counts across series."""
    if metrics_df.empty:
        return pd.DataFrame()
    cols = DEFAULT_METRIC_COLUMNS
    summary = metrics_df[cols].mean(numeric_only=True).to_frame().T
    summary.insert(0, "n_series", len(metrics_df))
    summary.insert(1, "n_points_total", int(metrics_df["n"].sum()))
    return summary


__all__ = [
    "ForecastResult",
    "build_predictor",
    "compute_metrics",
    "aggregate_chunked_block_results_by_item",
    "aggregate_fixed_train_rolling_results_by_item",
    "aggregate_metrics_by_asset_type",
    "build_chunk_holdout_entries",
    "ensure_paths",
    "finetune_tail_dart_from_ckpt",
    "few_shot_split_for_holdout",
    "listdataset",
    "load_tail_dart_from_ckpt",
    "load_dataset_frame",
    "macro_summary",
    "make_train_and_val_from_train_entries",
    "materialize_forecast_results",
    "metrics_frame",
    "prepare_temporal_series_cache",
    "returns_frame",
    "run_forecasts",
    "run_forecasts_raw",
    "distribution_statistics",
    "run_one_step_probabilistic_forecasts",
    "temporal_fixed_train_rolling_split",
    "temporal_split_for_holdout",
    "temporal_train_val_holdout_split",
    "_dataset_frequency",
]
