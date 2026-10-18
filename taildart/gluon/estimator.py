# Copyright 2024 Arjun Ashok
# Modifications Copyright 2026 Chiang Qin Zhi
#
# Modified as part of TailDart project (replacing LLaMA with Gemma backbone)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
from datetime import datetime
import os
import random
import re
import shutil
import time
import warnings
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.torch.distributions import (
    DistributionOutput,
    LaplaceOutput,
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    DummyValueImputation,
    InstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

from data.dataset_loader import SingleInstanceSampler
from taildart.gluon.torch_module import TailDartModule
from taildart.model.distributions import (
    HansenSkewTOutput,
    JohnsonSUOutput,
    NormalInverseGaussianOutput,
    StudentTJohnsonSUMixtureOutput,
    StudentTMixtureOutput,
)

ANSI_LABEL = "\033[1;36m"
ANSI_RESET = "\033[0m"

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]

# Forecast-head names exposed to the pretrain/finetune CLIs. Must mirror
# `TailDartEstimator._resolve_distr_output`.
DISTR_OUTPUT_CHOICES = (
    "studentT",
    "student_t",
    "neg_bin",
    "negative_binomial",
    "gaussian",
    "normal",
    "laplace",
    "skew_t",
    "skewt",
    "hansen_skew_t",
    "nig",
    "normal_inverse_gaussian",
    "jsu",
    "johnson_su",
    "johnsonsu",
    "mixture_t_jsu",
    "mixture_t_t",
)


class _WarmupDecayScheduler:
    """Batch-stepped warmup + optional decay scheduler wrapper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        decay_name: str,
        decay_args: dict[str, Any],
        num_batches_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_steps = int(warmup_steps)
        self.decay_name = str(decay_name).strip().lower()
        self.decay_args = dict(decay_args)
        self.num_batches_per_epoch = max(1, int(num_batches_per_epoch))
        self.global_step = 0
        self.post_warmup_steps = 0
        self._decay_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def _build_decay_scheduler(
        self,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if self.decay_name == "none":
            return None
        if self.decay_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.decay_args
            )
        if self.decay_name == "cosine_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.decay_args
            )
        raise ValueError(f"Unsupported decay.name: {self.decay_name}")

    def _set_lr_scale(self, scale: float) -> None:
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * float(scale)

    def _step_warmup(self) -> bool:
        if self.warmup_steps <= 0 or self.global_step > self.warmup_steps:
            return False
        progress = float(self.global_step) / float(self.warmup_steps)
        self._set_lr_scale(progress)
        return True

    def _step_decay(self) -> None:
        if self._decay_scheduler is None and self.decay_name != "none":
            self._decay_scheduler = self._build_decay_scheduler()
        if self._decay_scheduler is None:
            self._set_lr_scale(1.0)
            return
        self.post_warmup_steps += 1
        if self.decay_name == "cosine_restarts":
            # Keep phase anchored at warmup completion.
            epoch_progress = float(self.post_warmup_steps) / float(self.num_batches_per_epoch)
            self._decay_scheduler.step(epoch_progress)
            return
        self._decay_scheduler.step()

    def step(self) -> None:
        self.global_step += 1
        if self._step_warmup():
            return
        self._step_decay()

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "post_warmup_steps": int(self.post_warmup_steps),
            "warmup_steps": int(self.warmup_steps),
            "decay_name": self.decay_name,
            "decay_args": dict(self.decay_args),
            "decay_scheduler_state_dict": (
                self._decay_scheduler.state_dict() if self._decay_scheduler is not None else None
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_step = int(state_dict.get("global_step", 0))
        self.post_warmup_steps = int(state_dict.get("post_warmup_steps", 0))
        decay_state = state_dict.get("decay_scheduler_state_dict")
        if decay_state is not None and self._decay_scheduler is None and self.decay_name != "none":
            self._decay_scheduler = self._build_decay_scheduler()
        if self._decay_scheduler is not None and decay_state is not None:
            self._decay_scheduler.load_state_dict(decay_state)
        if self.global_step <= self.warmup_steps and self.warmup_steps > 0:
            progress = float(self.global_step) / float(self.warmup_steps)
            self._set_lr_scale(progress)


def _gluonts_collate_fn(batch: List[dict]) -> dict:
    """Stack a list of record dicts into a dict of tensors."""
    out = {}
    for key in batch[0]:
        values = [torch.as_tensor(item[key]) for item in batch]
        out[key] = torch.stack(values)
    return out


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class _PreloadedTrainingDataset(torch.utils.data.Dataset):
    """Preload transformed series and sample one training window per __getitem__."""

    def __init__(
        self,
        data: Dataset,
        transformation: Transformation,
        past_length: int,
        prediction_length: int,
        time_feat: bool,
        dataset_weights: Optional[Dict[int, float]] = None,
        pad_augment_prob: float = 0.0,
        allow_short_series_fallback: bool = False,
    ) -> None:
        self.past_length = int(past_length)
        self.prediction_length = int(prediction_length)
        self.time_feat = bool(time_feat)
        self.pad_augment_prob = float(pad_augment_prob)
        self._series: List[Dict[str, Any]] = []
        fallback_series: List[Dict[str, Any]] = []
        max_target_len = 0

        transformed = transformation.apply(data, is_train=True)
        min_past = self.past_length
        min_future = self.prediction_length
        for entry in transformed:
            target = np.asarray(entry[FieldName.TARGET], dtype=np.float32)
            observed = np.asarray(entry[FieldName.OBSERVED_VALUES], dtype=np.float32)
            if target.shape[-1] != observed.shape[-1]:
                raise ValueError("target/observed_values length mismatch.")
            max_target_len = max(max_target_len, int(target.shape[-1]))

            lower = int(min_past)
            upper = int(target.shape[-1] - min_future)
            if upper < 1:
                continue

            series_record: Dict[str, Any] = {
                "target": np.ascontiguousarray(target),
                "observed_values": np.ascontiguousarray(observed),
                "lower": max(1, lower),
                "upper": upper,
                "start": entry.get(FieldName.START),
                "freq": entry.get("freq"),
                "item_id": entry.get(FieldName.ITEM_ID),
                "data_id": entry.get("data_id"),
            }

            if self.time_feat:
                feat_time = np.asarray(entry[FieldName.FEAT_TIME], dtype=np.float32)
                # GluonTS keeps FEAT_TIME as (num_features, time_steps).
                if feat_time.ndim != 2:
                    raise ValueError("Expected 2D time features (features, time).")
                if feat_time.shape[-1] < target.shape[-1]:
                    raise ValueError("time_feat length shorter than target.")
                series_record["time_feat"] = np.ascontiguousarray(feat_time)

            if upper >= lower:
                self._series.append(series_record)
            elif allow_short_series_fallback:
                # Keep a fallback candidate and enable left-padding only if strict
                # sampling bounds would otherwise remove every series.
                series_record["lower"] = 1
                fallback_series.append(series_record)

        if allow_short_series_fallback and not self._series and fallback_series:
            self._series = fallback_series

        if not self._series:
            raise ValueError(
                "No valid training series after applying sampling bounds. "
                f"Required history={self.past_length}, future={self.prediction_length}, "
                f"max target length seen={max_target_len}."
            )

        # Compute per-series sampling weights from dataset-level weights.
        # For each series in dataset i: weight = dataset_weights[i] / n_series_in_dataset_i
        # This preserves the intended inter-dataset sampling proportions.
        self.sample_weights: Optional[List[float]] = None
        if dataset_weights is not None:
            count_per_dataset: Dict[int, int] = {}
            for s in self._series:
                did = s["data_id"]
                if did in dataset_weights:
                    count_per_dataset[did] = count_per_dataset.get(did, 0) + 1
            # Only activate weighted sampling if every series has a known data_id.
            if all(s["data_id"] in dataset_weights for s in self._series):
                self.sample_weights = [
                    dataset_weights[s["data_id"]] / count_per_dataset[s["data_id"]]
                    for s in self._series
                ]

    def __len__(self) -> int:
        return len(self._series)

    def __getitem__(self, index: int) -> dict:
        series = self._series[int(index) % len(self._series)]
        lower = series["lower"]
        upper = series["upper"]
        t = random.randint(lower, upper)
        past_start = t - self.past_length
        past_end = t
        future_end = t + self.prediction_length

        target = series["target"]
        observed = series["observed_values"]
        if past_start >= 0:
            past_target = target[..., past_start:past_end].copy()
            past_observed = observed[..., past_start:past_end].copy()
            pad_len = 0
        else:
            # Left-pad short histories with zeros so checkpoint models with large
            # lag requirements can still fine-tune on short datasets.
            pad_len = -past_start
            past_target = np.zeros(target.shape[:-1] + (self.past_length,), dtype=np.float32)
            past_observed = np.zeros(observed.shape[:-1] + (self.past_length,), dtype=np.float32)
            past_target[..., pad_len:] = target[..., :past_end]
            past_observed[..., pad_len:] = observed[..., :past_end]

        # Padded-lags augmentation: zero out a random left prefix to simulate
        # inference on series with insufficient history for full lag coverage.
        # Do not compound synthetic padding on histories that already contain
        # unobserved values (including NaN left-padding from short series).
        history_is_fully_observed = bool(np.all(past_observed > 0.0))
        if (
            history_is_fully_observed
            and self.pad_augment_prob > 0.0
            and random.random() < self.pad_augment_prob
        ):
            keep_ratio = random.uniform(0.2, 1.0)
            keep_len = max(1, int(keep_ratio * self.past_length))
            pad_len = self.past_length - keep_len
            past_target[..., :pad_len] = 0.0
            past_observed[..., :pad_len] = 0.0

        sample = {
            "past_target": past_target,
            "future_target": target[..., past_end:future_end],
            "past_observed_values": past_observed,
            "future_observed_values": observed[..., past_end:future_end],
        }

        if self.time_feat:
            feat_time = series["time_feat"]
            # Time features (day-of-week, month, etc.) are NOT zeroed during
            # padding augmentation.  They encode calendar position, not observed
            # values — timestamps are known even for a short series that lacks
            # historical target values.
            if past_start >= 0:
                sample["past_time_feat"] = feat_time[..., past_start:past_end].T
            else:
                n_feat = int(feat_time.shape[0])
                past_time_feat = np.zeros((self.past_length, n_feat), dtype=np.float32)
                past_time_feat[pad_len:, :] = feat_time[..., :past_end].T
                sample["past_time_feat"] = past_time_feat
            sample["future_time_feat"] = feat_time[..., past_end:future_end].T

        return sample


class TailDartEstimator:
    """Pure-PyTorch estimator API for TailDart forecasting.

    Most ``self.*`` attributes are constructor-time configuration cached for
    later use by data loaders, module construction, and checkpointing.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int] = None,
        input_size: int = 1,
        ffn_type: str = "dense",
        n_layer: int = 1,
        taildart_hidden_size: int = 144,
        taildart_intermediate_size: int = 576,
        taildart_num_heads: int = 8,
        taildart_num_kv_heads: int = 4,
        taildart_head_dim: int = 18,
        taildart_rms_norm_eps: float = 1e-6,
        moe_num_experts: int = 3,
        moe_top_k: int = 1,
        moe_aux_loss_weight: float = 0.01,
        moe_router_jitter_noise: float = 0.0,
        max_context_length: int = 2048,
        rope_scaling=None,
        scaling: Optional[str] = "mean",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        freq_mask_rate: float = 0.1,
        freq_mixing_rate: float = 0.1,
        jitter_prob: float = 0.0,
        jitter_sigma: float = 0.03,
        scaling_prob: float = 0.0,
        scaling_sigma: float = 0.1,
        rotation_prob: float = 0.0,
        permutation_prob: float = 0.0,
        permutation_max_segments: int = 5,
        permutation_seg_mode: str = "equal",
        magnitude_warp_prob: float = 0.0,
        magnitude_warp_sigma: float = 0.2,
        magnitude_warp_knot: int = 4,
        time_warp_prob: float = 0.0,
        time_warp_sigma: float = 0.2,
        time_warp_knot: int = 4,
        window_slice_prob: float = 0.0,
        window_slice_reduce_ratio: float = 0.9,
        window_warp_prob: float = 0.0,
        window_warp_window_ratio: float = 0.1,
        window_warp_scales: list = [0.5, 2.0],
        distr_output: Any = "studentT",
        loss: DistributionLoss = NegativeLogLikelihood(),
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        train_num_workers: int = 4,
        train_prefetch_factor: int = 2,
        train_pin_memory: bool = True,
        train_dataset_weights: Optional[Dict[int, float]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        time_feat: bool = False,
        dropout: float = 0.0,
        lags_seq: list = ["Q", "M", "W", "D", "H", "T", "S"],
        max_lag: Optional[int] = None,
        data_id_to_name_map: dict = {},
        warmup: Optional[dict] = None,
        decay: Optional[dict] = None,
        track_loss_per_series: bool = False,
        ckpt_path: Optional[str] = None,
        nonnegative_pred_samples: bool = False,
        use_single_pass_sampling: bool = False,
        pad_augment_prob: float = 0.0,
        allow_short_series_fallback: bool = False,
        val_keep_min: float = 1.0,
        val_keep_max: float = 1.0,
        val_keep_seed: int = 42,
        early_stopping_patience: int = 50,
        validate_at_steps: Optional[List[int]] = None,
        max_training_steps: Optional[int] = None,
        checkpoint_model_name: str = "taildart",
        checkpoint_version: str = "v1",
        checkpoint_run_date: Optional[str] = None,
        save_canonical_checkpoints: bool = True,
        save_named_checkpoints: bool = True,
        device: Any = None,
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)

        self.trainer_kwargs = default_trainer_kwargs
        self.limit_val_batches = self.trainer_kwargs.get("limit_val_batches")

        self.scaling = scaling
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.max_context_length = max_context_length
        if use_single_pass_sampling and prediction_length > 1:
            raise ValueError(
                "use_single_pass_sampling=True is only valid for prediction_length=1; "
                "multi-step forecasts would condition all samples on a shared mean trajectory."
            )

        # lag_ub in GluonTS is 1-based; add 1 so that the 0-based max equals max_lag.
        lag_ub = (max_lag + 1) if max_lag is not None else 1200
        lag_indices = []
        for freq in lags_seq:
            lag_indices.extend(get_lags_for_frequency(freq_str=freq, num_default_lags=1, lag_ub=lag_ub))

        if len(lag_indices):
            # Convert frequency aliases into zero-based lag offsets expected downstream.
            self.lags_seq = sorted(set(lag_indices))
            self.lags_seq = [lag_index - 1 for lag_index in self.lags_seq]
        else:
            self.lags_seq = []

        if max_lag is not None:
            print(
                f"{ANSI_LABEL}[TailDartEstimator] max_lag={max_lag}{ANSI_RESET}: "
                f"{len(self.lags_seq)} lag(s), effective max lag: {max(self.lags_seq, default=0)}"
            )

        self.ffn_type = str(ffn_type).strip().lower()
        if self.ffn_type not in {"dense", "moe"}:
            raise ValueError("`ffn_type` must be one of: dense, moe.")
        self.n_layer = n_layer
        self.taildart_hidden_size = taildart_hidden_size
        self.taildart_intermediate_size = taildart_intermediate_size
        self.taildart_num_heads = taildart_num_heads
        self.taildart_num_kv_heads = taildart_num_kv_heads
        self.taildart_head_dim = taildart_head_dim
        self.taildart_rms_norm_eps = taildart_rms_norm_eps
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)
        self.moe_router_jitter_noise = float(moe_router_jitter_noise)
        if self.ffn_type == "moe":
            if self.moe_num_experts < 2:
                raise ValueError("`moe_num_experts` must be >= 2.")
            if self.moe_top_k != 1:
                raise ValueError("`moe_top_k` must be exactly 1.")
        self.rope_scaling = rope_scaling

        self.lr = lr
        self.weight_decay = weight_decay
        self.distr_output = self._resolve_distr_output(distr_output)
        self.num_parallel_samples = num_parallel_samples
        self.loss = loss
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.train_num_workers = int(train_num_workers)
        self.train_prefetch_factor = int(train_prefetch_factor)
        self.train_pin_memory = bool(train_pin_memory)
        self.train_dataset_weights = train_dataset_weights
        self.nonnegative_pred_samples = nonnegative_pred_samples
        self.use_single_pass_sampling = bool(use_single_pass_sampling)
        self.pad_augment_prob = float(pad_augment_prob)
        self.allow_short_series_fallback = bool(allow_short_series_fallback)
        self.val_keep_min = float(val_keep_min)
        self.val_keep_max = float(val_keep_max)
        self.val_keep_seed = int(val_keep_seed)
        if not (0.0 < self.val_keep_min <= 1.0):
            raise ValueError("val_keep_min must be in (0.0, 1.0].")
        if not (0.0 < self.val_keep_max <= 1.0):
            raise ValueError("val_keep_max must be in (0.0, 1.0].")
        if self.val_keep_min > self.val_keep_max:
            raise ValueError("val_keep_min cannot be greater than val_keep_max.")
        self.use_mixed_keep_validation = not (
            self.val_keep_min == 1.0 and self.val_keep_max == 1.0
        )

        self.train_sampler = train_sampler or SingleInstanceSampler(
            min_past=self.context_length + max(self.lags_seq),
            min_future=prediction_length,
        )
        # InstanceSplitter already left-pads histories shorter than past_length,
        # which is the same policy used by the zero-shot prediction path.  When
        # short-series fallback is enabled, allow validation to reach that
        # padding logic instead of filtering every short series out first.
        validation_min_past = (
            0
            if self.allow_short_series_fallback
            else self.context_length + max(self.lags_seq)
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_past=validation_min_past,
            min_future=prediction_length,
        )

        self.aug_prob = aug_prob
        self.freq_mask_rate = freq_mask_rate
        self.freq_mixing_rate = freq_mixing_rate
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.scaling_prob = scaling_prob
        self.scaling_sigma = scaling_sigma
        self.rotation_prob = rotation_prob
        self.permutation_prob = permutation_prob
        self.permutation_max_segments = permutation_max_segments
        self.permutation_seg_mode = permutation_seg_mode
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.magnitude_warp_knot = magnitude_warp_knot
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knot = time_warp_knot
        self.window_slice_prob = window_slice_prob
        self.window_slice_reduce_ratio = window_slice_reduce_ratio
        self.window_warp_prob = window_warp_prob
        self.window_warp_window_ratio = window_warp_window_ratio
        self.window_warp_scales = window_warp_scales
        self.track_loss_per_series = track_loss_per_series

        self.time_feat = time_feat
        self.dropout = dropout
        self.data_id_to_name_map = data_id_to_name_map
        self.ckpt_path = ckpt_path

        warmup_cfg = dict(warmup or {})
        self.warmup_name = str(warmup_cfg.get("name", "none")).strip().lower()
        warmup_epochs = warmup_cfg.get("epochs", 0)
        if self.warmup_name not in {"none", "linear"}:
            raise ValueError("`warmup.name` must be one of: none, linear.")
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("`warmup.epochs` must be an integer >= 0.")
        self.warmup_epochs = int(warmup_epochs)

        decay_cfg = dict(decay or {})
        self.decay_name = str(decay_cfg.get("name", "none")).strip().lower()
        self.decay_args = dict(decay_cfg.get("args", {}))
        if self.decay_name not in {"none", "cosine", "cosine_restarts"}:
            raise ValueError("`decay.name` must be one of: none, cosine, cosine_restarts.")
        if not isinstance(decay_cfg.get("args", {}), dict):
            raise ValueError("`decay.args` must be an object/dict.")
        self.early_stopping_patience = int(early_stopping_patience)
        if validate_at_steps is None:
            self.validate_at_steps = None
        else:
            normalized_steps = sorted({int(step) for step in validate_at_steps})
            if not normalized_steps or normalized_steps[0] < 1:
                raise ValueError("validate_at_steps must contain positive global optimizer steps.")
            self.validate_at_steps = normalized_steps
            if self.early_stopping_patience >= 0:
                warnings.warn(
                    "early_stopping_patience counts validation events, not epochs, when "
                    "validate_at_steps is provided",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if max_training_steps is not None and int(max_training_steps) < 1:
            raise ValueError("max_training_steps must be a positive integer when provided.")
        self.max_training_steps = (
            None if max_training_steps is None else int(max_training_steps)
        )
        self.checkpoint_model_name = checkpoint_model_name
        self.checkpoint_version = checkpoint_version
        self.checkpoint_run_date = checkpoint_run_date
        self.save_canonical_checkpoints = bool(save_canonical_checkpoints)
        self.save_named_checkpoints = bool(save_named_checkpoints)
        self._cached_training_loader: Optional[torch.utils.data.DataLoader] = None
        self._cached_training_loader_key: Optional[tuple[int, int, bool]] = None
        # Populated during training when named checkpoint copies are written.
        self.last_named_ckpt_path: Optional[str] = None
        self.best_named_ckpt_path: Optional[str] = None
        if device is None:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def _resolve_distr_output(self, distr_output: Any) -> DistributionOutput:
        if isinstance(distr_output, DistributionOutput):
            return distr_output
        if not isinstance(distr_output, str):
            raise TypeError(
                "`distr_output` must be a string key or a DistributionOutput instance."
            )

        normalized = distr_output.strip().lower()
        if normalized in {"studentt", "student_t"}:
            return StudentTOutput()
        if normalized in {"neg_bin", "negative_binomial", "negativebinomial"}:
            return NegativeBinomialOutput()
        if normalized in {"gaussian", "normal"}:
            return NormalOutput()
        if normalized == "laplace":
            return LaplaceOutput()
        if normalized in {"skew_t", "skewt", "hansen_skew_t"}:
            return HansenSkewTOutput()
        if normalized in {"nig", "normal_inverse_gaussian"}:
            return NormalInverseGaussianOutput()
        if normalized in {"jsu", "johnson_su", "johnsonsu"}:
            return JohnsonSUOutput()
        if normalized == "mixture_t_jsu":
            return StudentTJohnsonSUMixtureOutput()
        if normalized == "mixture_t_t":
            return StudentTMixtureOutput(num_components=2)

        raise ValueError(
            "Unsupported distr_output "
            f"`{distr_output}`. Expected one of: studentT, neg_bin, gaussian, "
            "laplace, skew_t, nig, johnson_su, mixture_t_jsu, mixture_t_t."
        )

    @property
    def _use_amp(self) -> bool:
        """True when the current device supports BF16 autocast (CUDA only)."""
        return self.device.type == "cuda"

    @contextlib.contextmanager
    def _autocast(self):
        """BF16 autocast on CUDA; transparent no-op on CPU/MPS."""
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=False,
        ):
            yield

    def _describe_training_device(self) -> str:
        device = self.device
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            return f"{device} ({torch.cuda.get_device_name(index)})"
        return str(device)

    @staticmethod
    def _sample_val_keep_ratio(
        *,
        val_keep_seed: int,
        epoch: int,
        batch_idx: int,
        val_keep_min: float,
        val_keep_max: float,
    ) -> float:
        if val_keep_max <= val_keep_min:
            return float(val_keep_min)
        seed_value = (
            (int(val_keep_seed) * 1_000_003)
            + (int(epoch) * 1_009)
            + int(batch_idx)
        ) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(seed_value)
        return float(rng.uniform(float(val_keep_min), float(val_keep_max)))

    @staticmethod
    def _truncate_validation_batch(batch: dict, keep_ratio: float) -> dict:
        if keep_ratio >= 1.0:
            return batch
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_length = int(past_target.shape[-1])
        keep_len = max(1, int(float(keep_ratio) * past_length))
        pad_len = max(0, past_length - keep_len)
        if pad_len == 0:
            return batch

        truncated = dict(batch)
        truncated_target = past_target.clone()
        truncated_observed = past_observed_values.clone()
        truncated_target[..., :pad_len] = 0.0
        truncated_observed[..., :pad_len] = 0.0
        truncated["past_target"] = truncated_target
        truncated["past_observed_values"] = truncated_observed
        return truncated

    def _extract_model_state_dict(self, ckpt: dict) -> dict:
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            if any(k.startswith("model.") for k in state_dict):
                return {
                    k[len("model.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
            return state_dict

        raise ValueError("Checkpoint does not contain model weights.")

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        if self.time_feat:
            return Chain(
                [
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=time_features_from_frequency_str("S"),
                        pred_length=self.prediction_length,
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=DummyValueImputation(0.0),
                ),
            ]
        )

    def _model_kwargs(self) -> dict:
        """Collect model-construction args reused for init and checkpoint metadata."""
        return {
            "input_size": self.input_size,
            "context_length": self.context_length,
            "max_context_length": self.max_context_length,
            "lags_seq": self.lags_seq,
            "ffn_type": self.ffn_type,
            "n_layer": self.n_layer,
            "scaling": self.scaling,
            "distr_output": self.distr_output,
            "num_parallel_samples": self.num_parallel_samples,
            "rope_scaling": self.rope_scaling,
            "time_feat": self.time_feat,
            "dropout": self.dropout,
            "taildart_hidden_size": self.taildart_hidden_size,
            "taildart_intermediate_size": self.taildart_intermediate_size,
            "taildart_num_heads": self.taildart_num_heads,
            "taildart_num_kv_heads": self.taildart_num_kv_heads,
            "taildart_head_dim": self.taildart_head_dim,
            "taildart_rms_norm_eps": self.taildart_rms_norm_eps,
            "moe_num_experts": self.moe_num_experts,
            "moe_top_k": self.moe_top_k,
            "moe_aux_loss_weight": self.moe_aux_loss_weight,
            "moe_router_jitter_noise": self.moe_router_jitter_noise,
        }

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[_WarmupDecayScheduler]:
        if self.warmup_name == "none" and self.decay_name == "none":
            return None
        warmup_steps = (
            int(self.warmup_epochs) * int(self.num_batches_per_epoch)
            if self.warmup_name == "linear"
            else 0
        )
        return _WarmupDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            decay_name=self.decay_name,
            decay_args=self.decay_args,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def _step_scheduler(
        self,
        scheduler: Optional[_WarmupDecayScheduler],
    ) -> None:
        if scheduler is None:
            return
        scheduler.step()

    def create_module(self, use_kv_cache: bool = False) -> TailDartModule:
        module = TailDartModule(
            loss=self.loss,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            model_kwargs=self._model_kwargs(),
            aug_prob=self.aug_prob,
            freq_mask_rate=self.freq_mask_rate,
            freq_mixing_rate=self.freq_mixing_rate,
            jitter_prob=self.jitter_prob,
            jitter_sigma=self.jitter_sigma,
            scaling_prob=self.scaling_prob,
            scaling_sigma=self.scaling_sigma,
            rotation_prob=self.rotation_prob,
            permutation_prob=self.permutation_prob,
            permutation_max_segments=self.permutation_max_segments,
            permutation_seg_mode=self.permutation_seg_mode,
            magnitude_warp_prob=self.magnitude_warp_prob,
            magnitude_warp_sigma=self.magnitude_warp_sigma,
            magnitude_warp_knot=self.magnitude_warp_knot,
            time_warp_prob=self.time_warp_prob,
            time_warp_sigma=self.time_warp_sigma,
            time_warp_knot=self.time_warp_knot,
            window_slice_prob=self.window_slice_prob,
            window_slice_reduce_ratio=self.window_slice_reduce_ratio,
            window_warp_prob=self.window_warp_prob,
            window_warp_window_ratio=self.window_warp_window_ratio,
            window_warp_scales=self.window_warp_scales,
            use_kv_cache=use_kv_cache,
            nonnegative_pred_samples=self.nonnegative_pred_samples,
            use_single_pass_sampling=self.use_single_pass_sampling,
        )

        if self.ckpt_path is not None:
            print("#", self.ckpt_path)
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            model_state_dict = self._extract_model_state_dict(ckpt)
            module.model.load_state_dict(model_state_dict, strict=True)

        return module

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length + max(self.lags_seq),
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
            if self.time_feat
            else [FieldName.OBSERVED_VALUES],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        weights_key = (
            tuple(sorted(self.train_dataset_weights.items()))
            if self.train_dataset_weights
            else None
        )
        loader_key = (id(data), int(self.prediction_length), bool(self.time_feat), weights_key, float(self.pad_augment_prob))
        if (
            self._cached_training_loader is not None
            and self._cached_training_loader_key == loader_key
        ):
            return self._cached_training_loader

        train_dataset = _PreloadedTrainingDataset(
            data=data,
            transformation=self.create_transformation(),
            past_length=self.context_length + max(self.lags_seq),
            prediction_length=self.prediction_length,
            time_feat=self.time_feat,
            dataset_weights=self.train_dataset_weights,
            pad_augment_prob=self.pad_augment_prob,
            allow_short_series_fallback=self.allow_short_series_fallback,
        )
        num_samples = self.batch_size * self.num_batches_per_epoch
        if train_dataset.sample_weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(train_dataset.sample_weights, dtype=torch.float64),
                num_samples=num_samples,
                replacement=True,
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                train_dataset,
                replacement=True,
                num_samples=num_samples,
            )

        num_workers = self.train_num_workers
        loader_kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": self.train_pin_memory,
            "collate_fn": _gluonts_collate_fn,
            "worker_init_fn": _worker_init_fn,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.train_prefetch_factor
            loader_kwargs["persistent_workers"] = True
            try:
                loader_kwargs["multiprocessing_context"] = torch.multiprocessing.get_context(
                    "fork"
                )
            except ValueError:
                # Platforms without fork (for example Windows) fall back to default.
                pass

        loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
        self._cached_training_loader = loader
        self._cached_training_loader_key = loader_key
        return loader

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        transformed_data = self.create_transformation().apply(data, is_train=True)
        instances = self._create_instance_splitter("validation").apply(
            transformed_data, is_train=True
        )
        if self.time_feat:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                field_names=TRAINING_INPUT_NAMES + ["past_time_feat", "future_time_feat"],
                output_type=torch.tensor,
            )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TailDartModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter("test")
        if self.time_feat:
            input_names = PREDICTION_INPUT_NAMES + ["past_time_feat", "future_time_feat"]
        else:
            input_names = PREDICTION_INPUT_NAMES

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=self.device,
        )

    def _move_batch_to_device(self, batch: dict, device: torch.device) -> dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _save_checkpoint(
        self,
        path: str,
        module: TailDartModule,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_WarmupDecayScheduler],
        epoch: int,
        best_val_loss: float,
        global_step: Optional[int] = None,
        step_in_epoch: Optional[int] = None,
        validation_event: int = 0,
        validation_events_without_improvement: int = 0,
        best_interval: Optional[int] = None,
        best_global_step: Optional[int] = None,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": module.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "validation_event": validation_event,
            "validation_events_without_improvement": validation_events_without_improvement,
            "best_interval": best_interval,
            "best_global_step": best_global_step,
            "num_batches_per_epoch": self.num_batches_per_epoch,
            "validate_at_steps": self.validate_at_steps,
            "max_training_steps": self.max_training_steps,
            "metadata": {
                "model_kwargs": self._model_kwargs(),
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
            },
        }
        torch.save(ckpt, path)

    def _sanitize_checkpoint_token(self, value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
        return token or "unknown"

    def _resolve_checkpoint_run_date(self) -> str:
        """Lazily freeze one run date so all named checkpoints share the same prefix."""
        if self.checkpoint_run_date is None:
            self.checkpoint_run_date = datetime.now().strftime("%Y%m%d")
        return self.checkpoint_run_date

    def _named_checkpoint_filename(self, epoch: int, val_loss: float, run_date: str) -> str:
        model_name = self._sanitize_checkpoint_token(self.checkpoint_model_name)
        version = self._sanitize_checkpoint_token(self.checkpoint_version)
        return f"{model_name}-{version}_{run_date}_{int(epoch):04d}_{float(val_loss):.4f}.ckpt"

    def _save_named_checkpoint(
        self,
        checkpoint_dir: str,
        source_ckpt_path: str,
        epoch: int,
        val_loss: float,
        run_date: str,
    ) -> str:
        filename = self._named_checkpoint_filename(epoch=epoch, val_loss=val_loss, run_date=run_date)
        named_path = os.path.join(checkpoint_dir, filename)
        shutil.copy2(source_ckpt_path, named_path)
        return named_path

    def _evaluate_validation(
        self,
        module: TailDartModule,
        validation_data: Dataset,
        *,
        validation_seed: int,
    ) -> dict[str, Optional[float]]:
        module.eval()
        aggregates = {
            "standard": {
                "nll_sum": 0.0, "observed": 0.0, "aux_sum": 0.0, "samples": 0, "routing": None,
            },
            "mixed": {
                "nll_sum": 0.0, "observed": 0.0, "aux_sum": 0.0, "samples": 0, "routing": None,
            },
        }

        def accumulate(
            name: str,
            batch: dict,
            nll: torch.Tensor,
            aux: torch.Tensor,
            routing_statistics: Optional[
                list[tuple[torch.Tensor, torch.Tensor, int]]
            ],
        ) -> None:
            context_observed = batch["past_observed_values"]
            if self.context_length > 1:
                context_observed = context_observed[..., -(self.context_length - 1) :]
                context_count = float(context_observed.sum().detach().cpu().item())
            else:
                context_count = 0.0
            future_count = float(
                batch["future_observed_values"].sum().detach().cpu().item()
            )
            observed_count = context_count + future_count
            batch_size = int(batch["past_target"].shape[0])
            aggregate = aggregates[name]
            aggregate["nll_sum"] += float(nll.detach().cpu().item()) * observed_count
            aggregate["observed"] += observed_count
            aggregate["aux_sum"] += float(aux.detach().cpu().item()) * batch_size
            aggregate["samples"] += batch_size
            if routing_statistics:
                if aggregate["routing"] is None:
                    aggregate["routing"] = [
                        {
                            "prob_sum": torch.zeros_like(
                                prob_sum, device="cpu", dtype=torch.float64
                            ),
                            "assignment_sum": torch.zeros_like(
                                assignment_sum, device="cpu", dtype=torch.float64
                            ),
                            "tokens": 0,
                        }
                        for prob_sum, assignment_sum, _ in routing_statistics
                    ]
                if len(aggregate["routing"]) != len(routing_statistics):
                    raise RuntimeError("MoE routing layer count changed during validation.")
                for layer, (prob_sum, assignment_sum, token_count) in zip(
                    aggregate["routing"], routing_statistics
                ):
                    layer["prob_sum"] += prob_sum.detach().to(
                        device="cpu", dtype=torch.float64
                    )
                    layer["assignment_sum"] += assignment_sum.detach().to(
                        device="cpu", dtype=torch.float64
                    )
                    layer["tokens"] += int(token_count)

        def latest_routing_statistics() -> Optional[
            list[tuple[torch.Tensor, torch.Tensor, int]]
        ]:
            getter = getattr(module, "validation_routing_statistics", None)
            return getter() if getter is not None else None

        with torch.no_grad():
            for idx, batch in enumerate(self.create_validation_data_loader(validation_data)):
                if self.limit_val_batches is not None and idx >= int(self.limit_val_batches):
                    break
                batch = self._move_batch_to_device(batch, self.device)
                with self._autocast():
                    _, nll_standard, aux_standard = module.validation_loss_components(batch)
                accumulate(
                    "standard",
                    batch,
                    nll_standard,
                    aux_standard,
                    latest_routing_statistics(),
                )

                if self.use_mixed_keep_validation:
                    keep_ratio = self._sample_val_keep_ratio(
                        val_keep_seed=self.val_keep_seed,
                        epoch=validation_seed,
                        batch_idx=idx,
                        val_keep_min=self.val_keep_min,
                        val_keep_max=self.val_keep_max,
                    )
                    truncated = self._truncate_validation_batch(batch=batch, keep_ratio=keep_ratio)
                    with self._autocast():
                        _, nll_mixed, aux_mixed = module.validation_loss_components(truncated)
                    accumulate(
                        "mixed",
                        truncated,
                        nll_mixed,
                        aux_mixed,
                        latest_routing_statistics(),
                    )

        def finish(name: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
            aggregate = aggregates[name]
            if aggregate["samples"] == 0:
                return None, None, None
            nll = aggregate["nll_sum"] / max(aggregate["observed"], 1.0)
            if aggregate["routing"]:
                layer_aux = []
                for layer in aggregate["routing"]:
                    tokens = max(int(layer["tokens"]), 1)
                    mean_prob = layer["prob_sum"] / tokens
                    mean_assignment = layer["assignment_sum"] / tokens
                    num_experts = int(mean_prob.numel())
                    layer_aux.append(
                        num_experts * float(torch.sum(mean_prob * mean_assignment).item())
                    )
                aux = sum(layer_aux) / len(layer_aux)
            else:
                aux = aggregate["aux_sum"] / aggregate["samples"]
            total = nll + self.moe_aux_loss_weight * aux
            return total, nll, aux

        standard, nll_standard, aux_standard = finish("standard")
        mixed, nll_mixed, aux_mixed = finish("mixed")
        selected, nll_selected, aux_selected = (
            (mixed, nll_mixed, aux_mixed)
            if self.use_mixed_keep_validation
            else (standard, nll_standard, aux_standard)
        )
        return {
            "standard": standard,
            "selected": selected,
            "mixed": mixed,
            "nll_standard": nll_standard,
            "nll_selected": nll_selected,
            "nll_mixed": nll_mixed,
            "aux_standard": aux_standard,
            "aux_selected": aux_selected,
            "aux_mixed": aux_mixed,
        }

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Dataset,
        ckpt_path: Optional[str] = None,
        reset_training_state: bool = False,
    ):
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        module = self.create_module(use_kv_cache=False)
        module.to(self.device)
        #print("model param device:", next(module.model.parameters()).device)
        #print("estimator device:", self.device)

        optimizer = torch.optim.AdamW(
            module.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = self._build_scheduler(optimizer)

        start_epoch = 0
        resume_step_in_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        best_standard_val_loss = None
        best_mixed_val_loss = None
        best_interval = None
        best_global_step = None
        loaded_ckpt = None
        if self.ckpt_path is not None and os.path.isfile(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            loaded_ckpt = ckpt
            if not reset_training_state and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if (
                scheduler is not None
                and not reset_training_state
                and ckpt.get("scheduler_state_dict") is not None
            ):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if not reset_training_state:
                best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
                saved_global_step = ckpt.get("global_step")
                if saved_global_step is not None:
                    saved_batches_per_epoch = ckpt.get("num_batches_per_epoch")
                    if (
                        saved_batches_per_epoch is not None
                        and int(saved_batches_per_epoch) != self.num_batches_per_epoch
                    ):
                        raise ValueError(
                            "Cannot resume a step-aware checkpoint with a different "
                            "num_batches_per_epoch."
                        )
                    global_step = int(saved_global_step)
                    start_epoch, resume_step_in_epoch = divmod(
                        global_step, self.num_batches_per_epoch
                    )
                else:
                    ckpt_epoch = int(ckpt.get("epoch", -1))
                    start_epoch = ckpt_epoch + 1
                    global_step = start_epoch * self.num_batches_per_epoch
                if global_step > 0:
                    best_global_step = global_step

        default_root_dir = self.trainer_kwargs.get("default_root_dir", ".")
        checkpoint_dir = os.path.join(default_root_dir, "checkpoints")
        if self.save_canonical_checkpoints or self.save_named_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
        best_ckpt_path = os.path.join(checkpoint_dir, "best.ckpt")
        last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")

        max_epochs = int(self.trainer_kwargs.get("max_epochs", 100))
        patience = self.early_stopping_patience
        requested_validation_steps = (
            None if self.validate_at_steps is None else set(self.validate_at_steps)
        )
        validation_events_without_improvement = 0
        validation_event = 0
        if loaded_ckpt is not None and not reset_training_state:
            validation_events_without_improvement = int(
                loaded_ckpt.get("validation_events_without_improvement", 0)
            )
            if "validation_event" in loaded_ckpt:
                validation_event = int(loaded_ckpt["validation_event"])
            elif requested_validation_steps is None:
                validation_event = global_step // self.num_batches_per_epoch
            else:
                validation_event = sum(
                    1 for step in requested_validation_steps if step <= global_step
                )
            if loaded_ckpt.get("best_interval") is not None:
                best_interval = int(loaded_ckpt["best_interval"])
            elif best_val_loss < float("inf") and validation_event > 0:
                best_interval = validation_event
            if loaded_ckpt.get("best_global_step") is not None:
                best_global_step = int(loaded_ckpt["best_global_step"])
            elif best_val_loss < float("inf") and global_step > 0:
                best_global_step = global_step
        run_date = self._resolve_checkpoint_run_date()
        self.last_named_ckpt_path = None
        self.best_named_ckpt_path = None
        resume_status = self.ckpt_path if self.ckpt_path is not None else "from scratch"
        tqdm.write(
            f"{ANSI_LABEL}Training device:{ANSI_RESET} {self._describe_training_device()} | "
            f"{ANSI_LABEL}init:{ANSI_RESET} {resume_status}"
        )

        stop_training = False
        current_epoch = start_epoch
        for epoch in range(start_epoch, max_epochs):
            current_epoch = epoch
            module.train()
            train_loader = self.create_training_data_loader(training_data)

            processed_batches = 0
            interval_start_time = time.perf_counter()
            running_loss = 0.0
            running_nll_loss = 0.0
            running_moe_aux_loss = 0.0
            windows_processed = 0
            is_moe_model = module.model.ffn_type == "moe"
            progress_bar = tqdm(
                train_loader,
                total=self.num_batches_per_epoch,
                desc=f"Train interval {epoch + 1}/{max_epochs}",
                dynamic_ncols=True,
                leave=False,
            )
            #step_start_time = time.perf_counter()
            for interval_step, batch in enumerate(progress_bar, start=1):
                if epoch == start_epoch and interval_step <= resume_step_in_epoch:
                    continue
                if self.max_training_steps is not None and global_step >= self.max_training_steps:
                    stop_training = True
                    break
                #if interval_step == 1 and epoch == start_epoch: print("batch past_target before move:", batch["past_target"].device)
                #batch_fetch_done = time.perf_counter()
                #batch_fetch_time = batch_fetch_done - step_start_time
                batch_windows = int(batch["past_target"].shape[0])
                #move_start = time.perf_counter()
                batch = self._move_batch_to_device(batch, self.device)
                '''
                move_done = time.perf_counter()
                move_time = move_done - move_start
                if interval_step == 1 and epoch == start_epoch: print("batch past_target after move:", batch["past_target"].device)
                if self.device.type == "cuda": torch.cuda.synchronize()
                compute_start = time.perf_counter()
                '''
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, nll_loss, moe_aux_loss = module.training_loss_components(batch)
                loss.backward()
                optimizer.step()
                self._step_scheduler(scheduler)

                '''
                if self.device.type == "cuda": torch.cuda.synchronize()
                compute_done = time.perf_counter()
                compute_time = compute_done - compute_start
                step_start_time = time.perf_counter()
                if interval_step % 20 == 0:
                    print(
                        f"step {interval_step}: "
                        f"fetch={batch_fetch_time:.3f}s "
                        f"to_device={move_time:.3f}s "
                        f"compute={compute_time:.3f}s"
                    )
                '''
                global_step += 1
                processed_batches += 1
                windows_processed += batch_windows

                loss_value = float(loss.detach().cpu().item())
                nll_value = float(nll_loss.detach().cpu().item())
                moe_aux_value = float(moe_aux_loss.detach().cpu().item())
                running_loss += loss_value
                running_nll_loss += nll_value
                running_moe_aux_loss += moe_aux_value
                avg_train_loss = running_loss / processed_batches
                avg_train_nll_loss = running_nll_loss / processed_batches
                avg_train_moe_aux_loss = running_moe_aux_loss / processed_batches
                elapsed = max(time.perf_counter() - interval_start_time, 1e-12)
                windows_per_second = windows_processed / elapsed
                learning_rate = float(optimizer.param_groups[0]["lr"])
                postfix = {
                    "step": f"{global_step} ({interval_step}/{self.num_batches_per_epoch})",
                    "loss": f"{loss_value:.4f}",
                    "avg_loss": f"{avg_train_loss:.4f}",
                    "lr": f"{learning_rate:.2e}",
                    "win_s": f"{windows_per_second:.1f}",
                }
                if is_moe_model:
                    postfix["nll"] = f"{nll_value:.4f}"
                    postfix["aux"] = f"{moe_aux_value:.4f}"
                    postfix["avg_nll"] = f"{avg_train_nll_loss:.4f}"
                    postfix["avg_aux"] = f"{avg_train_moe_aux_loss:.4f}"
                progress_bar.set_postfix(**postfix)
                should_validate = (
                    global_step in requested_validation_steps
                    if requested_validation_steps is not None
                    else interval_step == self.num_batches_per_epoch
                )
                if should_validate:
                    validation_event += 1
                    metrics = self._evaluate_validation(
                        module,
                        validation_data,
                        validation_seed=(
                            global_step if requested_validation_steps is not None else epoch
                        ),
                    )
                    selected = metrics["selected"]
                    if selected is None:
                        raise RuntimeError("Validation produced no batches.")
                    improved = selected < best_val_loss
                    if improved:
                        best_val_loss = selected
                        best_standard_val_loss = metrics["standard"]
                        best_mixed_val_loss = metrics["mixed"]
                        best_interval = validation_event
                        best_global_step = global_step
                        validation_events_without_improvement = 0
                    else:
                        validation_events_without_improvement += 1

                    if self.save_canonical_checkpoints:
                        checkpoint_kwargs = dict(
                            module=module,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            global_step=global_step,
                            step_in_epoch=interval_step,
                            validation_event=validation_event,
                            validation_events_without_improvement=(
                                validation_events_without_improvement
                            ),
                            best_interval=best_interval,
                            best_global_step=best_global_step,
                        )
                        save_last_checkpoint = (
                            requested_validation_steps is not None
                            or (epoch + 1) % 10 == 0
                        )
                        if save_last_checkpoint:
                            self._save_checkpoint(last_ckpt_path, **checkpoint_kwargs)
                        if improved:
                            self._save_checkpoint(best_ckpt_path, **checkpoint_kwargs)
                        if self.save_named_checkpoints:
                            if save_last_checkpoint:
                                self.last_named_ckpt_path = self._save_named_checkpoint(
                                    checkpoint_dir=checkpoint_dir,
                                    source_ckpt_path=last_ckpt_path,
                                    epoch=epoch,
                                    val_loss=selected,
                                    run_date=run_date,
                                )
                            if improved:
                                self.best_named_ckpt_path = self._save_named_checkpoint(
                                    checkpoint_dir=checkpoint_dir,
                                    source_ckpt_path=best_ckpt_path,
                                    epoch=epoch,
                                    val_loss=selected,
                                    run_date=run_date,
                                )

                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    mean_train_loss = running_loss / processed_batches
                    interval_label = (
                        f"Interval {validation_event}"
                        if requested_validation_steps is not None
                        else f"Interval {epoch + 1}/{max_epochs}"
                    )
                    if self.use_mixed_keep_validation:
                        summary = (
                            f"{interval_label} | step={global_step} | "
                            f"train_loss={mean_train_loss:.6f} | "
                            f"val_loss_keep1={metrics['standard']:.6f} | "
                            f"val_loss_mixed={metrics['mixed']:.6f} | "
                            f"best_val_loss={best_val_loss:.6f} | lr={learning_rate:.2e} | "
                            f"patience={validation_events_without_improvement}/{patience}"
                        )
                    else:
                        summary = (
                            f"{interval_label} | step={global_step} | "
                            f"train_loss={mean_train_loss:.6f} | "
                            f"val_loss={metrics['standard']:.6f} | "
                            f"best_val_loss={best_val_loss:.6f} | lr={learning_rate:.2e} | "
                            f"patience={validation_events_without_improvement}/{patience}"
                        )
                    if is_moe_model:
                        summary += (
                            f" | train_loss_nll={running_nll_loss / processed_batches:.6f}"
                            f" | train_loss_moe_aux={running_moe_aux_loss / processed_batches:.6f}"
                            f" | val_loss_nll={metrics['nll_selected']:.6f}"
                            f" | val_loss_moe_aux={metrics['aux_selected']:.6f}"
                        )
                    tqdm.write(summary)
                    module.train()

                    if (
                        patience >= 0
                        and validation_events_without_improvement >= patience
                    ):
                        tqdm.write(
                            f"Early stopping triggered after {patience} validation events "
                            "without improvement."
                        )
                        stop_training = True
                        break

                if self.max_training_steps is not None and global_step >= self.max_training_steps:
                    stop_training = True
                    break
            progress_bar.close()
            resume_step_in_epoch = 0
            if stop_training:
                break

        if self.save_canonical_checkpoints and global_step > 0:
            self._save_checkpoint(
                last_ckpt_path,
                module,
                optimizer,
                scheduler,
                current_epoch,
                best_val_loss,
                global_step=global_step,
                step_in_epoch=global_step % self.num_batches_per_epoch,
                validation_event=validation_event,
                validation_events_without_improvement=validation_events_without_improvement,
                best_interval=best_interval,
                best_global_step=best_global_step,
            )

        if os.path.isfile(best_ckpt_path):
            self.ckpt_path = best_ckpt_path
        elif os.path.isfile(last_ckpt_path):
            self.ckpt_path = last_ckpt_path
        else:
            self.ckpt_path = None

        checkpoint_callback = SimpleNamespace(
            best_model_path=self.ckpt_path,
            best_named_model_path=self.best_named_ckpt_path,
            last_named_model_path=self.last_named_ckpt_path,
            best_val_loss=best_val_loss,
            best_selected_metric=(
                "mixed_keep_val_loss" if self.use_mixed_keep_validation else "val_loss"
            ),
            best_standard_val_loss=best_standard_val_loss,
            best_mixed_val_loss=best_mixed_val_loss,
            best_interval=best_interval,
            best_global_step=best_global_step,
            val_keep_min=self.val_keep_min,
            val_keep_max=self.val_keep_max,
            val_keep_seed=self.val_keep_seed,
            use_mixed_keep_validation=self.use_mixed_keep_validation,
        )
        trainer = SimpleNamespace(checkpoint_callback=checkpoint_callback)
        return SimpleNamespace(model=module, trainer=trainer)
