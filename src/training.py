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

"""Shared training helpers for TailDart configuration and estimator creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHECKPOINT_MODEL_NAME = "lag-gemma" #"tail_dart" #
CHECKPOINT_VERSION = "v1"
ESTIMATOR_LAGS_SEQ = ("Q", "M", "W", "D", "H", "T", "S")
REQUIRED_CONFIG_KEYS = (
    "n_layer",
    "gemma_hidden_size",
    "gemma_num_heads",
    "gemma_num_kv_heads",
    "gemma_head_dim",
    "gemma_intermediate_size",
    "warmup",
    "decay",
)
MOE_CONFIG_KEYS = (
    "moe_num_experts",
    "moe_top_k",
    "moe_aux_loss_weight",
    "moe_router_jitter_noise",
)
SUPPORTED_SCALING_MODES = {"mean", "std", "robust", "none", "nop"}


def _validate_model_config(model_cfg: dict[str, Any], *, require_gqa: bool) -> dict[str, Any]:
    validated = dict(model_cfg)
    validated["ffn_type"] = str(validated.get("ffn_type", "dense")).strip().lower()
    validated["scaling"] = str(validated.get("scaling", "mean")).strip().lower()
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in validated]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    ffn_type = validated["ffn_type"]
    gemma_num_heads = validated.get("gemma_num_heads")
    gemma_num_kv_heads = validated.get("gemma_num_kv_heads")
    gemma_hidden_size = validated.get("gemma_hidden_size")
    gemma_head_dim = validated.get("gemma_head_dim")
    warmup = validated.get("warmup")
    decay = validated.get("decay")

    if ffn_type not in {"dense", "moe"}:
        raise ValueError("`ffn_type` must be one of: dense, moe.")

    if gemma_num_heads is None:
        raise ValueError("`gemma_num_heads` must be specified explicitly.")
    if gemma_num_kv_heads is None:
        raise ValueError("`gemma_num_kv_heads` must be specified explicitly.")

    if gemma_num_heads is not None and gemma_num_kv_heads is not None:
        if int(gemma_num_heads) % int(gemma_num_kv_heads) != 0:
            raise ValueError("`gemma_num_heads` must be divisible by `gemma_num_kv_heads`.")
    if require_gqa and gemma_num_kv_heads == gemma_num_heads:
        raise ValueError(
            "`gemma_num_kv_heads` must differ from `gemma_num_heads` to enforce GQA."
        )
    if gemma_hidden_size is not None and gemma_num_heads is not None and gemma_head_dim is not None:
        if int(gemma_hidden_size) != int(gemma_num_heads) * int(gemma_head_dim):
            raise ValueError(
                "`gemma_hidden_size` must equal `gemma_num_heads * gemma_head_dim`."
            )
    if not isinstance(warmup, dict):
        raise ValueError("`warmup` must be an object/dict.")
    warmup_name = str(warmup.get("name", "none")).strip().lower()
    warmup_epochs = warmup.get("epochs", 0)
    if warmup_name not in {"none", "linear"}:
        raise ValueError("`warmup.name` must be one of: none, linear.")
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("`warmup.epochs` must be an integer >= 0.")

    if not isinstance(decay, dict):
        raise ValueError("`decay` must be an object/dict.")
    decay_name = str(decay.get("name", "none")).strip().lower()
    decay_args = decay.get("args")
    if decay_name not in {"none", "cosine", "cosine_restarts"}:
        raise ValueError("`decay.name` must be one of: none, cosine, cosine_restarts.")
    if not isinstance(decay_args, dict):
        raise ValueError("`decay.args` must be an object/dict.")
    if validated["scaling"] not in SUPPORTED_SCALING_MODES:
        raise ValueError(
            "`scaling` must be one of: mean, std, robust, none, nop."
        )

    if ffn_type == "moe":
        missing_moe = [key for key in MOE_CONFIG_KEYS if key not in validated]
        if missing_moe:
            raise ValueError(f"Model config missing required keys: {missing_moe}")
        moe_num_experts = validated.get("moe_num_experts")
        moe_top_k = validated.get("moe_top_k")
        if int(moe_num_experts) < 2:
            raise ValueError("`moe_num_experts` must be >= 2.")
        if int(moe_top_k) != 1:
            raise ValueError("`moe_top_k` must be exactly 1 for top-1 sparse routing.")
        
    return validated


def load_model_config(config_path: Path, model_size: str | None = "base") -> dict[str, Any]:
    """Load model hyperparameters from a JSON config file."""
    with config_path.open(encoding="utf-8") as handle:
        raw_config = json.load(handle)

    shared = raw_config.get("shared")
    model_sizes = raw_config.get("model_sizes")
    if not isinstance(shared, dict):
        raise ValueError(
            "Model config must use the hierarchical schema and define `shared` as an object."
        )
    if not isinstance(model_sizes, dict) or not model_sizes:
        raise ValueError(
            "Model config must use the hierarchical schema and define a non-empty `model_sizes` object."
        )

    if model_size not in model_sizes:
        available_size_names = model_sizes.keys()
        raise ValueError(
            f"Unknown model size `{model_size}`. Available sizes: {available_size_names}"
        )

    resolved = dict(shared)
    size_config = model_sizes[model_size]
    if not isinstance(size_config, dict):
        raise ValueError(f"Model size `{model_size}` must map to an object.")
    resolved.update(size_config)
    resolved["model_size"] = model_size
    return _validate_model_config(resolved, require_gqa=True)


def build_estimator_kwargs(
    model_cfg: dict[str, Any],
    *,
    prediction_length: int,
    context_length: int,
    batch_size: int,
    device: Any,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_batches_per_epoch: int | None = None,
    num_parallel_samples: int | None = None,
    use_single_pass_sampling: bool | None = None,
    trainer_kwargs: dict[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    validate_at_steps: list[int] | None = None,
    max_training_steps: int | None = None,
    ckpt_path: str | None = None,
    checkpoint_model_name: str | None = None,
    checkpoint_version: str | None = None,
    checkpoint_run_date: str | None = None,
    save_canonical_checkpoints: bool | None = None,
    save_named_checkpoints: bool | None = None,
    train_num_workers: int | None = None,
    train_prefetch_factor: int | None = None,
    train_pin_memory: bool | None = None,
    max_lag: int | None = None,
    distr_output: str | None = None,
    pad_augment_prob: float | None = None,
    val_keep_min: float | None = None,
    val_keep_max: float | None = None,
    val_keep_seed: int | None = None,
) -> dict[str, Any]:
    """Translate config JSON plus runtime options into estimator kwargs."""
    kwargs: dict[str, Any] = {
        "prediction_length": prediction_length,
        "context_length": context_length,
        "input_size": 1,
        "ffn_type": str(model_cfg.get("ffn_type", "dense")).strip().lower(),
        "n_layer": int(model_cfg.get("n_layer", 2)),
        "gemma_hidden_size": int(model_cfg.get("gemma_hidden_size", 64)),
        "gemma_intermediate_size": int(model_cfg.get("gemma_intermediate_size", 128)),
        "gemma_num_heads": int(model_cfg.get("gemma_num_heads", 4)),
        "gemma_num_kv_heads": int(model_cfg.get("gemma_num_kv_heads", 2)),
        "gemma_head_dim": int(model_cfg.get("gemma_head_dim", 16)),
        "gemma_rms_norm_eps": float(model_cfg.get("gemma_rms_norm_eps", 1e-6)),
        "moe_num_experts": int(model_cfg.get("moe_num_experts", 3)),
        "moe_top_k": int(model_cfg.get("moe_top_k", 1)),
        "moe_aux_loss_weight": float(model_cfg.get("moe_aux_loss_weight", 0.01)),
        "moe_router_jitter_noise": float(model_cfg.get("moe_router_jitter_noise", 0.0)),
        "warmup": dict(model_cfg.get("warmup", {"name": "none", "epochs": 0})),
        "decay": dict(model_cfg.get("decay", {"name": "none", "args": {}})),
        "scaling": str(model_cfg.get("scaling", "mean")).strip().lower(),
        "time_feat": bool(model_cfg.get("time_feat", False)),
        "aug_prob": float(model_cfg.get("aug_prob", 0.1)),
        "freq_mask_rate": float(model_cfg.get("freq_mask_rate", 0.1)),
        "freq_mixing_rate": float(model_cfg.get("freq_mixing_rate", 0.1)),
        "distr_output": str(
            distr_output
            if distr_output is not None
            else model_cfg.get("distr_output", "studentT")
        ),
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "lags_seq": list(ESTIMATOR_LAGS_SEQ),
        "batch_size": batch_size,
        "device": device,
    }
    if num_batches_per_epoch is not None:
        kwargs["num_batches_per_epoch"] = int(num_batches_per_epoch)
    if num_parallel_samples is not None:
        kwargs["num_parallel_samples"] = int(num_parallel_samples)
    if use_single_pass_sampling is not None:
        kwargs["use_single_pass_sampling"] = bool(use_single_pass_sampling)
    if trainer_kwargs is not None:
        kwargs["trainer_kwargs"] = trainer_kwargs
    if early_stopping_patience is not None:
        kwargs["early_stopping_patience"] = int(early_stopping_patience)
    if validate_at_steps is not None:
        kwargs["validate_at_steps"] = list(validate_at_steps)
    if max_training_steps is not None:
        kwargs["max_training_steps"] = int(max_training_steps)
    if ckpt_path is not None:
        kwargs["ckpt_path"] = ckpt_path
    if checkpoint_model_name is not None:
        kwargs["checkpoint_model_name"] = checkpoint_model_name
    if checkpoint_version is not None:
        kwargs["checkpoint_version"] = checkpoint_version
    if checkpoint_run_date is not None:
        kwargs["checkpoint_run_date"] = checkpoint_run_date
    if save_canonical_checkpoints is not None:
        kwargs["save_canonical_checkpoints"] = save_canonical_checkpoints
    if save_named_checkpoints is not None:
        kwargs["save_named_checkpoints"] = save_named_checkpoints
    if train_num_workers is not None:
        kwargs["train_num_workers"] = int(train_num_workers)
    if train_prefetch_factor is not None:
        kwargs["train_prefetch_factor"] = int(train_prefetch_factor)
    if train_pin_memory is not None:
        kwargs["train_pin_memory"] = bool(train_pin_memory)
    if max_lag is not None:
        kwargs["max_lag"] = int(max_lag)
    if pad_augment_prob is not None:
        kwargs["pad_augment_prob"] = float(pad_augment_prob)
    if val_keep_min is not None:
        kwargs["val_keep_min"] = float(val_keep_min)
    if val_keep_max is not None:
        kwargs["val_keep_max"] = float(val_keep_max)
    if val_keep_seed is not None:
        kwargs["val_keep_seed"] = int(val_keep_seed)
    return kwargs


def build_estimator(
    model_cfg: dict[str, Any],
    *,
    run_dir: Path,
    prediction_length: int,
    context_length: int,
    max_epochs: int,
    batch_size: int,
    num_batches_per_epoch: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int | None = None,
    validate_at_steps: list[int] | None = None,
    max_training_steps: int | None = None,
    save_canonical_checkpoints: bool | None = True,
    save_named_checkpoints: bool | None = True,
    train_num_workers: int | None = None,
    train_prefetch_factor: int | None = None,
    train_pin_memory: bool | None = None,
    max_lag: int | None = None,
    distr_output: str | None = None,
    pad_augment_prob: float | None = None,
    val_keep_min: float | None = None,
    val_keep_max: float | None = None,
    val_keep_seed: int | None = None,
    use_single_pass_sampling: bool | None = None,
    device: str,
):
    """Construct a training-ready LagGemmaEstimator from shared config."""
    import torch

    #from lag_gemma.gluon.estimator import LagGemmaEstimator
    from src.gluon.estimator import LagGemmaEstimator

    return LagGemmaEstimator(
        **build_estimator_kwargs(
            model_cfg,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_batches_per_epoch=num_batches_per_epoch,
            trainer_kwargs={"max_epochs": int(max_epochs), "default_root_dir": str(run_dir)},
            early_stopping_patience=(
                int(model_cfg.get("early_stopping_patience", 50))
                if early_stopping_patience is None
                else early_stopping_patience
            ),
            validate_at_steps=validate_at_steps,
            max_training_steps=max_training_steps,
            device=torch.device(device),
            ckpt_path=None,
            checkpoint_model_name=CHECKPOINT_MODEL_NAME,
            checkpoint_version=CHECKPOINT_VERSION,
            checkpoint_run_date=None,
            save_canonical_checkpoints=save_canonical_checkpoints,
            save_named_checkpoints=save_named_checkpoints,
            train_num_workers=train_num_workers,
            train_prefetch_factor=train_prefetch_factor,
            train_pin_memory=train_pin_memory,
            max_lag=max_lag,
            distr_output=distr_output,
            pad_augment_prob=pad_augment_prob,
            val_keep_min=val_keep_min,
            val_keep_max=val_keep_max,
            val_keep_seed=val_keep_seed,
            use_single_pass_sampling=use_single_pass_sampling,
        )
    )
