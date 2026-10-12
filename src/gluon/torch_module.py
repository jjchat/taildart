# Copyright 2024 Arjun Ashok
# Modifications Copyright 2026 Chiang Qin Zhi
#
# Modified as part of Lag-Gemma project (replacing LLaMA with Gemma backbone)
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

import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import repeat_along_dim, take_last

from data.augmentations.freq_mask import freq_mask
from data.augmentations.freq_mix import freq_mix
from data.augmentations.augmentations import (
    ApplyAugmentations,
    Jitter,
    MagnitudeWarp,
    Permutation,
    Rotation,
    Scaling,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)
from src.model.gemma_module import LagGemmaModel


class LagGemmaModule(nn.Module):
    """Pure-PyTorch wrapper around LagGemmaModel for training and inference."""

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
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
        nonnegative_pred_samples: bool = False,
        use_kv_cache: bool = True,
        use_single_pass_sampling: bool = False,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_kwargs = dict(model_kwargs)
        if use_single_pass_sampling and prediction_length > 1:
            raise ValueError(
                "use_single_pass_sampling=True is only valid for prediction_length=1; "
                "multi-step forecasts would condition all samples on a shared mean trajectory."
            )

        self.model = LagGemmaModel(**self.model_kwargs)
        self.loss = loss
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
        self.nonnegative_pred_samples = nonnegative_pred_samples

        self.time_feat = bool(self.model_kwargs["time_feat"])
        self.use_kv_cache = use_kv_cache
        self.use_single_pass_sampling = bool(use_single_pass_sampling)

        self.transforms = []
        aug_probs = dict(
            Jitter=dict(prob=self.jitter_prob, sigma=self.jitter_sigma),
            Scaling=dict(prob=self.scaling_prob, sigma=self.scaling_sigma),
            Rotation=dict(prob=self.rotation_prob),
            Permutation=dict(
                prob=self.permutation_prob,
                max_segments=self.permutation_max_segments,
                seg_mode=self.permutation_seg_mode,
            ),
            MagnitudeWarp=dict(
                prob=self.magnitude_warp_prob,
                sigma=self.magnitude_warp_sigma,
                knot=self.magnitude_warp_knot,
            ),
            TimeWarp=dict(
                prob=self.time_warp_prob,
                sigma=self.time_warp_sigma,
                knot=self.time_warp_knot,
            ),
            WindowSlice=dict(
                prob=self.window_slice_prob, reduce_ratio=self.window_slice_reduce_ratio
            ),
            WindowWarp=dict(
                prob=self.window_warp_prob,
                window_ratio=self.window_warp_window_ratio,
                warp_slices=self.window_warp_scales,
            ),
        )
        for aug, params in aug_probs.items():
            if params["prob"] > 0:
                if aug == "Jitter":
                    self.transforms.append(Jitter(params["prob"], params["sigma"]))
                elif aug == "Scaling":
                    self.transforms.append(Scaling(params["prob"], params["sigma"]))
                elif aug == "Rotation":
                    self.transforms.append(Rotation(params["prob"]))
                elif aug == "Permutation":
                    self.transforms.append(
                        Permutation(
                            params["prob"], params["max_segments"], params["seg_mode"]
                        )
                    )
                elif aug == "MagnitudeWarp":
                    self.transforms.append(
                        MagnitudeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "TimeWarp":
                    self.transforms.append(
                        TimeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "WindowSlice":
                    self.transforms.append(
                        WindowSlice(params["prob"], params["reduce_ratio"])
                    )
                elif aug == "WindowWarp":
                    self.transforms.append(
                        WindowWarp(
                            params["prob"],
                            params["window_ratio"],
                            params["warp_slices"],
                        )
                    )

        self.augmentations = ApplyAugmentations(self.transforms)

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]
        past_time_feat: Optional[torch.Tensor] = kwargs.get("past_time_feat")
        future_time_feat: Optional[torch.Tensor] = kwargs.get("future_time_feat")

        future_samples = []

        if self.use_single_pass_sampling:
            for t in range(self.prediction_length):
                params, loc, scale, _ = self.model(
                    *args,
                    past_time_feat=past_time_feat if self.time_feat else None,
                    future_time_feat=future_time_feat[..., : t + 1, :] if self.time_feat else None,
                    past_target=past_target,
                    past_observed_values=past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                )

                sliced_params = [p[:, -1:] for p in params]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                greedy_prediction = distr.mean

                repeated_sliced_params = [
                    p[:, -1:].repeat_interleave(self.model.num_parallel_samples, 0)
                    for p in params
                ]
                repeated_loc = loc.repeat_interleave(self.model.num_parallel_samples, 0)
                repeated_scale = scale.repeat_interleave(self.model.num_parallel_samples, 0)
                repeated_distr = self.model.distr_output.distribution(
                    repeated_sliced_params, repeated_loc, repeated_scale
                )
                sample = repeated_distr.sample()
                if self.nonnegative_pred_samples:
                    sample = F.relu(sample)
                future_samples.append(sample)

                past_target = torch.cat((past_target, greedy_prediction), dim=1)
                past_observed_values = torch.cat(
                    (past_observed_values, torch.ones_like(greedy_prediction)), dim=1
                )
        else:
            repeated_past_target = past_target.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            repeated_past_observed_values = past_observed_values.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            if self.time_feat:
                repeated_past_time_feat = past_time_feat.repeat_interleave(
                    self.model.num_parallel_samples, 0
                )
                repeated_future_time_feat = future_time_feat.repeat_interleave(
                    self.model.num_parallel_samples, 0
                )

            for t in range(self.prediction_length):
                if self.time_feat:
                    params, loc, scale, _ = self.model(
                        *args,
                        past_time_feat=repeated_past_time_feat,
                        future_time_feat=repeated_future_time_feat[..., : t + 1, :],
                        past_target=repeated_past_target,
                        past_observed_values=repeated_past_observed_values,
                        use_kv_cache=self.use_kv_cache,
                    )
                else:
                    params, loc, scale, _ = self.model(
                        *args,
                        past_time_feat=None,
                        future_time_feat=None,
                        past_target=repeated_past_target,
                        past_observed_values=repeated_past_observed_values,
                        use_kv_cache=self.use_kv_cache,
                    )

                sliced_params = [p[:, -1:] for p in params]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                sample = distr.sample()
                if self.nonnegative_pred_samples:
                    sample = F.relu(sample)
                future_samples.append(sample)

                repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
                repeated_past_observed_values = torch.cat(
                    (repeated_past_observed_values, torch.ones_like(sample)), dim=1
                )

        self.model.reset_cache()
        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )

    @torch.no_grad()
    def one_step_distribution(self, *args, **kwargs):
        """Return the exact next-step distribution used by the sampler.

        This is intentionally separate from ``forward`` so normal GluonTS
        prediction continues to return samples.  Probabilistic evaluation needs
        the distribution parameters before they are discarded by sampling.
        """
        if self.nonnegative_pred_samples:
            raise ValueError(
                "one_step_distribution does not support nonnegative_pred_samples=True: "
                "ReLU sampling creates a censored distribution with a point mass at zero."
            )
        if self.prediction_length != 1:
            raise ValueError("one_step_distribution requires prediction_length=1")

        try:
            params, loc, scale, _ = self.model(
                *args,
                past_time_feat=kwargs.get("past_time_feat") if self.time_feat else None,
                future_time_feat=kwargs.get("future_time_feat") if self.time_feat else None,
                past_target=kwargs["past_target"],
                past_observed_values=kwargs["past_observed_values"],
                use_kv_cache=self.use_kv_cache,
            )
            sliced_params = [param[:, -1:] for param in params]
            return self.model.distr_output.distribution(sliced_params, loc, scale)
        finally:
            self.model.reset_cache()

    def _compute_loss(self, batch, do_not_average=False, return_observed_values=False):
        """Compute training/evaluation loss components.

        Args:
            batch: Dict of tensors from the data loader.
            do_not_average: If False (default), returns averaged scalar losses
                suitable for training/validation. If True, returns elementwise
                NLL tensors suitable for per-sample evaluation metrics.
            return_observed_values: If True, wraps the loss tuple in an outer
                tuple with the observed_values mask appended.

        Returns:
            If return_observed_values is False:
                (total_loss, nll_loss, moe_aux_loss)
            If return_observed_values is True:
                ((total_loss, nll_loss, moe_aux_loss), observed_values)

            When do_not_average=False:
                total_loss = mean(NLL) + aux_weight * moe_aux_loss  (scalar)
                nll_loss   = observed-weighted mean NLL              (scalar)
                moe_aux_loss                                         (scalar)

            When do_not_average=True:
                total_loss = nll_loss = NLL * observed_values        (tensor)
                moe_aux_loss                                         (scalar)
                Note: aux loss is batch-level regularization and is excluded
                from total_loss in this mode to avoid broadcasting it across
                elements.
        """
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )

        distr_args, loc, scale, moe_aux_loss = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
        )

        context_target = take_last(past_target, dim=-1, num=self.context_length - 1)
        target = torch.cat((context_target, future_target_reshaped), dim=1)
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )
        observed_values = torch.cat((context_observed, future_observed_reshaped), dim=1)

        distr = self.model.distr_output.distribution(distr_args, loc=loc, scale=scale)
        nll_elementwise = self.loss(distr, target) * observed_values
        moe_aux_weight = float(self.model_kwargs.get("moe_aux_loss_weight", 0.0))
        if not do_not_average:
            nll_loss = nll_elementwise.sum() / observed_values.sum().clamp_min(1.0)
            total_loss = nll_loss + (moe_aux_weight * moe_aux_loss)
        else:
            # The routing auxiliary loss is batch-level regularization, so the
            # unreduced path exposes only elementwise NLL terms.
            nll_loss = nll_elementwise
            total_loss = nll_loss

        if not return_observed_values:
            return total_loss, nll_loss, moe_aux_loss
        return (total_loss, nll_loss, moe_aux_loss), observed_values

    def training_loss_components(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep the scalar `training_loss()` API stable while exposing split losses
        # for estimator-side logging without bypassing training-time augmentations.
        if random.random() < self.aug_prob:
            if self.freq_mask_rate > 0:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mask_rate,
                )
            if self.freq_mixing_rate:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mixing_rate,
                )
            if len(self.transforms):
                batch["past_target"], batch["future_target"] = self.augmentations(
                    batch["past_target"], batch["future_target"]
                )

        return self._compute_loss(batch, do_not_average=False, return_observed_values=False)

    def training_loss(self, batch: dict) -> torch.Tensor:
        total_loss, _, _ = self.training_loss_components(batch)
        return total_loss

    @torch.no_grad()
    def validation_loss_components(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Validation has no augmentation path, but mirrors the split-loss API used
        # by the estimator so logging does not change `validation_loss()` semantics.
        return self._compute_loss(batch, do_not_average=False, return_observed_values=False)

    @torch.no_grad()
    def validation_loss(self, batch: dict) -> torch.Tensor:
        total_loss, _, _ = self.validation_loss_components(batch)
        return total_loss

    def validation_routing_statistics(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        """Return additive MoE statistics from the latest validation forward pass."""
        return self.model.routing_statistics()
