# Copyright 2024 Arjun Ashok
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

from typing import List, Optional

import torch
from torch import nn

from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand

from gluon_utils.scalers.robust_scaler import RobustScaler


class TailDartModelBase(nn.Module):
    """Shared lag-based preprocessing and probabilistic output projection."""

    def __init__(
        self,
        context_length: int,
        scaling: str,
        input_size: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        hidden_size: int,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.lags_seq = lags_seq
        self.time_feat = time_feat
        if time_feat:
            self.feature_size = input_size * (len(self.lags_seq)) + 2 * input_size + 6
        else:
            self.feature_size = input_size * (len(self.lags_seq)) + 2 * input_size

        self.num_parallel_samples = num_parallel_samples
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        elif scaling == "robust":
            self.scaler = RobustScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(hidden_size)
        self.y_cache = False

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)

        if future_target is not None:
            input_seq = torch.cat(
                (
                    scaled_past_target[..., max(self.lags_seq) :],
                    (future_target[..., :-1] - loc) / scale,
                ),
                dim=-1,
            )
        else:
            input_seq = scaled_past_target[..., max(self.lags_seq) :]

        if (past_time_feat is not None) and (future_time_feat is not None):
            time_feat = torch.cat(
                (
                    past_time_feat[..., max(self.lags_seq) :, :],
                    future_time_feat[..., :-1, :],
                ),
                dim=1,
            )

        prior_input = (past_target[..., : max(self.lags_seq)] - loc) / scale
        lags = lagged_sequence_values(self.lags_seq, prior_input, input_seq, dim=-1)

        static_feat = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)
        expanded_static_feat = unsqueeze_expand(static_feat, dim=-2, size=lags.shape[-2])

        if (past_time_feat is not None) and (future_time_feat is not None):
            return torch.cat((lags, expanded_static_feat, time_feat), dim=-1), loc, scale
        return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale

    def _forward_backbone(
        self, transformer_input: torch.Tensor, use_kv_cache: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _reset_kv_cache(self) -> None:
        raise NotImplementedError

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )
        if use_kv_cache and self.y_cache:
            transformer_input = transformer_input[:, -1:]

        x, moe_aux_loss = self._forward_backbone(transformer_input, use_kv_cache)
        if use_kv_cache:
            self.y_cache = True

        params = self.param_proj(x)
        return params, loc, scale, moe_aux_loss

    def reset_cache(self) -> None:
        self.y_cache = False
        self._reset_kv_cache()
