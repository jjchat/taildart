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

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.torch.distributions import DistributionOutput

from forecasting.model.base_module import TailDartModelBase 
from forecasting.model.rope_norm import ( 
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    RMSNorm,
    apply_rotary_pos_emb,
)


@dataclass
class TailDartConfig:
    feature_size: int
    block_size: int
    n_layer: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_type: str = "dense"
    moe_num_experts: int = 3
    moe_top_k: int = 1
    moe_router_jitter_noise: float = 0.0
    moe_aux_loss_weight: float = 0.01
    rope_scaling: Optional[dict] = None
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6


class TailDartMLP(nn.Module):
    def __init__(self, config: TailDartConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return output, x.new_zeros(())


class TailDartMoE(nn.Module):
    def __init__(self, config: TailDartConfig) -> None:
        super().__init__()
        if int(config.moe_num_experts) < 2:
            raise ValueError("`moe_num_experts` must be >= 2 in MoE-only mode.")
        if int(config.moe_top_k) != 1:
            raise ValueError("Only top-1 routing is supported (`moe_top_k` must equal 1).")
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.router_jitter_noise = float(config.moe_router_jitter_noise)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([TailDartMLP(config) for _ in range(self.num_experts)])
        self.last_router_indices: Optional[torch.Tensor] = None
        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_tokens_per_expert: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        if self.training and self.router_jitter_noise > 0.0:
            router_logits = router_logits + self.router_jitter_noise * torch.randn_like(router_logits)

        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.max(router_probs, dim=-1)
        assignment = F.one_hot(top_indices, num_classes=self.num_experts).to(router_probs.dtype)

        combined = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            token_mask = top_indices == expert_idx
            if not torch.any(token_mask):
                continue
            expert_output, _ = expert(x[token_mask])
            combined[token_mask] = expert_output * top_probs[token_mask].unsqueeze(-1)

        mean_router_probs = router_probs.mean(dim=(0, 1))
        mean_assignment = assignment.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(mean_router_probs * mean_assignment)

        self.last_router_indices = top_indices.detach()
        self.last_router_probs = router_probs.detach()
        self.last_tokens_per_expert = assignment.sum(dim=(0, 1)).detach()
        return combined, aux_loss

    def routing_statistics(
        self,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, int]]:
        """Return the latest additive routing totals for this MoE layer."""
        if self.last_router_probs is None or self.last_tokens_per_expert is None:
            return None
        token_count = self.last_router_probs.shape[0] * self.last_router_probs.shape[1]
        return (
            self.last_router_probs.sum(dim=(0, 1)),
            self.last_tokens_per_expert,
            int(token_count),
        )


FFN_REGISTRY = {
    "dense": TailDartMLP,
    "moe": TailDartMoE,
}


class TailDartAttention(nn.Module):
    def __init__(self, config: TailDartConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.rope_scaling = config.rope_scaling
        self._rope_scaling_validation()

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self._init_rope()
        self.kv_cache = None

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`."
            )
        rope_scaling_type = self.rope_scaling.get("type")
        rope_scaling_factor = self.rope_scaling.get("factor")
        if rope_scaling_type not in ["linear", "dynamic", "nope"]:
            raise ValueError(
                f"`rope_scaling` type must be one of ['linear', 'dynamic', 'nope'], got {rope_scaling_type}"
            )
        if rope_scaling_type in ["linear", "dynamic"]:
            if (
                rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor < 1.0
            ):
                raise ValueError(
                    f"`rope_scaling` factor must be a float >= 1.0, got {rope_scaling_factor}"
                )

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.block_size
            )
            return

        scaling_type = self.rope_scaling["type"]
        scaling_factor = self.rope_scaling["factor"]
        if scaling_type == "nope":
            self.rotary_emb = None
        elif scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.block_size,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.block_size,
                scaling_factor=scaling_factor,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})."
            )
        return x.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        bsz, tgt_len, _ = x.size()
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cache_initialized = self.kv_cache is not None
        if use_kv_cache:
            if cache_initialized:
                k = torch.cat([self.kv_cache[0], k], dim=2)[:, :, 1:]
                v = torch.cat([self.kv_cache[1], v], dim=2)[:, :, 1:]
                self.kv_cache = (k, v)
            else:
                self.kv_cache = (k, v)

        true_seq_len = k.size(2)
        if self.rotary_emb is not None:
            if use_kv_cache and cache_initialized:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=true_seq_len)
                q, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=[-1])
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=true_seq_len)
                _, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
            else:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=tgt_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attention_dropout = self.dropout if self.training else 0.0

        if use_kv_cache and cache_initialized:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=attention_dropout, is_causal=False
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=attention_dropout, is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)
        return self.o_proj(y)


class TailDartBlock(nn.Module):
    def __init__(self, config: TailDartConfig) -> None:
        super().__init__()
        self.pre_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TailDartAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        try:
            ffn_cls = FFN_REGISTRY[config.ffn_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported ffn_type `{config.ffn_type}`.") from exc
        self.ffn = ffn_cls(config)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.self_attn(self.pre_attention_layernorm(x), use_kv_cache)
        x = x + self.post_attention_layernorm(attn_out)
        mlp_out, aux_loss = self.ffn(self.pre_mlp_layernorm(x))
        x = x + self.post_mlp_layernorm(mlp_out)
        return x, aux_loss


class TailDartModel(TailDartModelBase):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        dropout: float = 0.0,
        ffn_type: str = "dense",
        taildart_hidden_size: int = 144,
        taildart_intermediate_size: int = 576,
        taildart_num_heads: int = 8,
        taildart_num_kv_heads: int = 4,
        taildart_head_dim: int = 18,
        taildart_rms_norm_eps: float = 1e-6,
        moe_num_experts: int = 3,
        moe_top_k: int = 1,
        moe_router_jitter_noise: float = 0.0,
        moe_aux_loss_weight: float = 0.01,
        **_: dict,
    ) -> None:
        hidden_size = taildart_hidden_size
        num_heads = taildart_num_heads
        head_dim = taildart_head_dim
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"TailDart hidden size ({hidden_size}) must equal num_heads ({num_heads}) * head_dim ({head_dim})."
            )
        num_kv_heads = taildart_num_kv_heads
        intermediate_size = taildart_intermediate_size

        super().__init__(
            context_length=context_length,
            scaling=scaling,
            input_size=input_size,
            lags_seq=lags_seq,
            distr_output=distr_output,
            hidden_size=hidden_size,
            num_parallel_samples=num_parallel_samples,
            time_feat=time_feat,
        )
        self.ffn_type = str(ffn_type).strip().lower()
        if self.ffn_type not in FFN_REGISTRY:
            raise ValueError(f"`ffn_type` must be one of: {', '.join(sorted(FFN_REGISTRY))}.")
        config = TailDartConfig(
            feature_size=self.feature_size,
            block_size=max_context_length,
            n_layer=n_layer,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            ffn_type=self.ffn_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_router_jitter_noise=moe_router_jitter_noise,
            moe_aux_loss_weight=moe_aux_loss_weight,
            rope_scaling=rope_scaling,
            dropout=dropout,
            rms_norm_eps=taildart_rms_norm_eps,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.feature_size, config.hidden_size, bias=False),
                h=nn.ModuleList([TailDartBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        )

    def _forward_backbone(
        self, transformer_input: torch.Tensor, use_kv_cache: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.transformer.wte(transformer_input)
        moe_aux_losses = []
        for block in self.transformer.h:
            x, block_aux_loss = block(x, use_kv_cache)
            moe_aux_losses.append(block_aux_loss)
        if moe_aux_losses:
            moe_aux_loss = torch.stack(moe_aux_losses).mean()
        else:
            moe_aux_loss = x.new_tensor(0.0)
        return self.transformer.ln_f(x), moe_aux_loss

    def _reset_kv_cache(self) -> None:
        for block in self.transformer.h:
            block.self_attn.kv_cache = None

    def routing_statistics(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        """Return the latest routing totals for every MoE layer."""
        statistics = []
        for block in self.transformer.h:
            if isinstance(block.ffn, TailDartMoE):
                layer_statistics = block.ffn.routing_statistics()
                if layer_statistics is not None:
                    statistics.append(layer_statistics)
        return statistics
