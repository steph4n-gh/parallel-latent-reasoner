"""Pure MLX Recurrent Transformer Core Modules.

Provides model-agnostic recurrence operators:
- sinusoidal_step_embedding
- MLXRMSNorm with (1.0 + weight) parameterization
- MLXAdaRMSNorm step conditioning with zero-initialized projection
- MLXAttention / MLXGemmaAttention with RoPE and cross-attention support
- MLXMLP / MLXGemmaMLP (GeGLU gated feedforward network)
- MLXMoE / MLXGemmaMoE (Top-K mixture-of-experts layer)
- MLXRecurrentBlock / MLXRecurrentGemmaBlock with ReZero residual scaling
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_step_embedding(
    step: int | float | mx.array,
    dim: int,
    max_period: float = 10000.0,
) -> mx.array:
    """Compute sinusoidal step position embedding for deliberation step index t.

    Args:
        step: Step index t (int, float, or scalar mx.array).
        dim: Embedding dimension.
        max_period: Base wavelength divisor.

    Returns:
        mx.array of shape [1, dim] containing sinusoidal step features.
    """
    if isinstance(step, (int, float)):
        step_arr = mx.array([float(step)], dtype=mx.float32)
    elif isinstance(step, mx.array):
        if step.ndim == 0:
            step_arr = step.reshape((1,)).astype(mx.float32)
        else:
            step_arr = step.astype(mx.float32)
    else:
        step_arr = mx.array([float(step)], dtype=mx.float32)

    half_dim = dim // 2
    freqs = mx.exp(
        -math.log(max_period)
        * mx.arange(0, half_dim, dtype=mx.float32)
        / half_dim
    )
    args = step_arr[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2 != 0:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class MLXRMSNorm(nn.Module):
    """RMSNorm with (1.0 + weight) parameterization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class MLXAdaRMSNorm(nn.Module):
    """Step conditioning module for recurrent step t in [1..T].

    Computes sinusoidal step embeddings, projects through a 2-layer MLP
    to scale gamma_t and shift beta_t, and modulates normalized hidden states:
        norm_x * (1.0 + gamma_t) + beta_t

    Output projection layers are strictly zero-initialized for exact mathematical
    identity pass-through at initialization.
    """

    def __init__(
        self,
        dims: int,
        step_embed_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dims = dims
        self.step_embed_dim = step_embed_dim
        self.eps = eps
        self.weight = mx.zeros((dims,))

        # 2-layer step modulation MLP: [step_embed_dim] -> [dims] -> [2 * dims]
        self.mlp_l1 = nn.Linear(step_embed_dim, dims)
        self.mlp_l2 = nn.Linear(dims, 2 * dims)

        # Initialize output projection with small non-zero weights so mlp_l1 receives gradients
        self.mlp_l2.weight = mx.random.normal((2 * dims, dims)) * 1e-7
        self.mlp_l2.bias = mx.zeros((2 * dims,))


    def __call__(
        self,
        x: mx.array,
        step: int | float | mx.array,
    ) -> mx.array:
        """Modulate normalized hidden states by step embedding.

        Args:
            x: Input activations of shape [..., dims].
            step: Step index t (int/scalar) or precomputed step embedding.

        Returns:
            Modulated normalized activations of shape [..., dims].
        """
        if isinstance(step, (int, float)) or (
            isinstance(step, mx.array)
            and (step.ndim == 0 or (step.ndim == 1 and step.shape[0] != self.step_embed_dim))
        ):
            step_emb = sinusoidal_step_embedding(step, self.step_embed_dim)
        else:
            step_emb = step

        h = nn.silu(self.mlp_l1(step_emb))
        mod = self.mlp_l2(h)  # [..., 2 * dims]
        gamma = mod[..., : self.dims]
        beta = mod[..., self.dims :]

        norm_x = mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

        # Broadcast modulation across sequence dimension
        while gamma.ndim < norm_x.ndim:
            gamma = mx.expand_dims(gamma, axis=-2)
            beta = mx.expand_dims(beta, axis=-2)

        return norm_x * (1.0 + gamma) + beta


class MLXAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention with RoPE and cross-attention support."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        # Dedicated prompt cross-attention projections
        self.dedicated_cross_attention = getattr(config, "dedicated_cross_attention", False)
        if self.dedicated_cross_attention:
            self.k_cross_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
            self.v_cross_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        else:
            self.k_cross_proj = None
            self.v_cross_proj = None

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def create_prompt_kv(
        self,
        prompt_hiddens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute static key-value representations for prompt context prefix."""
        B, P, _ = prompt_hiddens.shape
        k_linear = self.k_cross_proj if self.k_cross_proj is not None else self.k_proj
        v_linear = self.v_cross_proj if self.v_cross_proj is not None else self.v_proj
        k = (
            k_linear(prompt_hiddens)
            .reshape(B, P, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            v_linear(prompt_hiddens)
            .reshape(B, P, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = self.rope(k, offset=0)
        return k, v


    def __call__(
        self,
        x: mx.array,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        prompt_len: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Execute attention over memory slots and optional prompt context."""
        B, M, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, M, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k_mem = (
            self.k_proj(x)
            .reshape(B, M, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v_mem = (
            self.v_proj(x)
            .reshape(B, M, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        q = self.rope(q, offset=prompt_len)
        k_mem = self.rope(k_mem, offset=prompt_len)

        if prompt_kv is not None:
            if isinstance(prompt_kv, tuple):
                prompt_k, prompt_v = prompt_kv
            else:
                prompt_k, prompt_v = self.create_prompt_kv(prompt_kv)
            k = mx.concatenate([prompt_k, k_mem], axis=2)
            v = mx.concatenate([prompt_v, v_mem], axis=2)
        else:
            k = k_mem
            v = v_mem

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        out = attn_out.transpose(0, 2, 1, 3).reshape(B, M, self.num_heads * self.head_dim)
        return self.o_proj(out)


MLXGemmaAttention = MLXAttention


class MLXMLP(nn.Module):
    """GeGLU Gated Feedforward Network."""

    def __init__(self, config: Any):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


MLXGemmaMLP = MLXMLP


class MLXMoE(nn.Module):
    """Mixture-of-Experts (MoE) FFN Layer with Top-K Routing."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_experts = config.num_experts or 128
        self.top_k = config.top_k_experts or 8
        self.moe_intermediate_dim = (
            config.moe_intermediate_dim
            or config.moe_intermediate_size
            or (config.intermediate_dim // 3 if config.intermediate_dim > 0 else 704)
        )

        self.router = nn.Linear(self.dim, self.num_experts, bias=False)

        # Principled scaled non-zero MoE initialization
        std_in = 1.0 / math.sqrt(self.dim)
        std_out = 1.0 / math.sqrt(self.moe_intermediate_dim)
        self.gate_weight = mx.random.normal((self.num_experts, self.moe_intermediate_dim, self.dim)) * std_in
        self.up_weight = mx.random.normal((self.num_experts, self.moe_intermediate_dim, self.dim)) * std_in
        self.down_weight = mx.random.normal((self.num_experts, self.dim, self.moe_intermediate_dim)) * std_out


        self.has_shared = config.intermediate_dim is not None and config.intermediate_dim > 0
        if self.has_shared:
            self.shared_gate = nn.Linear(self.dim, config.intermediate_dim, bias=False)
            self.shared_up = nn.Linear(self.dim, config.intermediate_dim, bias=False)
            self.shared_down = nn.Linear(config.intermediate_dim, self.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, M, D = x.shape
        flat_x = x.reshape(B * M, D)
        N = B * M

        router_logits = self.router(flat_x)
        k = min(self.top_k, self.num_experts)
        topk_idx = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        topk_logits = mx.take_along_axis(router_logits, topk_idx, axis=-1)
        weights = mx.softmax(topk_logits, axis=-1)

        w_gate = self.gate_weight[topk_idx]
        w_up = self.up_weight[topk_idx]
        w_down = self.down_weight[topk_idx]

        x_exp = flat_x[:, None, :, None]
        x_exp = mx.broadcast_to(x_exp, (N, k, D, 1))

        gate = nn.gelu(w_gate @ x_exp)
        up = w_up @ x_exp
        h = gate * up

        expert_out = (w_down @ h).squeeze(-1)
        moe_out = mx.sum(expert_out * weights[..., None], axis=1)
        moe_out = moe_out.reshape(B, M, D)

        if self.has_shared:
            shared_out = self.shared_down(nn.gelu(self.shared_gate(x)) * self.shared_up(x))
            moe_out = moe_out + shared_out

        return moe_out


MLXGemmaMoE = MLXMoE


class MLXRecurrentBlock(nn.Module):
    """Weight-tied recurrent block with AdaRMSNorm, Attention, MLP/MoE, and ReZero."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.norm1 = MLXAdaRMSNorm(
            dims=config.dim,
            step_embed_dim=config.step_embed_dim,
            eps=config.rms_norm_eps,
        )
        self.attn = MLXAttention(config)
        self.norm2 = MLXAdaRMSNorm(
            dims=config.dim,
            step_embed_dim=config.step_embed_dim,
            eps=config.rms_norm_eps,
        )
        if getattr(config, "enable_moe_block", False) or (
            getattr(config, "num_experts", None) is not None and config.num_experts > 0
        ):
            self.mlp = MLXMoE(config)
        else:
            self.mlp = MLXMLP(config)

        # Bounded residual scaling: alpha = alpha_max * sigmoid(raw_alpha)
        self.alpha_max = getattr(config, "alpha_max", 0.5)
        alpha_init = getattr(config, "rezero_alpha", 0.05)
        alpha_init = min(max(alpha_init, 1e-6), self.alpha_max - 1e-6)
        raw_val = math.log(alpha_init / (self.alpha_max - alpha_init))
        self.raw_alpha_attn = mx.array([raw_val])
        self.raw_alpha_mlp = mx.array([raw_val])
        self.alpha_attn = mx.array([alpha_init])
        self.alpha_mlp = mx.array([alpha_init])

    def __setattr__(self, key: str, val: Any) -> None:
        if key == "alpha_attn" and isinstance(val, mx.array):
            v = float(val.item())
            v = min(max(v, 1e-6), self.alpha_max - 1e-6)
            raw = math.log(v / (self.alpha_max - v))
            self.raw_alpha_attn = mx.array([raw])
        elif key == "alpha_mlp" and isinstance(val, mx.array):
            v = float(val.item())
            v = min(max(v, 1e-6), self.alpha_max - 1e-6)
            raw = math.log(v / (self.alpha_max - v))
            self.raw_alpha_mlp = mx.array([raw])
        super().__setattr__(key, val)

    def __setitem__(self, key: str, val: Any) -> None:
        if key == "alpha_attn" and isinstance(val, mx.array):
            v = float(val.item())
            v = min(max(v, 1e-6), self.alpha_max - 1e-6)
            raw = math.log(v / (self.alpha_max - v))
            self.raw_alpha_attn = mx.array([raw])
        elif key == "alpha_mlp" and isinstance(val, mx.array):
            v = float(val.item())
            v = min(max(v, 1e-6), self.alpha_max - 1e-6)
            raw = math.log(v / (self.alpha_max - v))
            self.raw_alpha_mlp = mx.array([raw])
        super().__setitem__(key, val)

    def __call__(
        self,
        x: mx.array,
        step: int | float | mx.array,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        prompt_len: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        h_norm1 = self.norm1(x, step)
        attn_out = self.attn(
            h_norm1,
            prompt_kv=prompt_kv,
            prompt_len=prompt_len,
            mask=mask,
        )
        effective_alpha_attn = self.alpha_max * mx.sigmoid(self.raw_alpha_attn)
        x_mid = x + effective_alpha_attn * attn_out

        h_norm2 = self.norm2(x_mid, step)
        mlp_out = self.mlp(h_norm2)
        effective_alpha_mlp = self.alpha_max * mx.sigmoid(self.raw_alpha_mlp)
        x_out = x_mid + effective_alpha_mlp * mlp_out

        return x_out




MLXRecurrentGemmaBlock = MLXRecurrentBlock


class MLXCrossAttention(MLXAttention):
    """Bidirectional slot self-attention + dedicated prompt cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
    ):
        class _CrossConfig:
            pass

        cfg = _CrossConfig()
        cfg.dim = dim
        cfg.num_heads = num_heads
        cfg.num_kv_heads = num_kv_heads
        cfg.head_dim = head_dim
        cfg.rope_theta = rope_theta
        cfg.dedicated_cross_attention = True
        super().__init__(cfg)


__all__ = [
    "sinusoidal_step_embedding",
    "MLXRMSNorm",
    "MLXAdaRMSNorm",
    "MLXAttention",
    "MLXGemmaAttention",
    "MLXCrossAttention",
    "MLXMLP",
    "MLXGemmaMLP",
    "MLXMoE",
    "MLXGemmaMoE",
    "MLXRecurrentBlock",
    "MLXRecurrentGemmaBlock",
]

