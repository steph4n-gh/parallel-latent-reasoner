"""MLX Model Architecture for Parallel Latent Reasoner.

Implements pure MLX modules for the Gemma-class recurrent latent deliberator:
- Sinusoidal step position embeddings
- MLXRMSNorm with Gemma (1.0 + weight) parameterization
- MLXAdaRMSNorm step conditioning with zero-initialized projection
- MLXGemmaAttention with RoPE and cross-attention support
- MLXGemmaMLP (GeGLU gated feedforward network)
- MLXRecurrentGemmaBlock with ReZero residual scaling (alpha <= 0.05)
- MLXPreludeProjection for memory slot initialization
- MLXCodaLMHead for discrete decoding with logit soft-capping
- MLXCompactGemmaModel complete unified architecture
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import mlx.core as mx
import mlx.nn as nn

from parallel_latent_reasoner.config import GemmaLatentConfig

if TYPE_CHECKING:
    from parallel_latent_reasoner.engine import DeliberationResult


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
    """Gemma-compliant RMSNorm with (1.0 + weight) parameterization.

    In canonical Gemma models, weights are initialized to 0.0 and scaled as (1.0 + weight).
    """

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

        # Zero-initialize output projection for exact identity at init
        self.mlp_l2.weight = mx.zeros((2 * dims, dims))
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


class MLXGemmaAttention(nn.Module):
    """Gemma Multi-Head / Grouped-Query Attention with RoPE and cross-attention support.

    Supports bidirectional non-causal attention across working memory slots, as well
    as cross-attention attending to static prompt key/value representations.
    """

    def __init__(self, config: GemmaLatentConfig):
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

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def create_prompt_kv(
        self,
        prompt_hiddens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute static key-value representations for the prompt context prefix.

        Args:
            prompt_hiddens: Context tensor of shape [B, P, D].

        Returns:
            Tuple (prompt_k, prompt_v) with shapes [B, num_kv_heads, P, head_dim].
        """
        B, P, _ = prompt_hiddens.shape
        k = (
            self.k_proj(prompt_hiddens)
            .reshape(B, P, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(prompt_hiddens)
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
        """Execute attention over memory slots and optional prompt context.

        Args:
            x: Working memory states of shape [B, M, D].
            prompt_kv: Precomputed prompt key/values (tuple) or prompt states (mx.array).
            prompt_len: RoPE offset for memory slots.
            mask: Optional attention mask (defaults to None for non-causal bidirectional).

        Returns:
            Attention output tensor of shape [B, M, D].
        """
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

        # Apply RoPE to memory slots with offset=prompt_len
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
        )  # [B, num_heads, M, head_dim]

        out = attn_out.transpose(0, 2, 1, 3).reshape(B, M, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLXGemmaMLP(nn.Module):
    """Gemma GeGLU Gated Feedforward Network.

    Computes: down_proj(gelu(gate_proj(x)) * up_proj(x)).
    """

    def __init__(self, config: GemmaLatentConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class MLXGemmaMoE(nn.Module):
    """Gemma 4 Mixture-of-Experts (MoE) FFN Layer with Top-K Routing.

    Supports top-k routing across E experts with intermediate dimension moe_intermediate_dim,
    and an optional shared GeGLU MLP. Vectorized parameter indexing and batched matmuls
    ensure optimal Metal GPU execution and strict memory residency.
    """

    def __init__(self, config: GemmaLatentConfig):
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

        # Top-k router projection
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)

        # 3D parameter matrices: [num_experts, out_features, in_features]
        self.gate_weight = mx.zeros((self.num_experts, self.moe_intermediate_dim, self.dim))
        self.up_weight = mx.zeros((self.num_experts, self.moe_intermediate_dim, self.dim))
        self.down_weight = mx.zeros((self.num_experts, self.dim, self.moe_intermediate_dim))

        # Shared dense MLP if intermediate_dim is configured
        self.has_shared = config.intermediate_dim is not None and config.intermediate_dim > 0
        if self.has_shared:
            self.shared_gate = nn.Linear(self.dim, config.intermediate_dim, bias=False)
            self.shared_up = nn.Linear(self.dim, config.intermediate_dim, bias=False)
            self.shared_down = nn.Linear(config.intermediate_dim, self.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Execute MoE routing and expert computation across memory slots.

        Args:
            x: Input activations of shape [B, M, D].

        Returns:
            Output activations of shape [B, M, D].
        """
        B, M, D = x.shape
        flat_x = x.reshape(B * M, D)
        N = B * M

        # 1. Routing logits and top-k selection
        router_logits = self.router(flat_x)  # [N, E]
        k = min(self.top_k, self.num_experts)
        topk_idx = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]  # [N, K]
        topk_logits = mx.take_along_axis(router_logits, topk_idx, axis=-1)  # [N, K]
        weights = mx.softmax(topk_logits, axis=-1)  # [N, K]

        # 2. Vectorized expert indexing & matmul
        w_gate = self.gate_weight[topk_idx]  # [N, K, moe_intermediate, D]
        w_up = self.up_weight[topk_idx]      # [N, K, moe_intermediate, D]
        w_down = self.down_weight[topk_idx]  # [N, K, D, moe_intermediate]

        x_exp = flat_x[:, None, :, None]
        x_exp = mx.broadcast_to(x_exp, (N, k, D, 1))

        gate = nn.gelu(w_gate @ x_exp)  # [N, K, moe_intermediate, 1]
        up = w_up @ x_exp               # [N, K, moe_intermediate, 1]
        h = gate * up                   # [N, K, moe_intermediate, 1]

        expert_out = (w_down @ h).squeeze(-1)  # [N, K, D]
        moe_out = mx.sum(expert_out * weights[..., None], axis=1)  # [N, D]
        moe_out = moe_out.reshape(B, M, D)

        if self.has_shared:
            shared_out = self.shared_down(nn.gelu(self.shared_gate(x)) * self.shared_up(x))
            moe_out = moe_out + shared_out

        return moe_out


class MLXRecurrentGemmaBlock(nn.Module):
    """Weight-tied recurrent block with AdaRMSNorm, Gemma Attention, GeGLU/MoE MLP, and ReZero.

    Applies residual updates scaled by ReZero alpha parameters (alpha <= 0.05):
        x = x + alpha_attn * Attention(AdaRMSNorm_1(x, t))
        x = x + alpha_mlp * MLP(AdaRMSNorm_2(x, t))
    """

    def __init__(self, config: GemmaLatentConfig):
        super().__init__()
        self.config = config
        self.norm1 = MLXAdaRMSNorm(
            dims=config.dim,
            step_embed_dim=config.step_embed_dim,
            eps=config.rms_norm_eps,
        )
        self.attn = MLXGemmaAttention(config)
        self.norm2 = MLXAdaRMSNorm(
            dims=config.dim,
            step_embed_dim=config.step_embed_dim,
            eps=config.rms_norm_eps,
        )
        if getattr(config, "enable_moe_block", False) or (
            config.num_experts is not None and config.num_experts > 0
        ):
            self.mlp = MLXGemmaMoE(config)
        else:
            self.mlp = MLXGemmaMLP(config)

        # ReZero residual scaling parameters
        self.alpha_attn = mx.array([config.rezero_alpha])
        self.alpha_mlp = mx.array([config.rezero_alpha])

    def __call__(
        self,
        x: mx.array,
        step: int | float | mx.array,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        prompt_len: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        # Pre-attention normalization and attention projection
        h_norm1 = self.norm1(x, step)
        attn_out = self.attn(
            h_norm1,
            prompt_kv=prompt_kv,
            prompt_len=prompt_len,
            mask=mask,
        )
        x_mid = x + self.alpha_attn * attn_out

        # Pre-FFN normalization and GeGLU projection
        h_norm2 = self.norm2(x_mid, step)
        mlp_out = self.mlp(h_norm2)
        x_out = x_mid + self.alpha_mlp * mlp_out

        return x_out


class MLXPreludeProjection(nn.Module):
    """Prelude Projection module initializing M continuous latent working memory slots.

    Scales token embeddings by sqrt(D) per Gemma specification, and combines learned
    base slot embeddings E_slot in R^(1, M, D) with pooled contextual prompt modulation.
    """

    def __init__(
        self,
        config: GemmaLatentConfig,
        embed_tokens: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_slots = config.num_memory_slots
        self.embed_tokens = (
            embed_tokens
            if embed_tokens is not None
            else nn.Embedding(config.vocab_size, config.dim)
        )

        # Learned base slot embeddings E_slot in R^(1, M, D)
        self.slot_embeddings = mx.zeros((1, self.num_slots, self.dim))

        # Context projection from pooled prompt
        self.context_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.norm = MLXRMSNorm(self.dim, eps=config.rms_norm_eps)

    def embed_prompt(self, input_ids: mx.array) -> mx.array:
        """Embed input tokens and scale by sqrt(D)."""
        scale = math.sqrt(self.dim)
        return self.embed_tokens(input_ids) * scale

    def __call__(
        self,
        prompt: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Initialize M memory slots S^(0) and return prompt contextual representations.

        Args:
            prompt: Token IDs [B, P] or raw hidden representations [B, P, D].

        Returns:
            Tuple of (initial_slots [B, M, D], prompt_hiddens [B, P, D]).
        """
        if prompt.ndim == 2 and prompt.dtype in (
            mx.int32,
            mx.int64,
            mx.uint32,
            mx.uint16,
            mx.int16,
        ):
            prompt_hiddens = self.embed_prompt(prompt)
        else:
            prompt_hiddens = prompt

        B, P, D = prompt_hiddens.shape
        if P == 0:
            raise ValueError("Prompt sequence length P must be >= 1, got 0.")
        # Pooled contextual representation across prompt tokens
        pooled = mx.mean(prompt_hiddens, axis=1, keepdims=True)  # [B, 1, D]
        context_mod = self.context_proj(pooled)  # [B, 1, D]

        # Combine learned slot anchors and contextual modulation
        base_slots = mx.broadcast_to(self.slot_embeddings, (B, self.num_slots, D))
        slots = base_slots + context_mod
        slots = self.norm(slots)

        return slots, prompt_hiddens


class MLXCodaLMHead(nn.Module):
    """Discrete Coda / LM Head decoding latent memory into vocabulary logits.

    Applies final RMSNorm, latent readout pooling / projection, and vocabulary
    projection with Gemma logit soft-capping. Supports weight-tying with token embeddings.
    """

    def __init__(
        self,
        config: GemmaLatentConfig,
        embed_tokens: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        self.soft_cap = config.final_logit_softcapping

        self.final_norm = MLXRMSNorm(self.dim, eps=config.rms_norm_eps)
        if not config.tie_word_embeddings or embed_tokens is None:
            self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
            self.embed_tokens = None
        else:
            self.lm_head = None
            self.embed_tokens = embed_tokens

        self.readout_proj = nn.Linear(self.dim, self.dim, bias=False)

    def pool_readout(self, memory_slots: mx.array) -> mx.array:
        """Pool M memory slots into a single deliberated state per batch item."""
        normed = self.final_norm(memory_slots)
        pooled = mx.mean(normed, axis=1)  # [B, D]
        return self.readout_proj(pooled)

    def project_logits(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits with optional soft-capping."""
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        elif self.embed_tokens is not None:
            logits = hidden_states @ self.embed_tokens.weight.T
        else:
            raise RuntimeError("Neither lm_head nor embed_tokens is configured.")

        if self.soft_cap is not None and self.soft_cap > 0:
            logits = self.soft_cap * mx.tanh(logits / self.soft_cap)

        return logits

    def __call__(self, memory_slots: mx.array, pool: bool = True) -> mx.array:
        """Compute vocabulary logits from refined memory slots.

        Args:
            memory_slots: Refined memory states of shape [B, M, D].
            pool: If True, returns pooled readout logits [B, V]. If False, returns [B, M, V].

        Returns:
            Logits tensor [B, V] or [B, M, V].
        """
        if pool:
            h = self.pool_readout(memory_slots)
            return self.project_logits(h)
        else:
            normed = self.final_norm(memory_slots)
            return self.project_logits(normed)


class MLXCompactGemmaModel(nn.Module):
    """Complete MLX-Native Compact Gemma Architecture with Parallel Latent Deliberation."""

    def __init__(self, config: GemmaLatentConfig):
        super().__init__()
        self.config = config

        # Shared token embedding table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        # 1. Prelude Projection
        self.prelude = MLXPreludeProjection(config, embed_tokens=self.embed_tokens)

        # 2. Parallel Latent Deliberation Engine (Weight-Tied Core)
        # Import engine lazily to avoid circular imports
        from parallel_latent_reasoner.engine import MLXParallelLatentEngine

        self.engine = MLXParallelLatentEngine(config)

        # 3. Discrete Coda / LM Head
        self.coda = MLXCodaLMHead(config, embed_tokens=self.embed_tokens)

    def deliberate(
        self,
        prompt: mx.array,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> DeliberationResult:
        """Run prelude projection and T-step latent deliberation on prompt."""
        slots, prompt_hiddens = self.prelude(prompt)
        prompt_len = prompt_hiddens.shape[1]
        prompt_kv = self.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

        return self.engine.deliberate(
            initial_memory=slots,
            prompt_kv=prompt_kv,
            steps=steps,
            prompt_len=prompt_len,
            return_trajectory=return_trajectory,
        )

    def forward(
        self,
        input_ids: mx.array,
        steps: int | None = None,
        pool: bool = True,
    ) -> mx.array:
        """End-to-end forward pass: prompt -> prelude -> deliberation -> coda logits."""
        delib_result = self.deliberate(input_ids, steps=steps, return_trajectory=False)
        return self.coda(delib_result.final_states, pool=pool)

    def __call__(
        self,
        input_ids: mx.array,
        steps: int | None = None,
        pool: bool = True,
    ) -> mx.array:
        return self.forward(input_ids, steps=steps, pool=pool)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 16,
        steps: int | None = None,
        temperature: float = 0.0,
    ) -> mx.array:
        """Generate discrete solution tokens from the deliberated latent state."""
        delib_res = self.deliberate(input_ids, steps=steps)
        readout = self.coda.pool_readout(delib_res.final_states)  # [B, D]

        B = input_ids.shape[0]
        generated: list[mx.array] = []
        curr_hidden = readout

        for _ in range(max_new_tokens):
            logits = self.coda.project_logits(curr_hidden)  # [B, V]
            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1, keepdims=True)  # [B, 1]
            else:
                next_tok = mx.random.categorical(logits / temperature)[:, None]
            generated.append(next_tok)

            tok_embed = self.prelude.embed_prompt(next_tok)[:, 0, :]  # [B, D]
            curr_hidden = self.coda.final_norm(curr_hidden + 0.1 * tok_embed)

        return mx.concatenate(generated, axis=-1)

    def get_trainable_parameters(self) -> dict[str, mx.array]:
        """Extract all trainable adapter parameters for PRLR distillation.

        Returns:
            Dictionary mapping parameter names to mx.array tensors for Prelude,
            AdaRMSNorm step modulation (norm1 and norm2 mlp_l1/mlp_l2/weight in each layer),
            ReZero alpha parameters (alpha_attn, alpha_mlp), and Coda head.
        """
        trainable: dict[str, mx.array] = {
            "prelude.slot_embeddings": self.prelude.slot_embeddings,
            "prelude.context_proj.weight": self.prelude.context_proj.weight,
            "prelude.norm.weight": self.prelude.norm.weight,
        }

        for i, layer in enumerate(self.engine.layers):
            trainable[f"engine.layers.{i}.norm1.weight"] = layer.norm1.weight
            trainable[f"engine.layers.{i}.norm1.mlp_l1.weight"] = layer.norm1.mlp_l1.weight
            trainable[f"engine.layers.{i}.norm1.mlp_l1.bias"] = layer.norm1.mlp_l1.bias
            trainable[f"engine.layers.{i}.norm1.mlp_l2.weight"] = layer.norm1.mlp_l2.weight
            trainable[f"engine.layers.{i}.norm1.mlp_l2.bias"] = layer.norm1.mlp_l2.bias

            trainable[f"engine.layers.{i}.norm2.weight"] = layer.norm2.weight
            trainable[f"engine.layers.{i}.norm2.mlp_l1.weight"] = layer.norm2.mlp_l1.weight
            trainable[f"engine.layers.{i}.norm2.mlp_l1.bias"] = layer.norm2.mlp_l1.bias
            trainable[f"engine.layers.{i}.norm2.mlp_l2.weight"] = layer.norm2.mlp_l2.weight
            trainable[f"engine.layers.{i}.norm2.mlp_l2.bias"] = layer.norm2.mlp_l2.bias

            trainable[f"engine.layers.{i}.alpha_attn"] = layer.alpha_attn
            trainable[f"engine.layers.{i}.alpha_mlp"] = layer.alpha_mlp

        trainable["coda.final_norm.weight"] = self.coda.final_norm.weight
        trainable["coda.readout_proj.weight"] = self.coda.readout_proj.weight
        if self.coda.lm_head is not None:
            trainable["coda.lm_head.weight"] = self.coda.lm_head.weight

        return trainable

    def freeze_base_model(self) -> None:
        """Freeze base model backbone weights while keeping adapter parameters trainable."""
        self.freeze()
        self.prelude.unfreeze()
        if hasattr(self.prelude, "embed_tokens") and self.prelude.embed_tokens is not None:
            self.prelude.embed_tokens.freeze()

        self.coda.unfreeze()
        if hasattr(self.coda, "embed_tokens") and self.coda.embed_tokens is not None:
            self.coda.embed_tokens.freeze()

        for layer in self.engine.layers:
            layer.norm1.unfreeze()
            layer.norm2.unfreeze()
            layer.unfreeze(keys=["alpha_attn", "alpha_mlp"])

    def save_adapter_weights(self, filepath: str | Path) -> None:
        """Save adapter weights to .npz or .safetensors file.

        Args:
            filepath: Target file path (.npz or .safetensors).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        params = self.get_trainable_parameters()
        suffix = path.suffix.lower()

        if suffix == ".safetensors":
            mx.save_safetensors(str(path), params)
        else:
            if suffix != ".npz":
                path = path.with_suffix(".npz")
            mx.savez(str(path), **params)

    def load_adapter_weights(self, filepath: str | Path) -> dict[str, mx.array]:
        """Load and restore adapter weights into the model instance.

        Args:
            filepath: Source file path (.npz or .safetensors).

        Returns:
            Dictionary of loaded adapter parameters.
        """
        from mlx.utils import tree_unflatten

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Adapter weight file not found: {path}")

        loaded = dict(mx.load(str(path)))
        nested = tree_unflatten(list(loaded.items()))
        if "engine" in nested and "layers" in nested["engine"] and isinstance(nested["engine"]["layers"], list):
            n_layers = len(self.engine.layers)
            nested["engine"]["layers"] = nested["engine"]["layers"][:n_layers]
        self.update(nested)
        return loaded


__all__ = [
    "sinusoidal_step_embedding",
    "MLXRMSNorm",
    "MLXAdaRMSNorm",
    "MLXGemmaAttention",
    "MLXGemmaMLP",
    "MLXGemmaMoE",
    "MLXRecurrentGemmaBlock",
    "MLXPreludeProjection",
    "MLXCodaLMHead",
    "MLXCompactGemmaModel",
]
