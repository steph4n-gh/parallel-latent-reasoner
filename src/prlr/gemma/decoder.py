"""Option A Causal Decoder for Pretrained Gemma.

Implements:
- Soft Prompt Prefix: prepends deliberated working memory slots (B, M, 2048) to prompt tokens
- Native MLX make_prompt_cache and KVCache integration
- Teacher-forcing causal forward pass with masked cross-entropy loss
- Autoregressive generate() with EOS halting checking {1, 107}
- Zero ungrounded linear recurrence (completely eradicating curr_hidden + 0.1 * tok_embed)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from prlr.kernel.recurrent_core import MLXRMSNorm


class GemmaCausalPrefixDecoder(nn.Module):
    """Option A Causal Prefix Decoder prepending soft memory slots to prompt tokens."""

    def __init__(
        self,
        backbone: Any,
        prefix_dim: int = 2048,
        hidden_dim: int = 2048,
        project_prefix: bool = False,
        eos_token_ids: Sequence[int] = (1, 107),
        adapter: Optional[Any] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.prefix_dim = prefix_dim
        self.hidden_dim = hidden_dim
        self.project_prefix = project_prefix
        self.eos_token_ids = set(eos_token_ids)

        if project_prefix and prefix_dim != hidden_dim:
            self.prefix_proj = nn.Linear(prefix_dim, hidden_dim, bias=False)
            self.prefix_norm = MLXRMSNorm(hidden_dim)
        elif project_prefix:
            self.prefix_proj = nn.Linear(prefix_dim, hidden_dim, bias=False)
            self.prefix_norm = MLXRMSNorm(hidden_dim)
        else:
            self.prefix_proj = None
            self.prefix_norm = None

    def prepare_prefix(self, latents: mx.array) -> mx.array:
        """Project and normalize working memory slots into soft prefix embeddings."""
        if self.prefix_proj is not None:
            latents = self.prefix_proj(latents)
            if self.prefix_norm is not None:
                latents = self.prefix_norm(latents)
        return latents

    def get_inner_model(self) -> Any:
        """Extract inner transformer model from backbone."""
        if hasattr(self.backbone, "model") and self.backbone.model is not None:
            if hasattr(self.backbone.model, "model"):
                return self.backbone.model.model
            return self.backbone.model
        return self.backbone

    def forward(
        self,
        prompt_ids: mx.array,
        prefix_latents: mx.array,
        target_ids: mx.array,
        target_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Teacher-forcing causal forward pass computing loss strictly on target tokens.

        Args:
            prompt_ids: Prompt token IDs of shape (B, P).
            prefix_latents: Deliberated working memory slots of shape (B, M, D).
            target_ids: Target token IDs of shape (B, T_len).
            target_mask: Optional boolean/float mask of shape (B, T_len).

        Returns:
            Tuple of (masked_loss, target_logits).
        """
        from mlx_lm.models.base import create_attention_mask

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None, :]
        if target_ids.ndim == 1:
            target_ids = target_ids[None, :]
        if prefix_latents.ndim == 2:
            prefix_latents = prefix_latents[None, :, :]

        B, P = prompt_ids.shape
        _, T_len = target_ids.shape
        B_lat, M, D = prefix_latents.shape
        assert B == B_lat, f"Batch size mismatch: prompt {B} vs latents {B_lat}"

        inner = self.get_inner_model()
        hidden_size = getattr(inner.args, "hidden_size", D)

        # 1. Embed prompt and scale by sqrt(D)
        prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)

        # 2. Prepare soft prefix slots
        soft_prefix = self.prepare_prefix(prefix_latents).astype(prompt_embeds.dtype)

        # 3. Embed teacher-forcing target prefix (all but last target token)
        if T_len > 1:
            target_inputs = target_ids[:, :-1]
            target_embeds = inner.embed_tokens(target_inputs) * (hidden_size ** 0.5)
            all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
        else:
            all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

        # 4. Causal forward evaluation
        mask = create_attention_mask(all_embeds, cache=None)
        h = all_embeds
        for layer in inner.layers:
            h = layer(h, mask=mask, cache=None)
        h = inner.norm(h)
        all_logits = inner.embed_tokens.as_linear(h)

        # 5. Extract target logits (positions predicting target_ids[0..T_len-1])
        start_idx = M + P - 1
        end_idx = start_idx + T_len
        target_logits = all_logits[:, start_idx:end_idx, :]

        # 6. Masked cross-entropy loss over active target tokens
        losses = nn.losses.cross_entropy(target_logits, target_ids)

        if target_mask is None:
            # Active tokens up to and including the first EOS token
            is_eos = (target_ids == 1) | (target_ids == 107)
            eos_cumsum = mx.cumsum(is_eos.astype(mx.int32), axis=1)
            valid_mask = (eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)
            # Exclude pad tokens (ID 0)
            valid_mask = valid_mask & (target_ids != 0)
            valid_mask = valid_mask.astype(mx.float32)
        else:
            valid_mask = target_mask.astype(mx.float32)

        num_valid = mx.maximum(mx.sum(valid_mask), 1.0)
        loss = mx.sum(losses * valid_mask) / num_valid

        return loss, target_logits

    def prefill_logits(
        self,
        prompt_ids: mx.array,
        prefix_latents: mx.array | None = None,
    ) -> mx.array:
        """Compute logits for the first token after prompt / prefix without decoding.

        Args:
            prompt_ids: Prompt token IDs of shape (B, P) or (P,).
            prefix_latents: Optional deliberated slots of shape (B, M, D) or (M, D).

        Returns:
            Logits tensor of shape (B, 1, V).
        """
        from mlx_lm.models.base import create_attention_mask

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None, :]
        if prefix_latents is not None and prefix_latents.ndim == 2:
            prefix_latents = prefix_latents[None, :, :]

        B, P = prompt_ids.shape
        inner = self.get_inner_model()
        hidden_size = getattr(inner.args, "hidden_size", 2048)

        prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
        if prefix_latents is not None and prefix_latents.shape[1] > 0:
            soft_prefix = self.prepare_prefix(prefix_latents).astype(prompt_embeds.dtype)
            h_prefill = mx.concatenate([soft_prefix, prompt_embeds], axis=1)
        else:
            h_prefill = prompt_embeds

        mask = create_attention_mask(h_prefill, cache=None)
        h = h_prefill
        for layer in inner.layers:
            h = layer(h, mask=mask, cache=None)
        h = inner.norm(h)
        logits = inner.embed_tokens.as_linear(h[:, -1:, :])
        return logits

    def generate(
        self,
        prompt_ids: mx.array,
        prefix_latents: mx.array | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        eos_token_ids: set[int] | Sequence[int] | None = None,
    ) -> mx.array:
        """Autoregressive generation with native MLX KVCache and EOS halting.

        Zero linear recurrence: generates tokens solely via causal self-attention
        conditioned on the deliberated soft prefix slots (or standard prompt if prefix is None).

        Args:
            prompt_ids: Prompt token IDs of shape (B, P) or (P,).
            prefix_latents: Deliberated working memory slots of shape (B, M, D) or (M, D), or None.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for greedy argmax).
            eos_token_ids: Set of token IDs triggering halt (defaults to {1, 107}).

        Returns:
            Generated token IDs of shape (B, generated_tokens).
        """
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import make_prompt_cache

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None, :]
        if prefix_latents is not None and prefix_latents.ndim == 2:
            prefix_latents = prefix_latents[None, :, :]

        stop_tokens = self.eos_token_ids if eos_token_ids is None else set(eos_token_ids)
        B, P = prompt_ids.shape
        if prefix_latents is not None and prefix_latents.shape[1] > 0:
            B_lat, M, D = prefix_latents.shape
            assert B == B_lat, f"Batch size mismatch: prompt {B} vs latents {B_lat}"
        else:
            D = 2048

        inner = self.get_inner_model()
        hidden_size = getattr(inner.args, "hidden_size", D)

        # 1. Embed prompt & prepare soft prefix
        prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
        if prefix_latents is not None and prefix_latents.shape[1] > 0:
            soft_prefix = self.prepare_prefix(prefix_latents).astype(prompt_embeds.dtype)
            h_prefill = mx.concatenate([soft_prefix, prompt_embeds], axis=1)
        else:
            h_prefill = prompt_embeds

        # 2. Native MLX prompt cache
        cache = make_prompt_cache(inner)
        mask = create_attention_mask(h_prefill, cache[0])

        # 3. Prefill pass
        h = h_prefill
        for layer, c in zip(inner.layers, cache):
            h = layer(h, mask=mask, cache=c)
        h = inner.norm(h)
        logits = inner.embed_tokens.as_linear(h[:, -1:, :])

        # 4. First token
        if temperature <= 1e-5:
            next_tok = mx.argmax(logits, axis=-1)  # (B, 1)
        else:
            next_tok = mx.random.categorical(logits / temperature)  # (B, 1)

        generated_tokens: list[mx.array] = [next_tok]

        # Check early exit on first token
        first_tok_val = next_tok[0, 0].item()
        if B == 1 and first_tok_val in stop_tokens:
            return next_tok

        # 5. Autoregressive step decoding with KV cache
        for _ in range(max_new_tokens - 1):
            tok_embed = inner.embed_tokens(next_tok) * (hidden_size ** 0.5)
            h = tok_embed
            for layer, c in zip(inner.layers, cache):
                h = layer(h, mask=None, cache=c)
            h = inner.norm(h)
            logits = inner.embed_tokens.as_linear(h)

            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1)
            else:
                next_tok = mx.random.categorical(logits / temperature)

            generated_tokens.append(next_tok)

            # Check halting condition
            if B == 1 and next_tok[0, 0].item() in stop_tokens:
                break

        return mx.concatenate(generated_tokens, axis=1)


__all__ = ["GemmaCausalPrefixDecoder"]
