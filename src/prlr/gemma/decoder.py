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
        eos_token_ids: Optional[Sequence[int]] = None,
        adapter: Optional[Any] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

        # Dynamically adapt hidden dimensions if backbone manifest indicates 3840 (Gemma 4)
        manifest = getattr(backbone, "manifest", None)
        manifest_dim = getattr(manifest, "hidden_dimension", None) if manifest is not None else None
        if manifest_dim is not None and prefix_dim == 2048 and hidden_dim == 2048 and manifest_dim != 2048:
            prefix_dim = manifest_dim
            hidden_dim = manifest_dim

        self.prefix_dim = prefix_dim
        self.hidden_dim = hidden_dim
        self.project_prefix = project_prefix

        if eos_token_ids is not None:
            self.eos_token_ids = set(eos_token_ids)
        else:
            # Architecture-aware EOS tokens: Gemma 4 uses {1, 106} (<eos>, <turn|>),
            # while Gemma 2 uses {1, 107} (<eos>, <end_of_turn>).
            # Note: Token 107 in Gemma 4 is '\n' and must not trigger early halting!
            if self.is_gemma4_architecture():
                self.eos_token_ids = {1, 106}
            else:
                self.eos_token_ids = {1, 107}

        if project_prefix and prefix_dim != hidden_dim:
            self.prefix_proj = nn.Linear(prefix_dim, hidden_dim, bias=False)
            self.prefix_norm = MLXRMSNorm(hidden_dim)
        elif project_prefix:
            self.prefix_proj = nn.Linear(prefix_dim, hidden_dim, bias=False)
            self.prefix_norm = MLXRMSNorm(hidden_dim)
        else:
            self.prefix_proj = None
            self.prefix_norm = None

    def decode_lm_head(self, h_sliced: mx.array) -> mx.array:
        """Sliced LM head decoding: projects sliced hidden states through the output head.

        Only evaluates the vocabulary projection over the required target positions,
        avoiding multi-gigabyte memory allocations for the 262,144 token vocabulary.
        """
        if self.is_gemma4_architecture():
            lang_model = getattr(self.backbone.model, "language_model", self.backbone.model)
            inner = getattr(lang_model, "model", lang_model)
            if getattr(lang_model, "tie_word_embeddings", True):
                logits = inner.embed_tokens.as_linear(h_sliced)
            else:
                logits = lang_model.lm_head(h_sliced)
            softcap = getattr(lang_model, "final_logit_softcapping", None)
            if softcap is not None:
                logits = mx.tanh(logits / softcap) * softcap
            return logits

        inner = self.get_inner_model()
        return inner.embed_tokens.as_linear(h_sliced)

    def prepare_prefix(self, latents: mx.array) -> mx.array:
        """Project and normalize working memory slots into soft prefix embeddings."""
        if self.prefix_proj is not None:
            latents = self.prefix_proj(latents)
            if self.prefix_norm is not None:
                latents = self.prefix_norm(latents)
        return latents

    def is_gemma4_architecture(self) -> bool:
        """Check whether backbone model uses Gemma 4 architecture."""
        model = getattr(self.backbone, "model", self.backbone)
        if hasattr(model, "language_model"):
            return True
        manifest = getattr(self.backbone, "manifest", None)
        if manifest is not None and "gemma-4" in getattr(manifest, "model_id", ""):
            return True
        return False

    def get_inner_model(self) -> Any:
        """Extract inner transformer model from backbone."""
        if hasattr(self.backbone, "model") and self.backbone.model is not None:
            if hasattr(self.backbone.model, "language_model"):
                return self.backbone.model.language_model.model
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
        hidden_size = getattr(
            getattr(inner, "config", None),
            "hidden_size",
            getattr(getattr(inner, "args", None), "hidden_size", self.hidden_dim),
        )

        if self.is_gemma4_architecture():
            prompt_embeds = inner.embed_tokens(prompt_ids)
            soft_prefix = (self.prepare_prefix(prefix_latents) * (hidden_size ** -0.5)).astype(prompt_embeds.dtype)
            if T_len > 1:
                target_inputs = target_ids[:, :-1]
                target_embeds = inner.embed_tokens(target_inputs)
                all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
            else:
                all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

            # Sliced LM head forward pass: evaluate layers without full vocabulary expansion
            h = inner(inputs=None, input_embeddings=all_embeds)
            start_idx = M + P - 1
            end_idx = start_idx + T_len
            h_target = h[:, start_idx:end_idx, :]
            target_logits = self.decode_lm_head(h_target)
        else:
            prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
            soft_prefix = self.prepare_prefix(prefix_latents).astype(prompt_embeds.dtype)

            if T_len > 1:
                target_inputs = target_ids[:, :-1]
                target_embeds = inner.embed_tokens(target_inputs) * (hidden_size ** 0.5)
                all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
            else:
                all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

            mask = create_attention_mask(all_embeds, cache=None)
            h = all_embeds
            for layer in inner.layers:
                h = layer(h, mask=mask, cache=None)
            h = inner.norm(h)
            all_logits = inner.embed_tokens.as_linear(h)
            start_idx = M + P - 1
            end_idx = start_idx + T_len
            target_logits = all_logits[:, start_idx:end_idx, :]

        losses = nn.losses.cross_entropy(target_logits, target_ids)

        if target_mask is None:
            eos_id_halt = 106 if self.is_gemma4_architecture() else 107
            is_eos = (target_ids == 1) | (target_ids == eos_id_halt)
            eos_cumsum = mx.cumsum(is_eos.astype(mx.int32), axis=1)
            valid_mask = (eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)
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
        hidden_size = getattr(
            getattr(inner, "config", None),
            "hidden_size",
            getattr(getattr(inner, "args", None), "hidden_size", 2048),
        )

        if self.is_gemma4_architecture():
            prompt_embeds = inner.embed_tokens(prompt_ids)
            if prefix_latents is not None and prefix_latents.shape[1] > 0:
                soft_prefix = (self.prepare_prefix(prefix_latents) * (hidden_size ** -0.5)).astype(prompt_embeds.dtype)
                h_prefill = mx.concatenate([soft_prefix, prompt_embeds], axis=1)
            else:
                h_prefill = prompt_embeds

            # Sliced LM head: only evaluate LM head on the last token hidden state
            h = inner(inputs=None, input_embeddings=h_prefill)
            h_last = h[:, -1:, :]
            return self.decode_lm_head(h_last)

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
            eos_token_ids: Set of token IDs triggering halt (defaults to architecture-aware EOS set).

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
            D = self.hidden_dim

        inner = self.get_inner_model()
        hidden_size = getattr(
            getattr(inner, "config", None),
            "hidden_size",
            getattr(getattr(inner, "args", None), "hidden_size", D),
        )

        if self.is_gemma4_architecture():
            prompt_embeds = inner.embed_tokens(prompt_ids)
            if prefix_latents is not None and prefix_latents.shape[1] > 0:
                soft_prefix = (self.prepare_prefix(prefix_latents) * (hidden_size ** -0.5)).astype(prompt_embeds.dtype)
                h_prefill = mx.concatenate([soft_prefix, prompt_embeds], axis=1)
            else:
                h_prefill = prompt_embeds

            cache = make_prompt_cache(self.backbone.model)
            # Sliced LM head prefill: pass through transformer body, slice last hidden, decode LM head
            h = inner(inputs=None, cache=cache, input_embeddings=h_prefill)
            h_last = h[:, -1:, :]
            logits = self.decode_lm_head(h_last)

            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1)
            else:
                next_tok = mx.random.categorical(logits / temperature)

            generated_tokens: list[mx.array] = [next_tok]

            first_tok_val = next_tok[0, 0].item()
            if B == 1 and first_tok_val in stop_tokens:
                return next_tok

            for _ in range(max_new_tokens - 1):
                # Sliced LM head step decoding
                h_step = inner(inputs=next_tok, cache=cache)
                step_logits = self.decode_lm_head(h_step)
                if temperature <= 1e-5:
                    next_tok = mx.argmax(step_logits, axis=-1)
                else:
                    next_tok = mx.random.categorical(step_logits / temperature)

                generated_tokens.append(next_tok)

                if B == 1 and next_tok[0, 0].item() in stop_tokens:
                    break

            return mx.concatenate(generated_tokens, axis=1)

        # Gemma 2 / compact model fallback path
        prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
        if prefix_latents is not None and prefix_latents.shape[1] > 0:
            soft_prefix = self.prepare_prefix(prefix_latents).astype(prompt_embeds.dtype)
            h_prefill = mx.concatenate([soft_prefix, prompt_embeds], axis=1)
        else:
            h_prefill = prompt_embeds

        cache = make_prompt_cache(inner)
        mask = create_attention_mask(h_prefill, cache[0])

        h = h_prefill
        for layer, c in zip(inner.layers, cache):
            h = layer(h, mask=mask, cache=c)
        h = inner.norm(h)
        logits = inner.embed_tokens.as_linear(h[:, -1:, :])

        if temperature <= 1e-5:
            next_tok = mx.argmax(logits, axis=-1)
        else:
            next_tok = mx.random.categorical(logits / temperature)

        generated_tokens: list[mx.array] = [next_tok]

        first_tok_val = next_tok[0, 0].item()
        if B == 1 and first_tok_val in stop_tokens:
            return next_tok

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

            if B == 1 and next_tok[0, 0].item() in stop_tokens:
                break

        return mx.concatenate(generated_tokens, axis=1)

    def decode_tokens(
        self,
        soft_prefix_latents: mx.array,
        max_new_tokens: int = 64,
        eos_token_id: int = 1,
        prompt_ids: mx.array | None = None,
    ) -> tuple[mx.array, str]:
        """Causal decoding convenience method conforming to PROJECT.md interface contract.

        Passes concatenated prefix latents and prompt embeds to the backbone model,
        generating causal next tokens and returning the token array and decoded string.
        """
        if prompt_ids is None:
            bos_id = getattr(self.backbone, "manifest", None)
            bos = getattr(bos_id, "bos_token_id", 2) if bos_id is not None else 2
            B = soft_prefix_latents.shape[0] if soft_prefix_latents.ndim == 3 else 1
            prompt_ids = mx.full((B, 1), bos, dtype=mx.int32)

        halt_tokens = {eos_token_id, 106 if self.is_gemma4_architecture() else 107}
        gen_tokens = self.generate(
            prompt_ids=prompt_ids,
            prefix_latents=soft_prefix_latents,
            max_new_tokens=max_new_tokens,
            eos_token_ids=halt_tokens,
        )
        tokenizer = getattr(self.backbone, "tokenizer", None)
        decoded_text = ""
        if tokenizer is not None:
            tok_list = gen_tokens[0].tolist() if gen_tokens.ndim == 2 else gen_tokens.tolist()
            decoded_text = tokenizer.decode(tok_list)
        return gen_tokens, decoded_text


__all__ = ["GemmaCausalPrefixDecoder"]
