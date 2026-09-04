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


class GatedCrossAttentionInjection(nn.Module):
    """Safe cross-attention conditioning injection with bounded zero-gate residual scaling.

    Guarantees the zero-gate base parity invariant:
        gate = 0 => behavior 100.000% bit-exact identical to frozen base model.

    Inputs:
        h: Contextual hidden representations from backbone of shape (B, L, D) or (L, D).
        slots: Deliberated latent slots from adapter of shape (B, M, D) or (M, D).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: int = 256,
        gamma_max: float = 0.5,
        rms_norm_eps: float = 1e-6,
        init_alpha: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gamma_max = gamma_max
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)

        # Scalar parameter initialized to init_alpha
        self.alpha = mx.array(init_alpha, dtype=mx.float32)

        # Slot normalization layer matching Gemma hidden state statistics
        self.slot_norm = MLXRMSNorm(dims=hidden_size, eps=rms_norm_eps)

        # Projections
        proj_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.out_proj = nn.Linear(proj_dim, hidden_size, bias=False)

        # Head-level QK-Norm normalizers
        self.q_norm = nn.RMSNorm(dims=head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=head_dim, eps=rms_norm_eps)

        self._training: bool = False
        self._last_telemetry: Optional[dict[str, float]] = None

    @property
    def training(self) -> bool:
        """Boolean indicating whether the injection module is in training mode."""
        return getattr(self, "_training", False)

    @training.setter
    def training(self, mode: bool) -> None:
        self._training = bool(mode)

    def _set_training_mode(self, mode: bool) -> None:
        self._training = bool(mode)

    @property
    def gate(self) -> mx.array:
        """Bounded gate value g = gamma_max * tanh(alpha) in (-gamma_max, +gamma_max)."""
        return self.gamma_max * mx.tanh(self.alpha)

    @property
    def gate_value(self) -> float:
        """Scalar gate value as a Python float."""
        return float(self.gate.item())

    @property
    def o_proj(self) -> nn.Linear:
        """Alias property for out_proj."""
        return self.out_proj

    def compute_gate(self) -> mx.array:
        """Compute bounded gate scalar array."""
        return self.gate

    def normalize_slots(self, slots: mx.array) -> mx.array:
        """Normalize latent slots against characteristic RMS scale."""
        return self.slot_norm(slots)

    def get_telemetry(self) -> dict[str, float]:
        """Return structured telemetry dictionary."""
        if self._last_telemetry is not None:
            return dict(self._last_telemetry)
        gate_val = float(self.gate.item())
        alpha_val = float(self.alpha.item())
        return {
            "gate": gate_val,
            "gate_val": gate_val,
            "alpha": alpha_val,
            "raw_alpha": alpha_val,
            "activation_norm": 0.0,
            "norm_backbone_token": 0.0,
            "norm_delta_token": 0.0,
            "norm_injected_token": 0.0,
            "injection_ratio": 0.0,
            "relative_injection_ratio": 0.0,
        }

    def __call__(
        self,
        h: mx.array,
        slots: mx.array,
        return_telemetry: bool = False,
    ) -> mx.array | tuple[mx.array, Optional[dict[str, float]]]:
        """Inject latent slots into contextual hidden states.

        Fast path: if alpha == 0.0 or gate == 0.0, returns h directly for bit-exact identity.
        """
        is_zero_alpha = (self.alpha.size == 1 and float(self.alpha.item()) == 0.0) or (
            self.alpha.size > 1 and bool(mx.all(self.alpha == 0.0).item())
        )
        gate = self.gate
        is_zero_gate = (gate.size == 1 and float(gate.item()) == 0.0) or (
            gate.size > 1 and bool(mx.all(gate == 0.0).item())
        )

        was_2d = False
        if h.ndim == 2:
            h = h[None, :, :]
            was_2d = True

        if (not getattr(self, "training", False)) and (is_zero_alpha or is_zero_gate or slots is None or slots.shape[1] == 0):
            norm_h = float((mx.linalg.norm(h) / math.sqrt(h.size)).item()) if h.size > 0 else 0.0
            self._last_telemetry = {
                "gate": float(gate.item()),
                "gate_val": float(gate.item()),
                "alpha": float(self.alpha.item()),
                "raw_alpha": float(self.alpha.item()),
                "activation_norm": norm_h,
                "norm_backbone_token": norm_h,
                "norm_delta_token": 0.0,
                "norm_injected_token": 0.0,
                "injection_ratio": 0.0,
                "relative_injection_ratio": 0.0,
            }
            res = h[0] if was_2d else h
            return (res, self._last_telemetry) if return_telemetry else res

        if slots is None or slots.shape[1] == 0:
            norm_h = float((mx.linalg.norm(h) / math.sqrt(h.size)).item()) if h.size > 0 else 0.0
            self._last_telemetry = {
                "gate": float(gate.item()),
                "gate_val": float(gate.item()),
                "alpha": float(self.alpha.item()),
                "raw_alpha": float(self.alpha.item()),
                "activation_norm": norm_h,
                "norm_backbone_token": norm_h,
                "norm_delta_token": 0.0,
                "norm_injected_token": 0.0,
                "injection_ratio": 0.0,
                "relative_injection_ratio": 0.0,
            }
            res = h[0] if was_2d else h
            return (res, self._last_telemetry) if return_telemetry else res

        if slots.ndim == 2:
            slots = slots[None, :, :]

        B, L, D = h.shape
        if slots.shape[0] == 1 and B > 1:
            slots = mx.broadcast_to(slots, (B, slots.shape[1], slots.shape[2]))
        _, M, _ = slots.shape

        s_norm = self.slot_norm(slots).astype(h.dtype)

        q = self.q_proj(h).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(s_norm).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(s_norm).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        delta_h = self.out_proj(attn.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim))

        if getattr(self, "training", False):
            h_out = h.astype(mx.float32) + (gate * delta_h)
        else:
            h_out = h + (gate * delta_h).astype(h.dtype)

        norm_h = float((mx.linalg.norm(h) / math.sqrt(h.size)).item())
        norm_delta = float((mx.linalg.norm(delta_h) / math.sqrt(delta_h.size)).item())
        gate_val = float(gate.item())
        norm_inj = abs(gate_val) * norm_delta
        ratio = norm_inj / (norm_h + 1e-8)
        self._last_telemetry = {
            "gate": gate_val,
            "gate_val": gate_val,
            "alpha": float(self.alpha.item()),
            "raw_alpha": float(self.alpha.item()),
            "activation_norm": norm_h,
            "norm_backbone_token": norm_h,
            "norm_delta_token": norm_delta,
            "norm_injected_token": norm_inj,
            "injection_ratio": ratio,
            "relative_injection_ratio": ratio,
        }

        res = h_out[0] if was_2d else h_out
        return (res, self._last_telemetry) if return_telemetry else res


class GemmaCausalPrefixDecoder(nn.Module):
    """Option A Causal Prefix Decoder supporting both legacy soft-prefix and safe cross-attention injection."""

    def __init__(
        self,
        backbone: Any,
        prefix_dim: int = 2048,
        hidden_dim: int = 2048,
        project_prefix: bool = False,
        eos_token_ids: Optional[Sequence[int]] = None,
        adapter: Optional[Any] = None,
        conditioning_mode: str = "cross_attention",
        gamma_max: float = 0.5,
        num_heads: int = 8,
        head_dim: int = 256,
        num_injection_heads: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

        valid_modes = {"cross_attention", "safe_injection", "prefix"}
        if conditioning_mode not in valid_modes:
            raise ValueError(f"Unknown conditioning_mode '{conditioning_mode}'. Expected one of {valid_modes}.")
        self.conditioning_mode = "cross_attention" if conditioning_mode in ("cross_attention", "safe_injection") else "prefix"

        # Dynamically adapt hidden dimensions if backbone manifest indicates 3840 (Gemma 4)
        manifest = getattr(backbone, "manifest", None)
        manifest_dim = getattr(manifest, "hidden_dimension", None) if manifest is not None else None
        if manifest_dim is not None and prefix_dim == 2048 and hidden_dim == 2048 and manifest_dim != 2048:
            prefix_dim = manifest_dim
            hidden_dim = manifest_dim

        self.prefix_dim = prefix_dim
        self.hidden_dim = hidden_dim
        self.project_prefix = project_prefix

        if num_injection_heads is not None:
            num_heads = num_injection_heads
        elif hidden_dim == 3840:
            num_heads = 16

        self.safe_injection = GatedCrossAttentionInjection(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gamma_max=gamma_max,
        )
        self.injection = self.safe_injection

        if eos_token_ids is not None:
            self.eos_token_ids = set(eos_token_ids)
            if self.is_gemma4_architecture() and 107 in self.eos_token_ids:
                self.eos_token_ids.discard(107)
                self.eos_token_ids.add(106)
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
        if manifest is not None and ("gemma-4" in getattr(manifest, "model_id", "") or "12b" in getattr(manifest, "model_id", "").lower()):
            return True
        tok = getattr(self.backbone, "tokenizer", None)
        if tok is not None:
            from prlr.domain.prompt_format import is_gemma4_tokenizer
            if is_gemma4_tokenizer(tok):
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

        if self.conditioning_mode == "cross_attention":
            if self.is_gemma4_architecture():
                prompt_embeds = inner.embed_tokens(prompt_ids)
                if T_len > 1:
                    target_inputs = target_ids[:, :-1]
                    target_embeds = inner.embed_tokens(target_inputs)
                    all_embeds = mx.concatenate([prompt_embeds, target_embeds], axis=1)
                else:
                    all_embeds = prompt_embeds

                h = inner(inputs=None, input_embeddings=all_embeds)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
                start_idx = P - 1
                end_idx = start_idx + T_len
                h_target = h[:, start_idx:end_idx, :]
                target_logits = self.decode_lm_head(h_target)
            else:
                prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
                if T_len > 1:
                    target_inputs = target_ids[:, :-1]
                    target_embeds = inner.embed_tokens(target_inputs) * (hidden_size ** 0.5)
                    all_embeds = mx.concatenate([prompt_embeds, target_embeds], axis=1)
                else:
                    all_embeds = prompt_embeds

                mask = create_attention_mask(all_embeds, cache=None)
                h = all_embeds
                for layer in inner.layers:
                    h = layer(h, mask=mask, cache=None)
                h = inner.norm(h)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
                all_logits = inner.embed_tokens.as_linear(h)
                start_idx = P - 1
                end_idx = start_idx + T_len
                target_logits = all_logits[:, start_idx:end_idx, :]
        else:
            # Legacy soft prefix mode
            if self.is_gemma4_architecture():
                prompt_embeds = inner.embed_tokens(prompt_ids)
                soft_prefix = (self.prepare_prefix(prefix_latents) * (hidden_size ** -0.5)).astype(prompt_embeds.dtype)
                if T_len > 1:
                    target_inputs = target_ids[:, :-1]
                    target_embeds = inner.embed_tokens(target_inputs)
                    all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
                else:
                    all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

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

        if self.conditioning_mode == "cross_attention":
            if self.is_gemma4_architecture():
                prompt_embeds = inner.embed_tokens(prompt_ids)
                h = inner(inputs=None, input_embeddings=prompt_embeds)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
                h_last = h[:, -1:, :]
                return self.decode_lm_head(h_last)
            else:
                prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
                mask = create_attention_mask(prompt_embeds, cache=None)
                h = prompt_embeds
                for layer in inner.layers:
                    h = layer(h, mask=mask, cache=None)
                h = inner.norm(h)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
                return inner.embed_tokens.as_linear(h[:, -1:, :])
        else:
            # Legacy soft prefix mode
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
        if self.is_gemma4_architecture() and 107 in stop_tokens:
            stop_tokens = set(stop_tokens)
            stop_tokens.discard(107)
            stop_tokens.add(106)
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

        if self.conditioning_mode == "cross_attention":
            if self.is_gemma4_architecture():
                prompt_embeds = inner.embed_tokens(prompt_ids)
                cache = make_prompt_cache(self.backbone.model)

                # Prefill through frozen backbone
                h = inner(inputs=None, cache=cache, input_embeddings=prompt_embeds)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
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
                    if prefix_latents is not None and prefix_latents.shape[1] > 0:
                        h_step = self.safe_injection(h_step, prefix_latents)
                    step_logits = self.decode_lm_head(h_step)
                    if temperature <= 1e-5:
                        next_tok = mx.argmax(step_logits, axis=-1)
                    else:
                        next_tok = mx.random.categorical(step_logits / temperature)

                    generated_tokens.append(next_tok)

                    if B == 1 and next_tok[0, 0].item() in stop_tokens:
                        break

                return mx.concatenate(generated_tokens, axis=1)
            else:
                # Gemma 2 / compact model fallback path in cross-attention mode
                prompt_embeds = inner.embed_tokens(prompt_ids) * (hidden_size ** 0.5)
                cache = make_prompt_cache(inner)
                mask = create_attention_mask(prompt_embeds, cache[0])

                h = prompt_embeds
                for layer, c in zip(inner.layers, cache):
                    h = layer(h, mask=mask, cache=c)
                h = inner.norm(h)
                if prefix_latents is not None and prefix_latents.shape[1] > 0:
                    h = self.safe_injection(h, prefix_latents)
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
                    if prefix_latents is not None and prefix_latents.shape[1] > 0:
                        h = self.safe_injection(h, prefix_latents)
                    logits = inner.embed_tokens.as_linear(h)

                    if temperature <= 1e-5:
                        next_tok = mx.argmax(logits, axis=-1)
                    else:
                        next_tok = mx.random.categorical(logits / temperature)

                    generated_tokens.append(next_tok)

                    if B == 1 and next_tok[0, 0].item() in stop_tokens:
                        break

                return mx.concatenate(generated_tokens, axis=1)

        # Legacy soft prefix mode
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


__all__ = ["GemmaCausalPrefixDecoder", "GatedCrossAttentionInjection"]
