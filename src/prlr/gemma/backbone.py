"""Pretrained Gemma Backbone Integration for Apple Silicon.

Enforces genuine Google weights, SentencePiece tokenization, and contextual hidden state
extraction. Strictly rejects unverified random models and character-modulo fallbacks under Rule 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from prlr.gemma.loader import LoadedModel, load_model
from prlr.manifest import (
    ModelManifest,
    Rule5ViolationError,
    RuleViolationError,
)


class PretrainedGemmaBackbone(nn.Module):
    """Genuine Pretrained Gemma Backbone.

    Requires cryptographically verified ModelManifest and loaded SentencePiece tokenizer.
    """

    def __init__(
        self,
        manifest: ModelManifest | None = None,
        load_weights: bool = True,
        verify_hashes: bool = True,
        allow_random_init: bool = False,
    ):
        super().__init__()
        if manifest is None:
            raise Rule5ViolationError(
                "PretrainedGemmaBackbone requires a valid ModelManifest; received manifest=None."
            )

        if (not manifest.is_pretrained or manifest.random_init) and not allow_random_init:
            raise Rule5ViolationError(
                f"PretrainedGemmaBackbone requires a genuine pretrained checkpoint (is_pretrained=True). "
                f"Received manifest '{manifest.model_id}' with is_pretrained={manifest.is_pretrained}, random_init={manifest.random_init}. "
                "Pass allow_random_init=True to explicitly override for testing."
            )

        manifest.validate(allow_gemma_random_init=allow_random_init)
        self.manifest = manifest
        self.model = None
        self.tokenizer = None

        if load_weights:
            loaded: LoadedModel = load_model(
                manifest=manifest,
                verify_hashes=verify_hashes,
                allow_gemma_random_init=allow_random_init,
            )
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer

    def encode_prompt_context(
        self,
        prompt: str | mx.array,
    ) -> tuple[mx.array, int]:
        """Encode prompt into token IDs using official tokenizer.

        Strictly prohibits character-modulo fallbacks (ord(c) % vocab) per Rule 5.
        """
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError(
                    "Cannot encode string prompt without loaded official tokenizer. "
                    "Character-modulo tokenization (ord(c) % vocab) is strictly prohibited under Rule 5."
                )
            # Tokenize using official SentencePiece / AutoTokenizer
            if hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            elif hasattr(self.tokenizer, "encode_as_ids"):
                tokens = self.tokenizer.encode_as_ids(prompt)
            else:
                tokens = list(self.tokenizer(prompt))

            token_ids = mx.array([tokens], dtype=mx.int32)
            return token_ids, len(tokens)
        elif isinstance(prompt, mx.array):
            if prompt.ndim == 1:
                prompt = prompt[None, :]
            return prompt, prompt.shape[1]
        else:
            raise TypeError(
                f"Expected prompt to be str or mx.array, got {type(prompt)}"
            )

    def extract_contextual_hiddens(
        self,
        input_ids: mx.array,
        layer_idx: int = 18,
    ) -> mx.array:
        """Extract contextual hidden representations from pretrained model.

        Args:
            input_ids: Token ID tensor of shape (B, L) or (L,).
            layer_idx: Extraction layer index (1..18, default 18 for full model).
                       If 18 or None, returns final normalized hidden states.
                       If < 18, unrolls up to layer_idx and normalizes with model norm.

        Returns:
            Contextual hidden representations of shape (B, L, 2048).
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot extract contextual hidden states without loaded pretrained model weights."
            )

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        inner = getattr(self.model, "model", self.model)

        if (
            hasattr(inner, "layers")
            and layer_idx is not None
            and layer_idx < len(inner.layers)
        ):
            from mlx_lm.models.base import create_attention_mask

            hidden_size = getattr(inner.args, "hidden_size", 2048)
            h = inner.embed_tokens(input_ids) * (hidden_size ** 0.5)
            mask = create_attention_mask(h, None)
            for layer in inner.layers[:layer_idx]:
                h = layer(h, mask=mask, cache=None)
            if hasattr(inner, "norm"):
                h = inner.norm(h)
            return h

        # Full model extraction (layer 18 or final normalized)
        if callable(inner):
            return inner(input_ids)
        elif hasattr(self.model, "__call__"):
            return self.model(input_ids)
        else:
            raise RuntimeError("Model does not support forward evaluation.")

    def health_check(self) -> dict[str, Any]:
        """Verify baseline health check on Apple Silicon Metal GPU.

        Performs:
        1. Manifest and disk integrity validation.
        2. Apple Silicon Metal GPU device confirmation.
        3. Official SentencePiece vocabulary & special token alignment.
        4. Contextual hidden extraction shape (1, L, 2048) and non-NaN verification.
        5. Greedy semantic generation baseline (e.g. 'Paris' for France capital prompt).

        Returns:
            Diagnostic dictionary with health metrics.
        """
        import time

        diagnostics: dict[str, Any] = {}

        # 1. Manifest verification
        if self.manifest is None:
            raise RuntimeError("Health check requires valid manifest.")
        self.manifest.validate(check_disk=True)
        diagnostics["manifest_validated"] = True
        diagnostics["model_id"] = self.manifest.model_id

        # 2. Metal GPU verification
        device = mx.default_device()
        diagnostics["device"] = str(device)
        diagnostics["is_metal_gpu"] = (device.type == mx.gpu)

        # 3. SentencePiece verification
        if self.tokenizer is None:
            raise RuntimeError("Health check requires loaded official tokenizer.")

        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None and hasattr(self.tokenizer, "get_piece_size"):
            vocab_size = self.tokenizer.get_piece_size()
        diagnostics["vocab_size"] = vocab_size

        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None and hasattr(self.tokenizer, "bos_id"):
            bos_id = self.tokenizer.bos_id()
        diagnostics["bos_id"] = bos_id

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(self.tokenizer, "eos_id"):
            eos_id = self.tokenizer.eos_id()
        diagnostics["eos_id"] = eos_id

        # 4. Contextual hidden extraction
        prompt = "The capital of France is"
        input_ids, _ = self.encode_prompt_context(prompt)
        t0 = time.perf_counter()
        hiddens_18 = self.extract_contextual_hiddens(input_ids, layer_idx=18)
        mx.eval(hiddens_18)
        extract_time_ms = (time.perf_counter() - t0) * 1000.0

        diagnostics["hidden_shape"] = list(hiddens_18.shape)
        diagnostics["hidden_dtype"] = str(hiddens_18.dtype)
        diagnostics["extract_time_ms"] = extract_time_ms
        diagnostics["has_nan"] = bool(mx.isnan(hiddens_18).any().item())
        diagnostics["has_inf"] = bool(mx.isinf(hiddens_18).any().item())

        if diagnostics["has_nan"] or diagnostics["has_inf"]:
            raise RuntimeError("Contextual hidden representations contain NaN or Inf!")

        if hiddens_18.shape[0] != 1 or hiddens_18.shape[2] != 2048:
            raise RuntimeError(f"Unexpected hidden shape: {hiddens_18.shape}, expected (1, L, 2048)")

        # Intermediate layer extraction (layer 12)
        hiddens_12 = self.extract_contextual_hiddens(input_ids, layer_idx=12)
        mx.eval(hiddens_12)
        diagnostics["intermediate_layer_12_shape"] = list(hiddens_12.shape)

        # 5. Greedy semantic generation baseline
        import mlx_lm
        t0 = time.perf_counter()
        gen_text = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=10,
            verbose=False,
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0
        diagnostics["generation_output"] = gen_text
        diagnostics["generation_time_ms"] = gen_time_ms
        diagnostics["semantic_passed"] = "Paris" in gen_text

        if not diagnostics["semantic_passed"]:
            raise RuntimeError(f"Baseline semantic completion failed to emit 'Paris': {gen_text}")

        diagnostics["status"] = "HEALTHY"
        return diagnostics

    def __call__(
        self,
        prompt: str | mx.array,
    ) -> tuple[mx.array, int]:
        return self.encode_prompt_context(prompt)


__all__ = ["PretrainedGemmaBackbone"]

