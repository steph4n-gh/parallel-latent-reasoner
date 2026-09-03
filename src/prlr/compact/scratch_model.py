"""MLX Compact Testbed Scratch Model.

Implements Prelude Projection, Coda LM Head, and the MLXCompactGemmaModel / CompactScratchModel
for CI, unit tests, and procedural learnability validation.
Strictly labeled 'prlr-compact-testbed' with is_pretrained=False.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from prlr.compact.config import CompactConfig, GemmaLatentConfig
from prlr.kernel.engine import DeliberationResult, MLXParallelLatentEngine
from prlr.kernel.recurrent_core import MLXRMSNorm
from prlr.manifest import ModelManifest


class MLXPreludeProjection(nn.Module):
    """Prelude Projection module initializing M continuous latent working memory slots."""

    def __init__(
        self,
        config: Any,
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
        """Initialize M memory slots S^(0) and return prompt contextual representations."""
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

        pooled = mx.mean(prompt_hiddens, axis=1, keepdims=True)  # [B, 1, D]
        context_mod = self.context_proj(pooled)  # [B, 1, D]

        base_slots = mx.broadcast_to(self.slot_embeddings, (B, self.num_slots, D))
        slots = base_slots + context_mod
        slots = self.norm(slots)

        return slots, prompt_hiddens


class MLXCodaLMHead(nn.Module):
    """Discrete Coda / LM Head decoding latent memory into vocabulary logits."""

    def __init__(
        self,
        config: Any,
        embed_tokens: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        self.soft_cap = config.final_logit_softcapping

        self.final_norm = MLXRMSNorm(self.dim, eps=config.rms_norm_eps)
        if not getattr(config, "tie_word_embeddings", True) or embed_tokens is None:
            self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
            self.embed_tokens = None
        else:
            self.lm_head = None
            self.embed_tokens = embed_tokens

        self.readout_proj = nn.Linear(self.dim, self.dim, bias=False)

    def pool_readout(self, memory_slots: mx.array) -> mx.array:
        normed = self.final_norm(memory_slots)
        pooled = mx.mean(normed, axis=1)  # [B, D]
        return self.readout_proj(pooled)

    def project_logits(self, hidden_states: mx.array) -> mx.array:
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
        if pool:
            h = self.pool_readout(memory_slots)
            return self.project_logits(h)
        else:
            normed = self.final_norm(memory_slots)
            return self.project_logits(normed)


class MLXCompactGemmaModel(nn.Module):
    """Compact testbed model for CI and unit tests."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.name = "prlr-compact-testbed"
        self.model_id = "prlr-compact-testbed"
        self.is_pretrained = False
        self.manifest = ModelManifest.compact_test()

        # Shared token embedding table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        # 1. Prelude Projection
        self.prelude = MLXPreludeProjection(config, embed_tokens=self.embed_tokens)

        # 2. Parallel Latent Deliberation Engine
        self.engine = MLXParallelLatentEngine(config)

        # 3. Discrete Coda / LM Head
        self.coda = MLXCodaLMHead(config, embed_tokens=self.embed_tokens)

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> MLXCompactGemmaModel:
        cfg = GemmaLatentConfig(
            dim=manifest.hidden_dimension,
            intermediate_dim=manifest.intermediate_dimension,
            num_heads=manifest.num_heads,
            num_kv_heads=manifest.num_kv_heads,
            head_dim=manifest.head_dimension,
            vocab_size=manifest.vocabulary_size,
            num_memory_slots=16,
            num_layers=manifest.num_layers,
        )
        model = cls(cfg)
        model.manifest = manifest
        return model

    def deliberate(
        self,
        prompt: mx.array,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> DeliberationResult:
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
        delib_res = self.deliberate(input_ids, steps=steps)
        readout = self.coda.pool_readout(delib_res.final_states)

        generated: list[mx.array] = []
        curr_hidden = readout

        for _ in range(max_new_tokens):
            logits = self.coda.project_logits(curr_hidden)
            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                next_tok = mx.random.categorical(logits / temperature)[:, None]
            generated.append(next_tok)

            tok_embed = self.prelude.embed_prompt(next_tok)[:, 0, :]
            curr_hidden = self.coda.final_norm(curr_hidden + 0.1 * tok_embed)

        return mx.concatenate(generated, axis=-1)

    def get_trainable_parameters(self) -> dict[str, mx.array]:
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


class CompactScratchModel(MLXCompactGemmaModel):
    """Explicitly labeled 256D scratch testbed model for CI and learnability tests."""

    def __init__(self, config: Any | None = None):
        if config is None:
            config = CompactConfig()
        super().__init__(config)
        self.name = "prlr-compact-testbed"
        self.model_id = "prlr-compact-testbed"
        self.is_pretrained = False
        self.manifest = ModelManifest.compact_test()


__all__ = [
    "MLXPreludeProjection",
    "MLXCodaLMHead",
    "MLXCompactGemmaModel",
    "CompactScratchModel",
]
