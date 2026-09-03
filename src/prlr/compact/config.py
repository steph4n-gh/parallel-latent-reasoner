"""Configuration for Compact 256D Testbed Model.

Strictly marked random-init for CI, unit testing, and learnability verification.
Never claims pretrained status or CoT speedups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class GemmaLatentConfig:
    """Configuration dataclass for Parallel Recurrent Latent Deliberation."""

    dim: int = 2048
    hidden_dim: int | None = None
    intermediate_dim: int = 8192
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 256
    vocab_size: int = 256000
    num_memory_slots: int = 16
    num_slots: int | None = None
    deliberation_steps: int = 8
    min_steps: int = 2
    max_steps: int = 12
    rezero_alpha: float = 0.05
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    num_layers: int = 1
    tie_word_embeddings: bool = True
    step_embed_dim: int = 128
    final_logit_softcapping: float | None = 30.0
    max_position_embeddings: int = 8192
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_dim: int | None = None
    moe_intermediate_size: int | None = None
    enable_moe_block: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim is not None:
            object.__setattr__(self, "dim", self.hidden_dim)
        else:
            object.__setattr__(self, "hidden_dim", self.dim)

        if self.num_slots is not None:
            object.__setattr__(self, "num_memory_slots", self.num_slots)
        else:
            object.__setattr__(self, "num_slots", self.num_memory_slots)

        if self.moe_intermediate_size is not None and self.moe_intermediate_dim is None:
            object.__setattr__(self, "moe_intermediate_dim", self.moe_intermediate_size)
        elif self.moe_intermediate_dim is not None and self.moe_intermediate_size is None:
            object.__setattr__(self, "moe_intermediate_size", self.moe_intermediate_dim)

        if self.num_experts is not None and self.num_experts > 0:
            object.__setattr__(self, "enable_moe_block", True)

        if self.head_dim <= 0:
            if self.num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            object.__setattr__(self, "head_dim", self.dim // self.num_heads)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_memory_slots <= 0:
            raise ValueError(f"num_memory_slots must be positive, got {self.num_memory_slots}")
        if self.rezero_alpha < 0.0:
            raise ValueError(f"rezero_alpha must be non-negative, got {self.rezero_alpha}")
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {self.min_steps}")
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})"
            )

        if self.deliberation_steps < self.min_steps or self.deliberation_steps > self.max_steps:
            object.__setattr__(
                self,
                "deliberation_steps",
                min(max(self.deliberation_steps, self.min_steps), self.max_steps),
            )

        if self.num_experts is not None:
            if self.num_experts <= 0:
                raise ValueError(f"num_experts must be positive, got {self.num_experts}")
            if self.top_k_experts is not None and (
                self.top_k_experts <= 0 or self.top_k_experts > self.num_experts
            ):
                raise ValueError(
                    f"top_k_experts ({self.top_k_experts}) must be in range [1, {self.num_experts}]"
                )
            if self.moe_intermediate_dim is not None and self.moe_intermediate_dim <= 0:
                raise ValueError(
                    f"moe_intermediate_dim must be positive, got {self.moe_intermediate_dim}"
                )

    @classmethod
    def compact_test(cls, **kwargs: Any) -> GemmaLatentConfig:
        """Lightweight resident tier for unit testing, CI, and rapid validation."""
        defaults: dict[str, Any] = {
            "dim": 256,
            "intermediate_dim": 512,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 64,
            "vocab_size": 1000,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "num_layers": 1,
            "tie_word_embeddings": True,
            "step_embed_dim": 64,
            "final_logit_softcapping": 30.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_2b(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 2048,
            "intermediate_dim": 8192,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 256,
            "vocab_size": 256000,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "num_layers": 1,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_9b(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 3584,
            "intermediate_dim": 14336,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "vocab_size": 256000,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "num_layers": 1,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_12b(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 3840,
            "intermediate_dim": 16384,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "vocab_size": 256000,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "num_layers": 1,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_12b_q4(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 3840,
            "intermediate_dim": 15360,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "vocab_size": 262144,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_layers": 48,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
            "max_position_embeddings": 262144,
            "enable_moe_block": False,
            "num_experts": None,
            "top_k_experts": None,
            "moe_intermediate_dim": None,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_4_12b_q4(cls, **kwargs: Any) -> GemmaLatentConfig:
        return cls.gemma_12b_q4(**kwargs)

    @classmethod
    def gemma_26b_a4b(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 2816,
            "intermediate_dim": 2112,
            "moe_intermediate_dim": 704,
            "moe_intermediate_size": 704,
            "num_experts": 128,
            "top_k_experts": 8,
            "enable_moe_block": True,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "vocab_size": 262144,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_layers": 30,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
            "max_position_embeddings": 262144,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gemma_4_26b_a4b(cls, **kwargs: Any) -> GemmaLatentConfig:
        return cls.gemma_26b_a4b(**kwargs)

    @classmethod
    def gemma_e4b(cls, **kwargs: Any) -> GemmaLatentConfig:
        defaults: dict[str, Any] = {
            "dim": 3072,
            "intermediate_dim": 12288,
            "num_heads": 12,
            "num_kv_heads": 4,
            "head_dim": 256,
            "vocab_size": 256000,
            "num_memory_slots": 16,
            "deliberation_steps": 8,
            "min_steps": 2,
            "max_steps": 12,
            "rezero_alpha": 0.05,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "num_layers": 1,
            "tie_word_embeddings": True,
            "step_embed_dim": 128,
            "final_logit_softcapping": 30.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GemmaLatentConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> GemmaLatentConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class CompactConfig(GemmaLatentConfig):
    """Configuration for honest 256D compact scratch model."""

    dim: int = 256
    intermediate_dim: int = 512
    num_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 1000
    num_memory_slots: int = 16
    deliberation_steps: int = 8
    min_steps: int = 2
    max_steps: int = 12
    rezero_alpha: float = 0.05
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    num_layers: int = 1
    tie_word_embeddings: bool = True
    step_embed_dim: int = 64
    final_logit_softcapping: float | None = 30.0


__all__ = [
    "GemmaLatentConfig",
    "CompactConfig",
]
