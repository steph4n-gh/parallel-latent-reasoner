"""Configuration for Pure Model-Agnostic MLX Recurrence Kernel.

Defines RecurrentKernelConfig dataclass for tensor operations on shapes [B, M, D].
Zero dependencies on token vocabularies, SentencePiece, or Hugging Face.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class RecurrentKernelConfig:
    """Configuration for fixed-width recurrent Transformer core operations.

    Attributes:
        dim: State hidden dimension D.
        intermediate_dim: Intermediate feedforward dimension.
        num_heads: Number of attention query heads.
        num_kv_heads: Number of key/value heads for GQA/MQA.
        head_dim: Dimension per attention head (d_k).
        num_memory_slots: Number of working memory slots M.
        deliberation_steps: Number of recurrent unroll steps T (default 8).
        min_steps: Minimum deliberation steps before early exit (default 2).
        max_steps: Hard upper bound on deliberation steps (default 12).
        rezero_alpha: Initial ReZero residual scaling scalar alpha <= 0.05.
        rms_norm_eps: Numerical epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        num_layers: Number of weight-tied recurrent layers per block.
        step_embed_dim: Dimension of sinusoidal step position embedding.
    """

    dim: int = 256
    hidden_dim: int | None = None
    intermediate_dim: int = 512
    num_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 64
    num_memory_slots: int = 16
    num_slots: int | None = None
    deliberation_steps: int = 8
    min_steps: int = 2
    max_steps: int = 12
    rezero_alpha: float = 0.05
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    num_layers: int = 1
    step_embed_dim: int = 64
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecurrentKernelConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


__all__ = ["RecurrentKernelConfig"]
