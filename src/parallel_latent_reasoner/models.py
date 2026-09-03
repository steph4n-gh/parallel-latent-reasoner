"""Backward compatibility shim for parallel_latent_reasoner.models.

Canonical implementations have moved:
- Recurrent modules & kernels: prlr.kernel.recurrent_core
- Compact testbed model: prlr.compact.scratch_model
"""

from __future__ import annotations

from prlr.compact.scratch_model import (
    CompactScratchModel,
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXPreludeProjection,
)
from prlr.kernel.recurrent_core import (
    MLXAdaRMSNorm,
    MLXAttention,
    MLXGemmaAttention,
    MLXGemmaMLP,
    MLXGemmaMoE,
    MLXMoE,
    MLXRecurrentBlock,
    MLXRecurrentGemmaBlock,
    MLXRMSNorm,
    sinusoidal_step_embedding,
)

__all__ = [
    "sinusoidal_step_embedding",
    "MLXRMSNorm",
    "MLXAdaRMSNorm",
    "MLXAttention",
    "MLXGemmaAttention",
    "MLXGemmaMLP",
    "MLXMoE",
    "MLXGemmaMoE",
    "MLXRecurrentBlock",
    "MLXRecurrentGemmaBlock",
    "MLXPreludeProjection",
    "MLXCodaLMHead",
    "MLXCompactGemmaModel",
    "CompactScratchModel",
]
