"""Backward compatibility shim for parallel_latent_reasoner.pipeline.

Canonical implementation has moved to prlr.compact.pipeline.
"""

from __future__ import annotations

from prlr.compact.pipeline import (
    DeliberationPipelineOutput,
    GemmaDeliberationPipeline,
    HybridDeliberationResult,
    PRLRPipeline,
    _find_adapter_checkpoint,
)

__all__ = [
    "HybridDeliberationResult",
    "DeliberationPipelineOutput",
    "PRLRPipeline",
    "GemmaDeliberationPipeline",
    "_find_adapter_checkpoint",
]
