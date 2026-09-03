"""Backward compatibility shim for parallel_latent_reasoner.engine.

Canonical implementation has moved to prlr.kernel.engine.
"""

from __future__ import annotations

from prlr.kernel.engine import (
    DeliberationResult,
    MLXParallelLatentEngine,
)

__all__ = [
    "DeliberationResult",
    "MLXParallelLatentEngine",
]
