"""Backward compatibility shim for parallel_latent_reasoner.config.

Canonical implementations have moved to:
- prlr.compact.config
- prlr.gemma.config
"""

from __future__ import annotations

from prlr.compact.config import CompactConfig, GemmaLatentConfig

__all__ = [
    "GemmaLatentConfig",
    "CompactConfig",
]
