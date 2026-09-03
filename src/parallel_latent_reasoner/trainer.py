"""Backward compatibility shim for parallel_latent_reasoner.trainer.

Canonical implementation has moved to prlr.compact.trainer.
"""

from __future__ import annotations

from prlr.compact.trainer import (
    PRLRBPTTTrainer,
    TrainerConfig,
    TrainMetrics,
    _compute_bptt_loss,
)

__all__ = [
    "TrainerConfig",
    "TrainMetrics",
    "PRLRBPTTTrainer",
    "_compute_bptt_loss",
]
