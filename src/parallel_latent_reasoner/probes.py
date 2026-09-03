"""Backward compatibility shim for parallel_latent_reasoner.probes.

Canonical implementation has moved to prlr.kernel.telemetry.
"""

from __future__ import annotations

from prlr.kernel.telemetry import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
    detect_limit_cycle,
)

__all__ = [
    "compute_effective_rank",
    "compute_slot_cosine_similarity",
    "compute_slot_velocity",
    "analyze_deliberation_trajectory",
    "detect_limit_cycle",
    "TrajectoryAnalysis",
]
