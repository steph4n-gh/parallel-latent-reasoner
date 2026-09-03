"""Backward compatibility shim for parallel_latent_reasoner.egate.

Canonical implementation has moved to prlr.kernel.gates.
"""

from __future__ import annotations

from prlr.kernel.gates import (
    DynamicConsensusEGate,
    DynamicDeliberationGate,
    GateDecision,
    GateTelemetry,
)

__all__ = [
    "GateTelemetry",
    "GateDecision",
    "DynamicDeliberationGate",
    "DynamicConsensusEGate",
]
