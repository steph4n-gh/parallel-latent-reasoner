"""Pure Model-Agnostic Recurrence Kernel Namespace.

Provides fixed-width tensor operations on shapes [B, M, D] with zero language model
or tokenizer dependencies.
"""

from __future__ import annotations

from prlr.kernel.config import RecurrentKernelConfig
from prlr.kernel.recurrent_core import (
    MLXAdaRMSNorm,
    MLXAttention,
    MLXMLP,
    MLXMoE,
    MLXRecurrentBlock,
    MLXRMSNorm,
    sinusoidal_step_embedding,
)
from prlr.kernel.engine import (
    DeliberationResult,
    MLXParallelLatentEngine,
)
from prlr.kernel.gates import (
    DynamicConsensusEGate,
    DynamicDeliberationGate,
    GateDecision,
    GateTelemetry,
)
from prlr.kernel.telemetry import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
    detect_limit_cycle,
)

__all__ = [
    "RecurrentKernelConfig",
    "sinusoidal_step_embedding",
    "MLXRMSNorm",
    "MLXAdaRMSNorm",
    "MLXAttention",
    "MLXMLP",
    "MLXMoE",
    "MLXRecurrentBlock",
    "DeliberationResult",
    "MLXParallelLatentEngine",
    "GateTelemetry",
    "GateDecision",
    "DynamicDeliberationGate",
    "DynamicConsensusEGate",
    "compute_effective_rank",
    "compute_slot_cosine_similarity",
    "compute_slot_velocity",
    "analyze_deliberation_trajectory",
    "detect_limit_cycle",
    "TrajectoryAnalysis",
]
