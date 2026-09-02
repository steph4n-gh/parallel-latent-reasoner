"""Parallel Latent Reasoner (PRLR) - Standalone MLX Package.

Non-autoregressive continuous latent deliberation on Apple Silicon with the
3-Signal Dynamic Consensus E-Gate.
"""

from __future__ import annotations

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import (
    DynamicConsensusEGate,
    DynamicDeliberationGate,
    GateDecision,
    GateTelemetry,
)
from parallel_latent_reasoner.engine import (
    DeliberationResult,
    MLXParallelLatentEngine,
)
from parallel_latent_reasoner.models import (
    MLXAdaRMSNorm,
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXGemmaAttention,
    MLXGemmaMLP,
    MLXPreludeProjection,
    MLXRecurrentGemmaBlock,
    MLXRMSNorm,
    sinusoidal_step_embedding,
)
from parallel_latent_reasoner.pipeline import (
    DeliberationPipelineOutput,
    GemmaDeliberationPipeline,
    HybridDeliberationResult,
    PRLRPipeline,
)
from parallel_latent_reasoner.probes import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
    detect_limit_cycle,
)
from parallel_latent_reasoner.dataset import (
    DistillationSample,
    PRLRDataLoader,
    PRLRDataset,
    ProceduralMultiDomainGenerator,
    check_split_contamination,
    generate_distillation_dataset,
    split_dataset,
    train_prlr_adapter,
)
from parallel_latent_reasoner.trainer import (
    PRLRBPTTTrainer,
    TrainerConfig,
    TrainMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "GemmaLatentConfig",
    "sinusoidal_step_embedding",
    "MLXRMSNorm",
    "MLXAdaRMSNorm",
    "MLXGemmaAttention",
    "MLXGemmaMLP",
    "MLXRecurrentGemmaBlock",
    "MLXPreludeProjection",
    "MLXCodaLMHead",
    "MLXCompactGemmaModel",
    "DeliberationResult",
    "MLXParallelLatentEngine",
    "compute_effective_rank",
    "compute_slot_cosine_similarity",
    "compute_slot_velocity",
    "analyze_deliberation_trajectory",
    "detect_limit_cycle",
    "TrajectoryAnalysis",
    "DynamicDeliberationGate",
    "DynamicConsensusEGate",
    "GateTelemetry",
    "GateDecision",
    "PRLRPipeline",
    "GemmaDeliberationPipeline",
    "HybridDeliberationResult",
    "DeliberationPipelineOutput",
    "TrainerConfig",
    "TrainMetrics",
    "PRLRBPTTTrainer",
    "DistillationSample",
    "ProceduralMultiDomainGenerator",
    "PRLRDataset",
    "PRLRDataLoader",
    "split_dataset",
    "check_split_contamination",
    "generate_distillation_dataset",
    "train_prlr_adapter",
    "__version__",
]
