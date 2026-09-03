"""Parallel Latent Reasoner (PRLR) - Backward Compatibility Facade.

Canonical implementations have moved to the `prlr` namespace:
- `prlr.kernel`: Pure recurrence kernel & telemetry (model-agnostic)
- `prlr.compact`: Honest 256D testbed model (CI & synthetic tests)
- `prlr.gemma`: Genuine pretrained Gemma vertical lane
- `prlr.manifest`: Cryptographic ModelManifest and hash verifier
"""

from __future__ import annotations

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    EvaluationResult,
    VerifierType,
    get_domain_summary,
    get_test_case_by_id,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)
from prlr.compact.config import GemmaLatentConfig
from prlr.compact.dataset import (
    DistillationSample,
    PRLRDataLoader,
    PRLRDataset,
    ProceduralMultiDomainGenerator,
    check_split_contamination,
    generate_distillation_dataset,
    split_dataset,
    train_prlr_adapter,
)
from prlr.compact.pipeline import (
    DeliberationPipelineOutput,
    GemmaDeliberationPipeline,
    HybridDeliberationResult,
    PRLRPipeline,
)
from prlr.compact.scratch_model import (
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXPreludeProjection,
)
from prlr.compact.trainer import (
    PRLRBPTTTrainer,
    TrainerConfig,
    TrainMetrics,
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
from prlr.kernel.recurrent_core import (
    MLXAdaRMSNorm,
    MLXGemmaAttention,
    MLXGemmaMLP,
    MLXRecurrentGemmaBlock,
    MLXRMSNorm,
    sinusoidal_step_embedding,
)
from prlr.kernel.telemetry import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
    detect_limit_cycle,
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
    "CognitiveTestCase",
    "DomainType",
    "EvaluationResult",
    "VerifierType",
    "get_domain_summary",
    "get_test_case_by_id",
    "load_cognitive_benchmark_suite",
    "verify_test_case_result",
    "__version__",
]
