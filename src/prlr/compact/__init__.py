"""Honest 256D Compact Testbed Model Namespace.

Strictly marked random-init for CI, unit testing, and learnability verification.
Never claims pretrained status or CoT speedups.
"""

from __future__ import annotations

from prlr.compact.benchmark import (
    BenchmarkResult,
    DomainSampleRecord,
    MultiDomainBenchmarkSuite,
    MultiScaleBenchmarkSuite,
    compute_max_ngram_repetition,
    compute_shannon_entropy,
)
from prlr.compact.config import CompactConfig, GemmaLatentConfig
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
    CompactScratchModel,
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXPreludeProjection,
)
from prlr.compact.trainer import (
    PRLRBPTTTrainer,
    TrainerConfig,
    TrainMetrics,
    _compute_bptt_loss,
)

__all__ = [
    "CompactConfig",
    "GemmaLatentConfig",
    "CompactScratchModel",
    "MLXCompactGemmaModel",
    "MLXPreludeProjection",
    "MLXCodaLMHead",
    "PRLRBPTTTrainer",
    "TrainerConfig",
    "TrainMetrics",
    "_compute_bptt_loss",
    "DistillationSample",
    "ProceduralMultiDomainGenerator",
    "PRLRDataset",
    "PRLRDataLoader",
    "split_dataset",
    "check_split_contamination",
    "generate_distillation_dataset",
    "train_prlr_adapter",
    "HybridDeliberationResult",
    "DeliberationPipelineOutput",
    "PRLRPipeline",
    "GemmaDeliberationPipeline",
    "BenchmarkResult",
    "DomainSampleRecord",
    "MultiScaleBenchmarkSuite",
    "MultiDomainBenchmarkSuite",
    "compute_shannon_entropy",
    "compute_max_ngram_repetition",
]
