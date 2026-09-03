"""Evaluation packages for PRLR."""

from prlr.eval.ablation import (
    AblationConditionSummary,
    AblationSpec,
    AblationSuiteReport,
    GemmaAblationHarness,
    InstanceEvaluationRecord,
    compute_bootstrap_ci_95,
)
from prlr.eval.microbench import (
    RULE_4_DISCLAIMER,
    KernelBenchmarkResult,
    KernelMicrobenchConfig,
    KernelMicrobenchmarkRunner,
    compute_kernel_bytes,
    compute_kernel_flops,
)
from prlr.eval.semantic_bench import (
    DISCLAIMER_SEMANTIC,
    InstancePredictionRecord,
    SemanticBenchmarkRunner,
    StageLatencyTelemetry,
    compute_bootstrap_ci_bca,
)

__all__ = [
    "AblationSpec",
    "InstanceEvaluationRecord",
    "AblationConditionSummary",
    "AblationSuiteReport",
    "compute_bootstrap_ci_95",
    "GemmaAblationHarness",
    "KernelMicrobenchConfig",
    "KernelBenchmarkResult",
    "KernelMicrobenchmarkRunner",
    "compute_kernel_flops",
    "compute_kernel_bytes",
    "RULE_4_DISCLAIMER",
    "SemanticBenchmarkRunner",
    "InstancePredictionRecord",
    "StageLatencyTelemetry",
    "compute_bootstrap_ci_bca",
    "DISCLAIMER_SEMANTIC",
]
