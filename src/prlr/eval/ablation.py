"""Controlled Ablation Evaluation Interface.

Re-exports ablation engine classes and routines from prlr.gemma.ablation.
"""

from prlr.gemma.ablation import (
    AblationConditionSummary,
    AblationSpec,
    AblationSuiteReport,
    GemmaAblationHarness,
    InstanceEvaluationRecord,
    compute_bootstrap_ci_95,
)

__all__ = [
    "AblationSpec",
    "InstanceEvaluationRecord",
    "AblationConditionSummary",
    "AblationSuiteReport",
    "compute_bootstrap_ci_95",
    "GemmaAblationHarness",
]
