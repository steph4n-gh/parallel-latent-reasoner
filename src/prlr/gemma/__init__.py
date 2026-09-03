"""Genuine Pretrained Gemma Vertical Lane.

Enforces cryptographic verification of official Google weights and tokenizers,
contextual hidden state prompt representations, and causal decoding.
"""

from __future__ import annotations

from prlr.gemma.adapter import (
    AdapterConfig,
    GemmaPreludeAdapter,
    GemmaRecurrentAdapter,
    init_orthogonal_slot_anchors,
)
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.config import GemmaLatentConfig
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.loader import LoadedModel, ManifestHashCache, load_model
from prlr.gemma.ablation import (
    AblationConditionSummary,
    AblationSpec,
    AblationSuiteReport,
    GemmaAblationHarness,
    InstanceEvaluationRecord,
    compute_bootstrap_ci_95,
)
from prlr.gemma.egate import (
    CalibratedGateThresholds,
    EGateCalibrator,
    EGateStepTelemetry,
    GemmaCalibratedEGate,
)
from prlr.gemma.trainer import (
    GemmaPRLRTrainer,
    GemmaTrainerConfig,
    compute_masked_ce_loss,
)
from prlr.manifest import (
    ArchitectureMismatchError,
    IntegrityError,
    ManifestError,
    ModelManifest,
    Rule5ViolationError,
    RuleViolationError,
    verify_model_manifest,
)

__all__ = [
    "ModelManifest",
    "PretrainedGemmaBackbone",
    "GemmaPreludeAdapter",
    "GemmaRecurrentAdapter",
    "init_orthogonal_slot_anchors",
    "AdapterConfig",
    "GemmaCausalPrefixDecoder",
    "load_model",
    "LoadedModel",
    "ManifestHashCache",
    "GemmaLatentConfig",
    "GemmaPRLRTrainer",
    "GemmaTrainerConfig",
    "compute_masked_ce_loss",
    "verify_model_manifest",
    "ManifestError",
    "IntegrityError",
    "Rule5ViolationError",
    "RuleViolationError",
    "ArchitectureMismatchError",
    "AblationSpec",
    "InstanceEvaluationRecord",
    "AblationConditionSummary",
    "AblationSuiteReport",
    "compute_bootstrap_ci_95",
    "GemmaAblationHarness",
    "CalibratedGateThresholds",
    "EGateStepTelemetry",
    "GemmaCalibratedEGate",
    "EGateCalibrator",
]

