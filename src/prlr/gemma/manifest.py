"""ModelManifest re-export for prlr.gemma namespace."""

from __future__ import annotations

from prlr.manifest import (
    ArchitectureMismatchError,
    IntegrityError,
    ManifestError,
    ModelManifest,
    Rule5ViolationError,
    RuleViolationError,
    VocabularyAlignmentError,
    VocabularyMismatchError,
    load_model,
    verify_model_manifest,
)

__all__ = [
    "ModelManifest",
    "verify_model_manifest",
    "load_model",
    "ManifestError",
    "IntegrityError",
    "Rule5ViolationError",
    "RuleViolationError",
    "ArchitectureMismatchError",
    "VocabularyMismatchError",
    "VocabularyAlignmentError",
]
