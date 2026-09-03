"""PRLR: Parallel Latent Deliberation on Apple Silicon.

Canonical Namespaces:
- `prlr.kernel`: Pure model-agnostic recurrence mathematics, Jacobi sweeps, attention, gates.
- `prlr.compact`: Honest 256D testbed model for CI and unit tests (is_pretrained=False).
- `prlr.gemma`: Pretrained Gemma vertical lane with cryptographic manifest validation.
- `prlr.manifest`: Cryptographic ModelManifest tracking weights, tokenizers, runtime versions.
"""

from __future__ import annotations

from prlr.manifest import (
    ArchitectureMismatchError,
    IntegrityError,
    ManifestError,
    ModelManifest,
    Rule5ViolationError,
    RuleViolationError,
    load_model,
    verify_model_manifest,
)
from prlr import kernel
from prlr import compact
from prlr import gemma
from prlr import domain

__version__ = "0.2.0.dev0"

__all__ = [
    "ModelManifest",
    "verify_model_manifest",
    "load_model",
    "IntegrityError",
    "Rule5ViolationError",
    "RuleViolationError",
    "ArchitectureMismatchError",
    "ManifestError",
    "kernel",
    "compact",
    "gemma",
    "domain",
    "__version__",
]
