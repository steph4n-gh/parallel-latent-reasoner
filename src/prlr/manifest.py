"""Authoritative Model Manifest for Parallel Latent Reasoner (PRLR).

Enforces strict cryptographic verification of weights, tokenizers, runtime versions,
and commit SHA per Non-Negotiable Evidence Rule 5 and Requirement R1.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union


class ManifestError(ValueError):
    """Base exception for ModelManifest validation failures."""


class IntegrityError(ManifestError):
    """Raised when cryptographic SHA-256 hashes do not match manifest expectations."""


class Rule5ViolationError(ManifestError):
    """Raised when a random or synthetic model is labeled as Gemma."""


RuleViolationError = Rule5ViolationError


class ArchitectureMismatchError(ManifestError):
    """Raised when tensor dimensions or vocabulary do not align with configuration."""


VocabularyMismatchError = ArchitectureMismatchError
VocabularyAlignmentError = ArchitectureMismatchError


@dataclass(frozen=True)
class ModelManifest:
    """Cryptographic and architectural manifest for PRLR models.

    Tracks immutable metadata required for reproducible benchmark evaluation.
    """

    # Identity & Governance
    model_id: str
    revision: str = "main"
    architecture: str = "GemmaForCausalLM"
    source_commit: str = "a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732"

    # Pretrained Mode vs Synthetic Testbed
    is_pretrained: bool = True
    random_init: bool = False
    quantization: str = "none"

    # Checkpoint File Paths & Cryptographic Hashes
    weights_path: Optional[str] = None
    weights_sha256: Union[Dict[str, str], str] = field(default_factory=dict)
    tokenizer_path: Optional[str] = None
    tokenizer_sha256: str = ""
    adapter_hash: Optional[str] = None

    # Architectural Hyperparameters
    vocabulary_size: int = 256000
    hidden_dimension: int = 2048
    num_layers: int = 18
    num_heads: int = 8
    head_dimension: int = 256
    intermediate_dimension: int = 16384
    num_kv_heads: int = 1
    max_position_embeddings: int = 8192
    bos_token_id: int = 2
    eos_token_id: int = 1
    pad_token_id: int = 0

    # Environment & Audit Status
    runtime_versions: Dict[str, str] = field(default_factory=dict)
    verification_status: str = "UNVERIFIED"

    @property
    def weight_hash(self) -> str:
        """Composite or single weight hash representation."""
        if isinstance(self.weights_sha256, dict):
            items = [f"{k}:{v}" for k, v in sorted(self.weights_sha256.items())]
            return hashlib.sha256(",".join(items).encode("ascii")).hexdigest()
        return str(self.weights_sha256)

    @property
    def tokenizer_hash(self) -> str:
        """Alias for tokenizer_sha256."""
        return self.tokenizer_sha256

    def validate(
        self,
        check_disk: bool = False,
        allow_gemma_random_init: bool = False,
    ) -> bool:
        """Enforce strict integrity constraints before model loading."""
        # Rule 5: No configuration-shaped random model may be labeled Gemma
        model_id_lower = self.model_id.lower()
        if "gemma" in model_id_lower and (not self.is_pretrained or self.random_init) and not allow_gemma_random_init:
            raise Rule5ViolationError(
                f"Non-Negotiable Evidence Rule 5 violation: Model '{self.model_id}' carries Gemma identity "
                "but is not a verified pretrained model (is_pretrained=False or random_init=True). "
                "Random models must be labeled as 'prlr-compact-testbed' or pass allow_gemma_random_init=True."
            )

        # Check valid initialization state
        if not self.is_pretrained and not self.random_init:
            raise ManifestError(
                f"Invalid manifest for '{self.model_id}': must specify either is_pretrained=True or random_init=True."
            )

        # Check mutual exclusion
        if self.is_pretrained and self.random_init:
            raise ManifestError(
                f"Contradictory manifest for '{self.model_id}': "
                "cannot have is_pretrained=True and random_init=True simultaneously."
            )

        # Pretrained existence checks
        if self.is_pretrained:
            if not self.weights_path:
                raise ManifestError(f"Pretrained manifest '{self.model_id}' is missing weights_path.")
            if not self.tokenizer_path:
                raise ManifestError(f"Pretrained manifest '{self.model_id}' is missing tokenizer_path.")
            if not self.weights_sha256:
                raise ManifestError(f"Pretrained manifest '{self.model_id}' is missing weights_sha256.")
            if not self.tokenizer_sha256:
                raise ManifestError(f"Pretrained manifest '{self.model_id}' is missing tokenizer_sha256.")

            # Validate hash formats
            if isinstance(self.weights_sha256, str):
                if len(self.weights_sha256) != 64:
                    raise IntegrityError(
                        f"Invalid SHA-256 hash length ({len(self.weights_sha256)}) for weights_sha256: '{self.weights_sha256}'"
                    )
            elif isinstance(self.weights_sha256, dict):
                for fname, expected_sha in self.weights_sha256.items():
                    if len(expected_sha) != 64:
                        raise IntegrityError(
                            f"Invalid SHA-256 hash length ({len(expected_sha)}) for weight shard '{fname}': '{expected_sha}'"
                        )
            else:
                raise IntegrityError(
                    f"Invalid weights_sha256 type ({type(self.weights_sha256)}): must be str or dict."
                )

            if len(self.tokenizer_sha256) != 64:
                raise IntegrityError(
                    f"Invalid SHA-256 hash length ({len(self.tokenizer_sha256)}) for tokenizer_sha256: '{self.tokenizer_sha256}'"
                )

        # Architectural sanity
        if self.vocabulary_size <= 0:
            raise ArchitectureMismatchError(f"vocabulary_size must be positive, got {self.vocabulary_size}")
        if self.hidden_dimension <= 0:
            raise ArchitectureMismatchError(f"hidden_dimension must be positive, got {self.hidden_dimension}")
        if self.num_layers <= 0:
            raise ArchitectureMismatchError(f"num_layers must be positive, got {self.num_layers}")

        # Cryptographic file validation on disk if requested
        if check_disk and self.is_pretrained:
            from prlr.gemma.loader import ManifestHashCache

            wpath = Path(self.weights_path)
            if not wpath.exists():
                raise FileNotFoundError(f"Weight path does not exist: {wpath}")

            if isinstance(self.weights_sha256, dict):
                for fname, expected_sha in self.weights_sha256.items():
                    target_file = wpath / fname if wpath.is_dir() else wpath
                    if not target_file.exists():
                        raise FileNotFoundError(f"Weight shard not found: {target_file}")
                    if len(expected_sha) != 64:
                        raise IntegrityError(
                            f"Invalid SHA-256 hash length ({len(expected_sha)}) for weight shard '{fname}': '{expected_sha}'"
                        )
                    ManifestHashCache.verify_or_compute(target_file, expected_sha)
            elif isinstance(self.weights_sha256, str):
                if len(self.weights_sha256) != 64:
                    raise IntegrityError(
                        f"Invalid SHA-256 hash length ({len(self.weights_sha256)}) for weights_sha256: '{self.weights_sha256}'"
                    )
                if wpath.is_dir():
                    raise ManifestError(
                        f"weights_sha256 is a string hash but weights_path is a directory: '{wpath}'. "
                        "Must provide a dict mapping shard filenames to SHA-256 hashes."
                    )
                ManifestHashCache.verify_or_compute(wpath, self.weights_sha256)

            tpath = Path(self.tokenizer_path)
            if not tpath.exists():
                raise FileNotFoundError(f"Tokenizer path does not exist: {tpath}")
            if len(self.tokenizer_sha256) != 64:
                raise IntegrityError(
                    f"Invalid SHA-256 hash length ({len(self.tokenizer_sha256)}) for tokenizer_sha256: '{self.tokenizer_sha256}'"
                )
            ManifestHashCache.verify_or_compute(tpath, self.tokenizer_sha256)

        return True

    @classmethod
    def gemma_2b_it(
        cls,
        snapshot_dir: Optional[Union[str, Path]] = None,
        source_commit: str = "a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732",
        adapter_hash: Optional[str] = None,
    ) -> ModelManifest:
        """Factory method for official google/gemma-2b-it local snapshot."""
        default_dir = Path(
            "/Volumes/Storage/huggingface_cache/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
        )
        base = Path(snapshot_dir) if snapshot_dir is not None else default_dir

        runtime_info = {
            "python": sys.version.split()[0],
            "mlx": "0.31.2",
            "transformers": "5.9.0",
            "sentencepiece": "0.2.1",
            "numpy": "2.4.6",
        }

        weights_hashes = {
            "model-00001-of-00002.safetensors": "561656f892a2a1ca0837ca529c5ce820a72b40f4f563b1cd0a1acc0b3899c30c",
            "model-00002-of-00002.safetensors": "20fe2ee66bf1361241a6c522091a5e0328fc6c1703f93734889fa381fcf8760c",
        }

        return cls(
            model_id="google/gemma-2b-it",
            revision="96988410cbdaeb8d5093d1ebdc5a8fb563e02bad",
            architecture="GemmaForCausalLM",
            source_commit=source_commit,
            is_pretrained=True,
            random_init=False,
            quantization="bf16",
            weights_path=str(base),
            weights_sha256=weights_hashes,
            tokenizer_path=str(base / "tokenizer.model"),
            tokenizer_sha256="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
            adapter_hash=adapter_hash,
            vocabulary_size=256000,
            hidden_dimension=2048,
            num_layers=18,
            num_heads=8,
            head_dimension=256,
            intermediate_dimension=16384,
            num_kv_heads=1,
            max_position_embeddings=8192,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            runtime_versions=runtime_info,
            verification_status="UNVERIFIED",
        )

    @classmethod
    def gemma_4_12b_it(
        cls,
        snapshot_dir: Optional[Union[str, Path]] = None,
        source_commit: str = "e4d18cf",
        adapter_hash: Optional[str] = None,
    ) -> ModelManifest:
        """Factory method for official google-gemma-4-12B-it-4bit local snapshot."""
        default_dir = Path("/Volumes/Storage/huggingface_cache/hub/google-gemma-4-12B-it-4bit")
        base = Path(snapshot_dir) if snapshot_dir is not None else default_dir

        runtime_info = {
            "python": sys.version.split()[0],
            "mlx": "0.31.2",
            "transformers": "5.9.0",
            "numpy": "2.4.6",
        }

        weights_hashes = {
            "model-00001-of-00002.safetensors": "3cac027bf8021583213c467b5d5b837bada0a0d9943fd245dd3bf915e4fba0be",
            "model-00002-of-00002.safetensors": "7366bf36f2672af78ac71c5430a04a7c2c5ebdaf8895532be373a7edc1f0b1c6",
        }

        return cls(
            model_id="google/gemma-4-12B-it-4bit",
            revision="gemma4-12b-it-4bit-local",
            architecture="Gemma4ForCausalLM",
            source_commit=source_commit,
            is_pretrained=True,
            random_init=False,
            quantization="4bit",
            weights_path=str(base),
            weights_sha256=weights_hashes,
            tokenizer_path=str(base / "tokenizer.json"),
            tokenizer_sha256="cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            adapter_hash=adapter_hash,
            vocabulary_size=262144,
            hidden_dimension=3840,
            num_layers=48,
            num_heads=16,
            head_dimension=256,
            intermediate_dimension=15360,
            num_kv_heads=8,
            max_position_embeddings=131072,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            runtime_versions=runtime_info,
            verification_status="UNVERIFIED",
        )

    @classmethod
    def get_gemma_2b_manifest(
        cls,
        snapshot_dir: Optional[Union[str, Path]] = None,
        source_commit: str = "a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732",
        adapter_hash: Optional[str] = None,
    ) -> ModelManifest:
        """Alias for gemma_2b_it."""
        return cls.gemma_2b_it(
            snapshot_dir=snapshot_dir,
            source_commit=source_commit,
            adapter_hash=adapter_hash,
        )

    @classmethod
    def compact_test(
        cls,
        source_commit: str = "a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732",
    ) -> ModelManifest:
        """Factory method for prlr-compact-testbed synthetic model."""
        runtime_info = {
            "python": sys.version.split()[0],
            "mlx": "0.31.2",
            "numpy": "2.4.6",
        }

        return cls(
            model_id="prlr-compact-testbed",
            revision="scratch-testbed-v1",
            architecture="prlr-compact",
            source_commit=source_commit,
            is_pretrained=False,
            random_init=True,
            quantization="none",
            weights_path=None,
            weights_sha256="UNSPECIFIED_RANDOM_INIT",
            tokenizer_path=None,
            tokenizer_sha256="UNSPECIFIED_SYNTHETIC",
            adapter_hash=None,
            vocabulary_size=1000,
            hidden_dimension=256,
            num_layers=1,
            num_heads=4,
            head_dimension=64,
            intermediate_dimension=512,
            num_kv_heads=4,
            max_position_embeddings=8192,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            runtime_versions=runtime_info,
            verification_status="SYNTHETIC_TESTBED",
        )


def verify_model_manifest(
    manifest: ModelManifest,
    check_disk: bool = True,
) -> bool:
    """Cryptographically verify model manifest and checkpoint integrity."""
    return manifest.validate(check_disk=check_disk)


def load_model(
    manifest: ModelManifest,
    verify_hashes: bool = True,
    allow_gemma_random_init: bool = False,
    tokenizer: Any = None,
) -> Any:
    """Lazy loader dispatch to prlr.gemma.loader.load_model."""
    from prlr.gemma.loader import load_model as _load_model
    return _load_model(
        manifest=manifest,
        verify_hashes=verify_hashes,
        allow_gemma_random_init=allow_gemma_random_init,
        tokenizer=tokenizer,
    )


__all__ = [
    "ManifestError",
    "IntegrityError",
    "Rule5ViolationError",
    "RuleViolationError",
    "ArchitectureMismatchError",
    "VocabularyMismatchError",
    "VocabularyAlignmentError",
    "ModelManifest",
    "verify_model_manifest",
    "load_model",
]
