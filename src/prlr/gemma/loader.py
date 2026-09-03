"""Cryptographic model loader and integrity verifier for PRLR.

Implements streaming SHA-256 verification, caching, vocabulary alignment,
and Rule 5 enforcement before returning model instances.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import mlx.core as mx

from prlr.manifest import (
    ArchitectureMismatchError,
    IntegrityError,
    ManifestError,
    ModelManifest,
    Rule5ViolationError,
)


class LoadedModel(NamedTuple):
    """Container for loaded and cryptographically verified model components."""

    model: Any
    tokenizer: Any
    manifest: ModelManifest


class ManifestHashCache:
    """Mtime and size-keyed in-memory hash cache to avoid repeated 5GB disk scans."""

    _cache: dict[str, tuple[int, int, str]] = {}

    @classmethod
    def verify_or_compute(
        cls,
        path: Path,
        expected_sha256: str,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Weight/tokenizer file not found: {path}")

        stat = path.stat()
        key = str(path.resolve())
        cached = cls._cache.get(key)

        if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            actual_sha256 = cached[2]
        else:
            hasher = hashlib.sha256()
            with open(path, "rb") as fp:
                while chunk := fp.read(chunk_size):
                    hasher.update(chunk)
            actual_sha256 = hasher.hexdigest()
            cls._cache[key] = (stat.st_mtime_ns, stat.st_size, actual_sha256)

        if actual_sha256.lower() != expected_sha256.lower():
            raise IntegrityError(
                f"Cryptographic SHA-256 mismatch for '{path.name}'!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Observed: {actual_sha256}\n"
                f"Possible file corruption, tampering, or Rule 5 violation."
            )

        return actual_sha256


def load_model(
    manifest: ModelManifest,
    verify_hashes: bool = True,
    allow_gemma_random_init: bool = False,
    tokenizer: Any = None,
) -> LoadedModel:
    """Load model weights and tokenizer with strict manifest verification.

    Args:
        manifest: Target ModelManifest describing model identity, hashes, and architecture.
        verify_hashes: Whether to compute streaming SHA-256 checks.
        allow_gemma_random_init: Explicit override allowing random init for Gemma (default False).
        tokenizer: Optional pre-loaded or mock tokenizer instance.

    Returns:
        LoadedModel namedtuple with (model, tokenizer, verified_manifest).
    """
    # 1. Manifest logical validation
    manifest.validate(allow_gemma_random_init=allow_gemma_random_init)

    # 2. Synthetic compact testbed branch
    if not manifest.is_pretrained:
        from prlr.compact.scratch_model import MLXCompactGemmaModel

        model = MLXCompactGemmaModel.from_manifest(manifest)
        verified_manifest = dataclasses.replace(manifest, verification_status="VERIFIED_SCRATCH")
        return LoadedModel(model=model, tokenizer=tokenizer, manifest=verified_manifest)

    # 3. Checkpoint file paths existence
    if not manifest.weights_path:
        raise FileNotFoundError(f"Model weights path missing from manifest: {manifest.model_id}")
    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights path does not exist: {weights_path}")

    if not manifest.tokenizer_path:
        raise FileNotFoundError(f"Tokenizer path missing from manifest: {manifest.model_id}")
    tokenizer_path = Path(manifest.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    # 4. Cryptographic SHA-256 validation
    if verify_hashes:
        if isinstance(manifest.weights_sha256, dict):
            for filename, expected_hash in manifest.weights_sha256.items():
                file_to_check = weights_path / filename if weights_path.is_dir() else weights_path
                if len(expected_hash) != 64:
                    raise IntegrityError(
                        f"Invalid SHA-256 hash length ({len(expected_hash)}) for weight shard '{filename}': '{expected_hash}'"
                    )
                ManifestHashCache.verify_or_compute(file_to_check, expected_hash)
        elif isinstance(manifest.weights_sha256, str):
            if len(manifest.weights_sha256) != 64:
                raise IntegrityError(
                    f"Invalid SHA-256 hash length ({len(manifest.weights_sha256)}) for weights_sha256: '{manifest.weights_sha256}'"
                )
            if weights_path.is_dir():
                raise ManifestError(
                    f"weights_sha256 is a string hash but weights_path is a directory: '{weights_path}'. "
                    "Must provide a dict mapping shard filenames to SHA-256 hashes."
                )
            ManifestHashCache.verify_or_compute(weights_path, manifest.weights_sha256)
        else:
            raise IntegrityError(
                f"Invalid weights_sha256 type ({type(manifest.weights_sha256)}): must be str or dict."
            )

        if len(manifest.tokenizer_sha256) != 64:
            raise IntegrityError(
                f"Invalid SHA-256 hash length ({len(manifest.tokenizer_sha256)}) for tokenizer_sha256: '{manifest.tokenizer_sha256}'"
            )
        ManifestHashCache.verify_or_compute(tokenizer_path, manifest.tokenizer_sha256)

    # 5. Tokenizer loading and vocabulary alignment
    if tokenizer is not None:
        loaded_tokenizer = tokenizer
    else:
        try:
            from transformers import AutoTokenizer

            tok_dir = tokenizer_path.parent if tokenizer_path.is_file() else tokenizer_path
            loaded_tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        except Exception:
            import sentencepiece as spm

            loaded_tokenizer = spm.SentencePieceProcessor()
            loaded_tokenizer.load(str(tokenizer_path))

    tok_vocab = getattr(loaded_tokenizer, "vocab_size", None)
    if tok_vocab is None and hasattr(loaded_tokenizer, "get_piece_size"):
        tok_vocab = loaded_tokenizer.get_piece_size()
    if tok_vocab is not None and tok_vocab != manifest.vocabulary_size:
        raise ArchitectureMismatchError(
            f"Tokenizer vocab_size ({tok_vocab}) does not match manifest "
            f"vocabulary_size ({manifest.vocabulary_size})."
        )

    bos_id = getattr(loaded_tokenizer, "bos_token_id", None)
    if bos_id is None and hasattr(loaded_tokenizer, "bos_id"):
        bos_id = loaded_tokenizer.bos_id()
    if bos_id is not None and bos_id != manifest.bos_token_id:
        raise ArchitectureMismatchError(
            f"Tokenizer bos_token_id ({bos_id}) != manifest ({manifest.bos_token_id})."
        )

    eos_id = getattr(loaded_tokenizer, "eos_token_id", None)
    if eos_id is None and hasattr(loaded_tokenizer, "eos_id"):
        eos_id = loaded_tokenizer.eos_id()
    if eos_id is not None and eos_id != manifest.eos_token_id:
        raise ArchitectureMismatchError(
            f"Tokenizer eos_token_id ({eos_id}) != manifest ({manifest.eos_token_id})."
        )

    # 6. Load MLX Model Backbone
    import mlx_lm

    model, _ = mlx_lm.load(str(weights_path))
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if len(model.model.layers) != manifest.num_layers:
            raise ArchitectureMismatchError(
                f"Loaded model layer count ({len(model.model.layers)}) != manifest ({manifest.num_layers})"
            )

    verified_manifest = dataclasses.replace(manifest, verification_status="VERIFIED")
    return LoadedModel(model=model, tokenizer=loaded_tokenizer, manifest=verified_manifest)


__all__ = [
    "LoadedModel",
    "ManifestHashCache",
    "load_model",
]
