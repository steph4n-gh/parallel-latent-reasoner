"""Two-Stage Target-Free Evidence Harness & Cryptographic Provenance.

Operationalizes Non-Negotiable Evidence Rules 1, 2, 5, and 10:
- Rule 1: No inference or generation function may receive, access, capture, or derive
  the ground truth, expected constraints, verifier answer, or evaluation metadata.
- Rule 2: The answer key may be accessed only after immutable generated output has
  been recorded and cryptographically sealed with SHA-256.
- Rule 5: No configuration-shaped random model may be labeled Gemma. Abort immediately
  if requested adapter checkpoint is missing (never fallback to random weights).
- Rule 10: Every published result must include raw predictions, exact commands,
  hardware, model/tokenizer identifiers, hashes, seeds, runtime versions, and commit SHA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union
import unicodedata

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from mlx_lm.tokenizer_utils import TokenizerWrapper

from prlr.domain.prompt_format import (
    extract_user_body,
    format_canonical_prompt,
    is_gemma4_tokenizer,
)
from prlr.domain.solver_lane import DOMAIN_CATALOGUES, ProceduralVerifier
from prlr.eval.semantic_bench import (
    compute_max_ngram_repetition,
    compute_shannon_entropy,
)
from prlr.gemma.adapter import GemmaNonRecurrentAdapter, GemmaRecurrentAdapter
from prlr.gemma.backbone import GemmaTokenizerWrapper, PretrainedGemmaBackbone
from prlr.gemma.decoder import GatedCrossAttentionInjection, GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest

# Canonical order-reversal derangement for M=16 working memory slots
FIXED_SLOT_PERMUTATION: Tuple[int, ...] = (
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
)


# ==============================================================================
# 1. Exception Hierarchy
# ==============================================================================

class HarnessError(Exception):
    """Base exception for PRLR evaluation harness failures."""


class OracleLeakageError(HarnessError, ValueError):
    """Raised when an inference input contains ground-truth, target, or verifier fields (Rule 1)."""


# Aliases for compatibility across specs
TargetLeakageException = OracleLeakageError
Rule1ViolationError = OracleLeakageError


class MissingCheckpointError(HarnessError, FileNotFoundError):
    """Raised when an adapter checkpoint is required but missing (Rule 5)."""


CheckpointMissingError = MissingCheckpointError


class PredictionIntegrityError(HarnessError, ValueError):
    """Raised when a prediction file fails SHA-256 sidecar or checksum verification (Rule 2)."""


# Compatibility aliases (MUST remain available in __all__ and module exports)
IntegrityError = PredictionIntegrityError
TamperedPredictionError = PredictionIntegrityError


class MissingSidecarError(PredictionIntegrityError, FileNotFoundError):
    """Raised when mandatory SHA-256 sidecar file does not exist on disk."""


class EmptySidecarError(PredictionIntegrityError):
    """Raised when SHA-256 sidecar file exists but has 0 bytes or whitespace-only content."""


class MalformedSidecarError(PredictionIntegrityError):
    """Raised when SHA-256 sidecar text does not contain a valid 64-character hex hash."""


class ChecksumMismatchError(PredictionIntegrityError):
    """Raised when the actual computed SHA-256 of the prediction file diverges from the sidecar."""


class CrossRunMergeRejectionError(HarnessError, RuntimeError):
    """Raised when an attempt is made to consolidate summaries from incompatible runs."""


SummaryMergeConflictError = CrossRunMergeRejectionError


class SampleMismatchError(HarnessError, ValueError):
    """Raised when prediction records and quarantined answer keys do not match."""


# ==============================================================================
# 2. Forbidden Oracle Terms & Target-Free Validators
# ==============================================================================

FORBIDDEN_ORACLE_TERMS: Set[str] = {
    # Direct ground truth & target solution labels
    "target_solution",
    "ground_truth",
    "expected_route",
    "terminal_tool",
    "target_goal",
    "target_ids",
    "target_tokens",
    "target_mask",
    "labels",
    "solution_sha256",
    # Verifier configurations & oracles
    "verifier_config",
    "verifier_type",
    "oracle_solution",
    "oracle_route",
    "gold_route",
    "gold_solution",
    "reference_solution",
    "answer_key",
    "answer_keys",
    "expected_terminal",
    "bfs_oracle",
    "key_record",
    # Generic solution & answer leaks
    "target",
    "solution",
    "answer",
    "expected",
    "verifier",
    "oracle",
}

SQUASHED_ORACLE_TERMS: Set[str] = {
    re.sub(r"[^a-z0-9]", "", term) for term in FORBIDDEN_ORACLE_TERMS
}

# Root tokens that are strictly prohibited as distinct words in any input key
FORBIDDEN_ROOT_TOKENS: Set[str] = {
    "target", "targets",
    "solution", "solutions",
    "verifier", "verifiers",
    "oracle", "oracles",
    "answer", "answers",
    "label", "labels",
    "expected",
}

# Squashed substrings with zero innocent English collisions
FORBIDDEN_SQUASHED_SUBSTRINGS: Tuple[str, ...] = (
    "target",
    "verifier",
    "oracle",
    "groundtruth",
    "expectedroute",
    "terminaltool",
    "targetgoal",
    "answerkey",
    "goldroute",
    "goldsolution",
    "goldanswer",
    "verifierspec",
    "bfsoracle",
)

# Compound phrases in snake_case format
FORBIDDEN_COMPOUND_PHRASES: Tuple[str, ...] = (
    "target_solution",
    "ground_truth",
    "expected_route",
    "verifier_config",
    "terminal_tool",
    "target_goal",
    "answer_key",
    "oracle_solution",
    "gold_route",
    "gold_solution",
    "reference_solution",
    "bfs_oracle",
    "expected_terminal",
    "verifier_spec",
    "gold_answer",
    "custom_verifier",
    "final_solution",
    "model_answer",
    "user_target",
    "oracle_path",
)

FORBIDDEN_KEY_SUBSTRINGS: Tuple[str, ...] = (
    "target_solution",
    "ground_truth",
    "expected_route",
    "verifier_config",
    "answer_key",
    "oracle_solution",
)

# Legitimate non-oracle words that contain the substring 'solution'
LEGITIMATE_SOLUTION_WORDS: Set[str] = {
    "resolution",
    "resolutions",
    "dissolution",
    "absolution",
}


def split_camel_case(s: str) -> str:
    """Split camelCase and PascalCase word boundaries into underscore-separated components."""
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)


def normalize_oracle_key(key: str) -> Tuple[str, str, Set[str]]:
    """Normalizes an input dictionary key into canonical snake, squashed, and token representations."""
    key_clean = unicodedata.normalize("NFKC", str(key)).strip()
    key_split = split_camel_case(key_clean)
    key_snake = re.sub(r"[^a-zA-Z0-9]+", "_", key_split).strip("_").lower()
    key_squashed = re.sub(r"[^a-zA-Z0-9]", "", key_clean).lower()
    tokens = set(t for t in key_snake.split("_") if t)
    return key_snake, key_squashed, tokens


def is_forbidden_oracle_key(key: str) -> Tuple[bool, str]:
    """Determines whether a dictionary key violates Evidence Rule 1.

    Returns:
        tuple of (is_forbidden: bool, reason: str)
    """
    key_snake, key_squashed, tokens = normalize_oracle_key(key)

    # 1. Exact match against canonical terms (snake-case or squashed)
    if key_snake in FORBIDDEN_ORACLE_TERMS or key_squashed in SQUASHED_ORACLE_TERMS:
        return True, f"exact match with prohibited oracle term in '{key}'"

    # 2. Distinct root word token match
    root_match = tokens.intersection(FORBIDDEN_ROOT_TOKENS)
    if root_match:
        return True, f"prohibited oracle root word(s) {root_match} in key '{key}'"

    # 3. Compound phrase match in snake-case representation
    for phrase in FORBIDDEN_COMPOUND_PHRASES:
        if phrase in key_snake:
            return True, f"prohibited compound phrase '{phrase}' in key '{key}'"

    # 4. Squashed substring match for collision-free oracle stems
    for s_sub in FORBIDDEN_SQUASHED_SUBSTRINGS:
        if s_sub in key_squashed:
            return True, f"prohibited squashed oracle stem '{s_sub}' in key '{key}'"

    # 5. Solution substring check with false-positive guard
    if "solution" in key_squashed:
        if not any(lw in key_snake for lw in LEGITIMATE_SOLUTION_WORDS):
            return True, f"prohibited solution keyword in key '{key}'"

    # 6. Answer substring in squashed representation
    if "answer" in key_squashed:
        return True, f"prohibited answer keyword in key '{key}'"

    # 7. Multi-token semantic combinations
    if "ground" in tokens and "truth" in tokens:
        return True, f"prohibited ground truth semantic tokens in key '{key}'"
    if "terminal" in tokens and "tool" in tokens:
        return True, f"prohibited terminal tool semantic tokens in key '{key}'"
    if "expected" in tokens and "route" in tokens:
        return True, f"prohibited expected route semantic tokens in key '{key}'"

    return False, ""


def validate_target_free_dict(
    data: Any,
    context: str = "Inference Input",
    path: str = "root",
    depth: int = 0,
) -> None:
    """Recursively validates that data contains strictly zero oracle terms.

    Raises:
        OracleLeakageError: If any forbidden key or string pattern is detected.
        ValueError: If maximum recursion depth is exceeded.
    """
    if depth > 32:
        raise ValueError(f"Excessive nesting depth ({depth}) in {context} at path '{path}'.")

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(key, str):
                raise OracleLeakageError(
                    f"Non-string key '{key}' at path '{path}' in {context} violates schema."
                )

            is_forbidden, reason = is_forbidden_oracle_key(key)
            if is_forbidden:
                raise OracleLeakageError(
                    f"Evidence Rule 1 Violation: Prohibited oracle key '{key}' ({reason}) "
                    f"detected at path '{path}.{key}' in {context}!"
                )

            # Recurse into nested structures
            validate_target_free_dict(value, context=context, path=f"{path}.{key}", depth=depth + 1)

    elif isinstance(data, (list, tuple, set)):
        for idx, item in enumerate(data):
            validate_target_free_dict(item, context=context, path=f"{path}[{idx}]", depth=depth + 1)


def validate_target_free_record(record: Any) -> None:
    """Validate that an in-memory sample or record contains zero oracle data."""
    if isinstance(record, dict):
        validate_target_free_dict(record, context="Target-free record validation")
    elif hasattr(record, "to_dict") and callable(record.to_dict):
        for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
            if hasattr(record, forbidden):
                raise OracleLeakageError(
                    f"Evidence Rule 1 Violation: Object of type {type(record).__name__} contains oracle attribute '{forbidden}'."
                )
        validate_target_free_dict(record.to_dict(), context=f"{type(record).__name__} dict validation")
    else:
        for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
            if hasattr(record, forbidden):
                raise OracleLeakageError(
                    f"Evidence Rule 1 Violation: Object of type {type(record).__name__} contains oracle attribute '{forbidden}'."
                )


# ==============================================================================
# 3. Core Data Schemas (frozen=True)
# ==============================================================================

@dataclass(frozen=True)
class EvaluationInput:
    """Isolated problem input for model evaluation.

    Guarantees strict zero ground-truth per Evidence Rule 1.
    """
    id: str
    split: str
    domain: str
    difficulty: int
    num_steps: int
    prompt: str
    prompt_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("EvaluationInput.id must be a non-empty string.")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("EvaluationInput.domain must be a non-empty string.")
        if not isinstance(self.difficulty, int) or self.difficulty < 1:
            raise ValueError(f"EvaluationInput.difficulty must be an integer >= 1, got {self.difficulty}")
        if not isinstance(self.num_steps, int) or self.num_steps < 1:
            raise ValueError(f"EvaluationInput.num_steps must be an integer >= 1, got {self.num_steps}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("EvaluationInput.prompt must be a non-empty string.")

        computed_sha = hashlib.sha256(self.prompt.strip().encode("utf-8")).hexdigest()
        if self.prompt_sha256:
            if self.prompt_sha256 != computed_sha:
                raise ValueError(
                    f"Prompt SHA-256 mismatch for sample '{self.id}': "
                    f"expected '{self.prompt_sha256}', computed '{computed_sha}'"
                )
        else:
            object.__setattr__(self, "prompt_sha256", computed_sha)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "num_steps": self.num_steps,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationInput:
        validate_target_free_dict(data, context=f"EvaluationInput.from_dict for id='{data.get('id', 'unknown')}'")
        return cls(
            id=data["id"],
            split=data.get("split", "unknown"),
            domain=data["domain"],
            difficulty=int(data.get("difficulty", 1)),
            num_steps=int(data.get("num_steps", 1)),
            prompt=data["prompt"],
            prompt_sha256=data.get("prompt_sha256", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationInput:
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class PredictionRecord:
    """Immutable record of an individual generation trial emitted atomically by Stage 1."""
    sample_id: str
    domain: str
    condition: str
    recurrence_depth: int
    generated_token_ids: Tuple[int, ...]
    decoded_text: str
    latency_ms: float
    prompt_sha256: str
    git_commit_sha: str
    is_dirty: bool
    checkpoint_sha256: str
    model_id: str
    tokenizer_id: str
    tokenizer_sha256: str = ""
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    runtime_versions: Dict[str, str] = field(default_factory=dict)
    timestamp_utc: str = ""
    prediction_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("PredictionRecord.sample_id must not be empty.")
        if not self.condition:
            raise ValueError("PredictionRecord.condition must not be empty.")
        if self.recurrence_depth < 0:
            raise ValueError(f"PredictionRecord.recurrence_depth must be >= 0, got {self.recurrence_depth}")
        if self.latency_ms < 0.0:
            raise ValueError(f"PredictionRecord.latency_ms must be >= 0.0, got {self.latency_ms}")

        if isinstance(self.generated_token_ids, list):
            object.__setattr__(self, "generated_token_ids", tuple(self.generated_token_ids))

        for tok in self.generated_token_ids:
            if not isinstance(tok, int) or tok < 0:
                raise ValueError(f"Invalid token ID in prediction: {tok}")

        forbidden_score_fields = {
            "exact_match", "terminal_match", "is_valid", "expected_route",
            "target_solution", "ground_truth", "verifier_config", "score"
        }
        for field_name in forbidden_score_fields:
            if field_name in self.__dict__:
                raise OracleLeakageError(
                    f"Rule 1 & 2 Violation: PredictionRecord cannot contain scoring field '{field_name}'."
                )

        if not self.prediction_sha256:
            digest_payload = (
                f"{self.sample_id}|{self.condition}|{self.recurrence_depth}|"
                f"{','.join(map(str, self.generated_token_ids))}|"
                f"{self.decoded_text.strip()}|{self.prompt_sha256}"
            ).encode("utf-8")
            object.__setattr__(self, "prediction_sha256", hashlib.sha256(digest_payload).hexdigest())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "condition": self.condition,
            "recurrence_depth": self.recurrence_depth,
            "generated_token_ids": list(self.generated_token_ids),
            "decoded_text": self.decoded_text,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_sha256": self.prompt_sha256,
            "git_commit_sha": self.git_commit_sha,
            "is_dirty": self.is_dirty,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hardware_info": self.hardware_info,
            "runtime_versions": self.runtime_versions,
            "timestamp_utc": self.timestamp_utc,
            "prediction_sha256": self.prediction_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PredictionRecord:
        tokens = tuple(data["generated_token_ids"]) if isinstance(data["generated_token_ids"], list) else data["generated_token_ids"]
        return cls(
            sample_id=data["sample_id"],
            domain=data.get("domain", "unknown"),
            condition=data["condition"],
            recurrence_depth=int(data["recurrence_depth"]),
            generated_token_ids=tokens,
            decoded_text=data["decoded_text"],
            latency_ms=float(data["latency_ms"]),
            prompt_sha256=data["prompt_sha256"],
            git_commit_sha=data["git_commit_sha"],
            is_dirty=bool(data["is_dirty"]),
            checkpoint_sha256=data.get("checkpoint_sha256", ""),
            model_id=data.get("model_id", ""),
            tokenizer_id=data.get("tokenizer_id", ""),
            tokenizer_sha256=data.get("tokenizer_sha256", ""),
            hardware_info=dict(data.get("hardware_info", {})),
            runtime_versions=dict(data.get("runtime_versions", {})),
            timestamp_utc=data.get("timestamp_utc", ""),
            prediction_sha256=data.get("prediction_sha256", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PredictionRecord:
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class LatencyStatistics:
    """Latency distribution summary in milliseconds."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionScoredSummary:
    """Summary metrics for a single experimental condition."""
    condition: str
    recurrence_depth: int
    sample_count: int
    exact_match_count: int
    exact_match_pct: float
    terminal_match_count: int
    terminal_match_pct: float
    valid_json_count: int
    valid_json_pct: float
    max_4gram_repetition: int
    mean_4gram_repetition: float
    mean_shannon_entropy: float
    latency: LatencyStatistics
    checkpoint_sha256: str
    model_id: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency"] = self.latency.to_dict()
        return d


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Cryptographic provenance sealing an evaluation run."""
    git_commit_sha: str
    is_dirty: bool
    dataset_name: str
    dataset_sha256: str
    split: str
    sample_count: int
    inputs_file: str
    inputs_file_sha256: str
    keys_file: str
    keys_file_sha256: str
    runtime_versions: Dict[str, str]
    hardware_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredSummaryArtifact:
    """Authoritative output of Stage 2 (score_predictions)."""
    schema_version: str
    created_at_utc: str
    provenance: ProvenanceMetadata
    prediction_files_sha256: Dict[str, str]
    conditions: Dict[str, ConditionScoredSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "provenance": self.provenance.to_dict(),
            "prediction_files_sha256": self.prediction_files_sha256,
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoredSummaryArtifact:
        prov = ProvenanceMetadata(**data["provenance"])
        conds = {}
        for c_name, c_data in data["conditions"].items():
            lat_data = c_data["latency"]
            lat = LatencyStatistics(**lat_data) if isinstance(lat_data, dict) else lat_data
            c_dict = dict(c_data)
            c_dict["latency"] = lat
            conds[c_name] = ConditionScoredSummary(**c_dict)
        return cls(
            schema_version=data["schema_version"],
            created_at_utc=data["created_at_utc"],
            provenance=prov,
            prediction_files_sha256=dict(data.get("prediction_files_sha256", {})),
            conditions=conds,
        )


# ==============================================================================
# 4. Provenance & System Telemetry Helpers
# ==============================================================================

def get_git_metadata() -> Dict[str, Any]:
    """Retrieve current git commit SHA, branch, and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        is_dirty = len(status) > 0
    except Exception:
        is_dirty = False

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        branch = "unknown"

    return {
        "git_commit_sha": commit,
        "is_dirty": is_dirty,
        "branch": branch,
    }


def get_hardware_metadata() -> Dict[str, Any]:
    """Retrieve hardware and device info on macOS / Metal."""
    hw: Dict[str, Any] = {
        "platform": sys.platform,
        "os_version": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        dev = mx.default_device()
        hw["mlx_device"] = str(dev)
        hw["is_metal_gpu"] = (dev.type == mx.gpu)
    except Exception:
        hw["mlx_device"] = "unknown"
        hw["is_metal_gpu"] = False
    return hw


def get_runtime_versions() -> Dict[str, str]:
    """Retrieve key package runtime versions."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
    }
    for pkg in ["mlx", "mlx_lm", "transformers", "numpy", "torch", "safetensors"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


# ==============================================================================
# 5. File Ingestion, Checkpoint Guards & Atomic Serialization
# ==============================================================================

def load_target_free_inputs(
    input_path: Path,
    limit: Optional[int] = None,
) -> Tuple[List[EvaluationInput], str]:
    """Load and validate target-free evaluation inputs.

    Returns:
        tuple of (list of EvaluationInput, input_file_sha256)
    Raises:
        FileNotFoundError: If input_path does not exist.
        OracleLeakageError: If any forbidden oracle key is present.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation input file not found: {input_path}")

    raw_bytes = input_path.read_bytes()
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    inputs: List[EvaluationInput] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            record: Dict[str, Any] = json.loads(line_str)
            sample_id = record.get("id", f"line_{line_idx}")

            # 1. Deep validation of record dict
            validate_target_free_dict(record, context=f"Input file '{input_path.name}:{line_idx}' (id='{sample_id}')")

            # 2. Construct EvaluationInput
            item = EvaluationInput.from_dict(record)
            inputs.append(item)

            if limit is not None and len(inputs) >= limit:
                break

    return inputs, input_sha256


def verify_adapter_checkpoint(
    checkpoint_path: Optional[Path],
    condition: str,
) -> Tuple[Path, str]:
    """Validate checkpoint existence and compute cryptographic hash.

    Raises:
        MissingCheckpointError: If checkpoint path is missing or non-existent (never fall back).
        ValueError: If checkpoint is empty (0 bytes).
    """
    if checkpoint_path is None:
        raise MissingCheckpointError(
            f"Evidence Rule 5 Violation: Required adapter checkpoint for condition '{condition}' "
            "was not specified (None). Aborting immediately."
        )

    cp = Path(checkpoint_path).resolve()
    if not cp.exists() or not cp.is_file():
        raise MissingCheckpointError(
            f"Evidence Rule 5 Violation: Required adapter checkpoint not found at '{cp}' "
            f"for condition '{condition}'. Silent fallback to uninitialized weights is strictly prohibited."
        )

    if cp.stat().st_size == 0:
        raise ValueError(f"Checkpoint at '{cp}' is 0 bytes (empty file).")

    hasher = hashlib.sha256()
    with open(cp, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    cp_sha = hasher.hexdigest()

    return cp, cp_sha


def atomic_serialize_prediction_artifact(
    artifact_data: Union[Dict[str, Any], List[Any]],
    output_path: Path,
) -> Tuple[Path, Path, str]:
    """Atomically serialize prediction artifact and compute SHA-256 sidecar.

    Uses temporary file with flush and fsync, then atomic replace.
    Emits both output_path and output_path.sha256.

    Returns:
        tuple of (output_path, sidecar_path, sha256_hash)
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_bytes = json.dumps(artifact_data, indent=2).encode("utf-8")
    sha256_hash = hashlib.sha256(json_bytes).hexdigest()

    # 1. Atomic write of prediction file
    tmp_path = output_path.with_name(f".{output_path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    with open(tmp_path, "wb") as f:
        f.write(json_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)

    # 2. Atomic write of sidecar file
    sidecar_path = output_path.with_name(f"{output_path.name}.sha256")
    tmp_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    sidecar_content = f"{sha256_hash}  {output_path.name}\n".encode("utf-8")
    with open(tmp_sidecar, "wb") as f:
        f.write(sidecar_content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_sidecar, sidecar_path)

    return output_path, sidecar_path, sha256_hash


def verify_prediction_file(
    prediction_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Verify SHA-256 of prediction file against sidecar before accessing answer keys.

    Operationalizes Evidence Rule 2:
    - Asserts prediction file exists and is a non-empty regular file.
    - Asserts mandatory .sha256 sidecar exists (Defect 2 remediation).
    - Asserts sidecar file is non-empty and contains valid 64-char hex tokens (Defect 5 remediation).
    - Asserts computed prediction file SHA-256 matches sidecar recorded SHA-256.
    - Asserts computed SHA-256 matches caller-specified expected_sha256 if provided.
    - Decodes UTF-8 and parses valid JSON root artifact.

    Args:
        prediction_path: Path to the JSON prediction artifact.
        expected_sha256: Optional expected SHA-256 digest to enforce.

    Returns:
        Tuple of (parsed_artifact_dict, computed_sha256_hash)

    Raises:
        FileNotFoundError: If the prediction file does not exist.
        PredictionIntegrityError: If sidecar is missing, empty, malformed, or checksum mismatches.
    """
    pred_path = Path(prediction_path).resolve()

    # 1. Prediction file validation
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not pred_path.is_file():
        raise PredictionIntegrityError(f"Prediction path is not a regular file: {pred_path}")

    actual_bytes = pred_path.read_bytes()
    if len(actual_bytes) == 0:
        raise PredictionIntegrityError(f"Prediction file is empty (0 bytes): '{pred_path.name}'.")

    actual_sha = hashlib.sha256(actual_bytes).hexdigest()

    # 2. Mandatory Sidecar Existence (Remediates Defect 2)
    sidecar_path = pred_path.with_name(f"{pred_path.name}.sha256")
    if not sidecar_path.exists():
        raise MissingSidecarError(
            f"Mandatory SHA-256 sidecar missing: {sidecar_path.name}"
        )
    if not sidecar_path.is_file():
        raise PredictionIntegrityError(
            f"Mandatory SHA-256 sidecar is not a regular file: '{sidecar_path.name}'."
        )

    # 3. Non-Empty Sidecar & Tokenization (Remediates Defect 5)
    if sidecar_path.stat().st_size == 0:
        raise EmptySidecarError(
            f"Empty or zero-byte SHA-256 sidecar file: '{sidecar_path.name}'."
        )

    sidecar_text = sidecar_path.read_text(encoding="utf-8").strip()
    if not sidecar_text:
        raise EmptySidecarError(
            f"Empty or zero-byte SHA-256 sidecar file: '{sidecar_path.name}'."
        )

    tokens = sidecar_text.split()
    if not tokens:
        raise EmptySidecarError(
            f"Empty or zero-byte SHA-256 sidecar file: '{sidecar_path.name}'."
        )

    recorded_sha = tokens[0].lower()

    # 4. Sidecar Syntax & Hex Validation
    if len(recorded_sha) != 64 or any(c not in "0123456789abcdef" for c in recorded_sha):
        raise MalformedSidecarError(
            f"Malformed SHA-256 sidecar file '{sidecar_path.name}': "
            f"expected 64-character hex hash, got '{tokens[0]}' (length {len(tokens[0])})."
        )

    # 5. Sidecar Filename Validation (if present)
    if len(tokens) > 1:
        recorded_filename = tokens[1].lstrip("*")
        if recorded_filename != pred_path.name:
            raise MalformedSidecarError(
                f"Sidecar filename mismatch in '{sidecar_path.name}': "
                f"sidecar specifies '{recorded_filename}', but verifying file '{pred_path.name}'."
            )

    # 6. Checksum Verification (Evidence Rule 2)
    if actual_sha != recorded_sha:
        raise ChecksumMismatchError(
            f"Prediction file tampering detected for '{pred_path.name}': "
            f"expected {recorded_sha}, computed {actual_sha}."
        )

    # 7. Caller-Supplied Expected Digest Validation
    if expected_sha256 and actual_sha != expected_sha256.lower():
        raise ChecksumMismatchError(
            f"Prediction file hash mismatch for '{pred_path.name}': "
            f"expected {expected_sha256}, computed {actual_sha}."
        )

    # 8. JSON Artifact Parsing & Root Validation
    try:
        artifact: Dict[str, Any] = json.loads(actual_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise PredictionIntegrityError(
            f"Corrupted prediction file '{pred_path.name}': invalid JSON content ({e})."
        ) from e

    if not isinstance(artifact, dict):
        raise PredictionIntegrityError(
            f"Prediction file '{pred_path.name}' root must be a JSON object, got {type(artifact).__name__}."
        )

    return artifact, actual_sha


def load_quarantined_answer_keys(keys_path: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Load answer keys into an ID-indexed lookup table post-verification."""
    keys_path = Path(keys_path).resolve()
    if not keys_path.exists():
        raise FileNotFoundError(f"Quarantined answer keys file not found: {keys_path}")

    raw_bytes = keys_path.read_bytes()
    keys_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    keys: Dict[str, Dict[str, Any]] = {}
    with open(keys_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                keys[rec["id"]] = rec

    return keys, keys_sha256


# ==============================================================================
# 6. Cross-Run Anti-Merge Guard
# ==============================================================================

RECURRENT_ADAPTER_CONDITIONS: frozenset[str] = frozenset({
    "adapter_t0",
    "adapter_t1",
    "adapter_t2",
    "adapter_t4",
    "adapter_recurrent",
    "control_zeroed",
    "control_shuffled",
    "control_random",
})

NON_RECURRENT_CONDITIONS: frozenset[str] = frozenset({
    "non_recurrent",
})

ALL_CHECKPOINT_CONDITIONS: frozenset[str] = (
    RECURRENT_ADAPTER_CONDITIONS | NON_RECURRENT_CONDITIONS
)


def is_valid_git_commit_sha(
    sha: Optional[str],
    allow_offline: bool = False,
) -> Tuple[bool, str]:
    """Validate git commit SHA format and presence.

    Returns:
        (is_valid, error_reason)
    """
    if sha is None or not str(sha).strip():
        return False, "git_commit_sha is empty or whitespace-only"

    clean_sha = str(sha).strip()
    if clean_sha.lower() in ("unknown", "<unknown>", "none", "null"):
        if allow_offline:
            return True, ""
        return (
            False,
            f"git_commit_sha is '{clean_sha}'; unversioned runs cannot be merged into summary artifacts without offline development flag (PRLR_ALLOW_OFFLINE_GIT=1)",
        )

    # In test environments, allow mock prefix 'commit_<hex/digits>'
    if clean_sha.startswith("commit_") and re.match(r"^commit_[0-9a-fA-F0-9]{4,64}$", clean_sha):
        return True, ""

    # In strict mode, commit must be 7 to 64 hexadecimal characters
    if not allow_offline:
        if not re.match(r"^[0-9a-fA-F]{7,64}$", clean_sha):
            return (
                False,
                f"git_commit_sha '{clean_sha}' has invalid format: must be 7-64 hexadecimal characters",
            )
        return True, ""

    # In offline mode, allow test fixture identifiers
    if not re.match(r"^[0-9a-zA-Z_\-\.]{4,64}$", clean_sha):
        return False, f"offline git_commit_sha '{clean_sha}' contains invalid characters"
    return True, ""


def validate_can_merge(
    existing: Union[ScoredSummaryArtifact, Dict[str, Any]],
    incoming_provenance: Union[ProvenanceMetadata, Dict[str, Any]],
    incoming_condition: str,
    incoming_checkpoint_sha256: str = "",
    incoming_sample_count: Optional[int] = None,
    allow_offline_git: bool = False,
) -> None:
    """Verifies that an incoming condition can safely be merged into an existing summary.

    Enforces Evidence Rules 1, 2, 5, and 10:
    - Rejects empty, whitespace-only, or 'unknown' git_commit_sha (unless offline flag is active).
    - Requires strict commit format compliance (7-64 hex characters).
    - Enforces commit SHA and dirty status matching between existing and incoming runs.
    - Enforces dataset SHA and sample count parity.
    - Enforces checkpoint SHA-256 matching for ALL conditions using checkpoints:
      recurrent adapters (adapter_t*, control_zeroed, control_shuffled, control_random)
      and parameter-matched non-recurrent adapter (non_recurrent).
    - Enforces MLX, Python, and Transformers runtime version parity.

    Raises:
        CrossRunMergeRejectionError: If any cross-run incompatibility is detected.
    """
    mismatches: List[str] = []

    # Check offline development mode from flag or environment variable
    offline_mode = (
        allow_offline_git
        or os.environ.get("PRLR_ALLOW_OFFLINE_GIT", "").strip().lower() in ("1", "true", "yes")
        or os.environ.get("PRLR_DEV_MODE", "").strip().lower() in ("1", "true", "yes")
    )

    # Extract existing provenance
    if isinstance(existing, ScoredSummaryArtifact):
        ex_prov = existing.provenance
        ex_conds = existing.conditions
    elif isinstance(existing, dict):
        ex_meta = existing.get("provenance", existing.get("metadata", {}))
        ex_prov = ProvenanceMetadata(
            git_commit_sha=str(ex_meta.get("git_commit_sha", ex_meta.get("git_commit", ""))),
            is_dirty=bool(ex_meta.get("is_dirty", False)),
            dataset_name=str(ex_meta.get("dataset_name", "")),
            dataset_sha256=str(ex_meta.get("dataset_sha256", "")),
            split=str(ex_meta.get("split", "")),
            sample_count=int(ex_meta.get("sample_count", 0)),
            inputs_file=str(ex_meta.get("inputs_file", "")),
            inputs_file_sha256=str(ex_meta.get("inputs_file_sha256", "")),
            keys_file=str(ex_meta.get("keys_file", "")),
            keys_file_sha256=str(ex_meta.get("keys_file_sha256", "")),
            runtime_versions=dict(ex_meta.get("runtime_versions", ex_meta.get("runtime", {}))),
            hardware_info=dict(ex_meta.get("hardware_info", {})),
        )
        ex_conds = existing.get("conditions", {})
    else:
        raise TypeError(f"Unsupported existing summary type: {type(existing)}")

    # Extract incoming provenance
    if isinstance(incoming_provenance, ProvenanceMetadata):
        in_prov = incoming_provenance
    elif isinstance(incoming_provenance, dict):
        in_prov = ProvenanceMetadata(
            git_commit_sha=str(incoming_provenance.get("git_commit_sha", incoming_provenance.get("git_commit", ""))),
            is_dirty=bool(incoming_provenance.get("is_dirty", False)),
            dataset_name=str(incoming_provenance.get("dataset_name", "")),
            dataset_sha256=str(incoming_provenance.get("dataset_sha256", "")),
            split=str(incoming_provenance.get("split", "")),
            sample_count=int(incoming_provenance.get("sample_count", 0)),
            inputs_file=str(incoming_provenance.get("inputs_file", "")),
            inputs_file_sha256=str(incoming_provenance.get("inputs_file_sha256", "")),
            keys_file=str(incoming_provenance.get("keys_file", "")),
            keys_file_sha256=str(incoming_provenance.get("keys_file_sha256", "")),
            runtime_versions=dict(incoming_provenance.get("runtime_versions", incoming_provenance.get("runtime", {}))),
            hardware_info=dict(incoming_provenance.get("hardware_info", {})),
        )
    else:
        raise TypeError(f"Unsupported incoming provenance type: {type(incoming_provenance)}")

    # 1. Commit check (Defect 6 remediation)
    in_valid, in_err = is_valid_git_commit_sha(in_prov.git_commit_sha, allow_offline=offline_mode)
    if not in_valid:
        mismatches.append(f"Incoming {in_err}")

    ex_valid, ex_err = is_valid_git_commit_sha(ex_prov.git_commit_sha, allow_offline=offline_mode)
    if not ex_valid:
        mismatches.append(f"Existing {ex_err}")

    if in_valid and ex_valid:
        if not offline_mode and ex_prov.git_commit_sha.strip() != in_prov.git_commit_sha.strip():
            mismatches.append(
                f"git_commit_sha mismatch: existing '{ex_prov.git_commit_sha.strip()}' != incoming '{in_prov.git_commit_sha.strip()}'"
            )

    if not offline_mode and ex_prov.is_dirty != in_prov.is_dirty:
        mismatches.append(
            f"is_dirty mismatch: existing {ex_prov.is_dirty} != incoming {in_prov.is_dirty}"
        )

    # 2. Dataset input hash check
    ex_d_hash = ex_prov.inputs_file_sha256 or ex_prov.dataset_sha256
    in_d_hash = in_prov.inputs_file_sha256 or in_prov.dataset_sha256
    if ex_d_hash and in_d_hash and ex_d_hash != in_d_hash:
        mismatches.append(
            f"dataset/inputs hash mismatch: existing '{ex_d_hash}' != incoming '{in_d_hash}'"
        )

    # 3. Sample count check
    in_cnt = incoming_sample_count if incoming_sample_count is not None else in_prov.sample_count
    if ex_prov.sample_count and in_cnt and ex_prov.sample_count != in_cnt:
        mismatches.append(
            f"sample_count mismatch: existing {ex_prov.sample_count} != incoming {in_cnt}"
        )

    # 4. Checkpoint hash check for all checkpoint conditions (Defect 4 remediation)
    clean_in_cp_sha = str(incoming_checkpoint_sha256).strip() if incoming_checkpoint_sha256 else ""

    is_recurrent_cond = (
        incoming_condition in RECURRENT_ADAPTER_CONDITIONS
        or incoming_condition.startswith("adapter_t")
        or incoming_condition.startswith("adapter_")
    )
    is_non_recurrent_cond = (
        incoming_condition in NON_RECURRENT_CONDITIONS
        or incoming_condition.startswith("non_recurrent")
    )

    if is_recurrent_cond or is_non_recurrent_cond:
        if not clean_in_cp_sha:
            mismatches.append(
                f"Missing required checkpoint_sha256 for checkpoint-dependent condition '{incoming_condition}'"
            )

    # 4a. Checkpoint hash validation for recurrent adapter family
    if is_recurrent_cond and clean_in_cp_sha:
        for cond_name, cond_summary in ex_conds.items():
            cond_is_rec = (
                cond_name in RECURRENT_ADAPTER_CONDITIONS
                or cond_name.startswith("adapter_t")
                or cond_name.startswith("adapter_")
            )
            if cond_is_rec:
                ex_cp_sha = getattr(cond_summary, "checkpoint_sha256", None)
                if ex_cp_sha is None and isinstance(cond_summary, dict):
                    ex_cp_sha = cond_summary.get("checkpoint_sha256")
                ex_cp_sha_clean = str(ex_cp_sha).strip() if ex_cp_sha else ""
                if ex_cp_sha_clean and clean_in_cp_sha and ex_cp_sha_clean != clean_in_cp_sha:
                    mismatches.append(
                        f"checkpoint_sha256 mismatch for recurrent adapter condition '{incoming_condition}': "
                        f"existing '{ex_cp_sha_clean}' ({cond_name}) != incoming '{clean_in_cp_sha}'"
                    )

    # 4b. Checkpoint hash validation for non-recurrent adapter family
    if is_non_recurrent_cond and clean_in_cp_sha:
        for cond_name, cond_summary in ex_conds.items():
            cond_is_non_rec = (
                cond_name in NON_RECURRENT_CONDITIONS
                or cond_name.startswith("non_recurrent")
            )
            if cond_is_non_rec:
                ex_cp_sha = getattr(cond_summary, "checkpoint_sha256", None)
                if ex_cp_sha is None and isinstance(cond_summary, dict):
                    ex_cp_sha = cond_summary.get("checkpoint_sha256")
                ex_cp_sha_clean = str(ex_cp_sha).strip() if ex_cp_sha else ""
                if ex_cp_sha_clean and clean_in_cp_sha and ex_cp_sha_clean != clean_in_cp_sha:
                    mismatches.append(
                        f"checkpoint_sha256 mismatch for non-recurrent condition '{incoming_condition}': "
                        f"existing '{ex_cp_sha_clean}' ({cond_name}) != incoming '{clean_in_cp_sha}'"
                    )

    # 5. Runtime version checks
    for pkg in ("python", "mlx", "transformers"):
        ex_v = ex_prov.runtime_versions.get(pkg)
        in_v = in_prov.runtime_versions.get(pkg)
        if ex_v and in_v and ex_v != in_v:
            mismatches.append(f"Runtime version mismatch for '{pkg}': existing '{ex_v}' != incoming '{in_v}'")

    if mismatches:
        err_msg = (
            f"Refusing to merge condition '{incoming_condition}' into summary artifact! "
            f"Detected {len(mismatches)} cross-run incompatibilities:\n - " + "\n - ".join(mismatches)
        )
        raise CrossRunMergeRejectionError(err_msg)


def safe_merge_condition_summary(
    summary_path: Path,
    new_condition_summary: Dict[str, Any],
    incoming_provenance: Optional[ProvenanceMetadata] = None,
) -> Dict[str, Any]:
    """Merge condition summary into composite summary under strict parity guards."""
    summary_path = Path(summary_path).resolve()
    composite: Dict[str, Any] = {}
    if summary_path.exists():
        composite = json.loads(summary_path.read_text(encoding="utf-8"))

    cond_name = new_condition_summary["condition"]
    cond_metrics = new_condition_summary.get("summary", new_condition_summary.get("metrics", new_condition_summary))
    prov_data = incoming_provenance.to_dict() if incoming_provenance else new_condition_summary.get("metadata", {})

    if composite and ("provenance" in composite or "metadata" in composite):
        validate_can_merge(
            existing=composite,
            incoming_provenance=prov_data,
            incoming_condition=cond_name,
            incoming_checkpoint_sha256=new_condition_summary.get("checkpoint_sha256", ""),
            incoming_sample_count=new_condition_summary.get("sample_count"),
        )
    else:
        composite = {
            "schema_version": "prlr.empirical_summary.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": prov_data,
            "prediction_files_sha256": {},
            "conditions": {},
        }

    composite.setdefault("conditions", {})[cond_name] = cond_metrics
    composite["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    atomic_serialize_prediction_artifact(composite, summary_path)
    return composite


# ==============================================================================
# 7. Generation Mechanics & Direct Frozen Token 106 Halting
# ==============================================================================

def generate_direct_frozen(
    backbone: PretrainedGemmaBackbone,
    canonical_prompt: str,
    max_tokens: int = 96,
) -> Tuple[List[int], str, float]:
    """Execute direct frozen Gemma 4 generation with guaranteed token 106 halting.

    Halts on turn delimiter token 106 (<turn|>) and EOS token 1 (<eos>).
    Returns:
        tuple of (generated_token_ids, decoded_text, latency_ms)
    """
    import mlx_lm

    tokenizer = backbone.tokenizer
    is_g4 = is_gemma4_tokenizer(tokenizer)
    stop_ids = {1, 106} if is_g4 else {1, 107}

    if isinstance(tokenizer, TokenizerWrapper):
        tokenizer.eos_token_ids = stop_ids
        wrapped_tok = tokenizer
    else:
        wrapped_tok = TokenizerWrapper(tokenizer, eos_token_ids=stop_ids)

    t0 = time.perf_counter()
    tokens: List[int] = []
    text_segments: List[str] = []

    for response in mlx_lm.stream_generate(
        model=backbone.model,
        tokenizer=wrapped_tok,
        prompt=canonical_prompt,
        max_tokens=max_tokens,
    ):
        tokens.append(response.token)
        text_segments.append(response.text)

    lat_ms = (time.perf_counter() - t0) * 1000.0
    decoded_text = "".join(text_segments)

    return tokens, decoded_text, lat_ms


# ==============================================================================
# 8. Stage 1: Target-Free generate_predictions()
# ==============================================================================

def load_adapter_and_injection_weights(
    checkpoint_path: str | Path,
    adapter: Any,
    decoder: Optional[Any] = None,
) -> None:
    """Load weights into adapter and optional decoder safe_injection, supporting prefixed and flat checkpoints."""
    loaded = mx.load(str(checkpoint_path))
    has_adapter_prefix = any(k.startswith("adapter.") for k in loaded.keys())
    if has_adapter_prefix:
        adapter_weights = {}
        injection_weights = {}
        for k, v in loaded.items():
            if k.startswith("adapter."):
                adapter_weights[k[len("adapter."):]] = v
            elif k.startswith("safe_injection."):
                injection_weights[k[len("safe_injection."):]] = v
            elif not k.startswith("injection."):
                adapter_weights[k] = v

        if adapter_weights and adapter is not None and hasattr(adapter, "parameters"):
            adapter_param_keys = set(dict(tree_flatten(adapter.parameters())).keys())
            filtered_adapter = {k: v for k, v in adapter_weights.items() if k in adapter_param_keys}
            adapter.load_weights(list(filtered_adapter.items()))
        elif adapter_weights and adapter is not None:
            adapter.load_weights(list(adapter_weights.items()))

        if injection_weights and decoder is not None and hasattr(decoder, "safe_injection"):
            if "q_proj.weight" in injection_weights and hasattr(decoder.safe_injection, "q_proj"):
                ckpt_proj_dim = injection_weights["q_proj.weight"].shape[0]
                curr_proj_dim = decoder.safe_injection.q_proj.weight.shape[0]
                if ckpt_proj_dim != curr_proj_dim:
                    head_dim = decoder.safe_injection.head_dim
                    new_num_heads = ckpt_proj_dim // head_dim
                    decoder.safe_injection = GatedCrossAttentionInjection(
                        hidden_size=decoder.hidden_dim,
                        num_heads=new_num_heads,
                        head_dim=head_dim,
                        gamma_max=decoder.safe_injection.gamma_max,
                    )
                    decoder.injection = decoder.safe_injection
            inj_param_keys = set(dict(tree_flatten(decoder.safe_injection.parameters())).keys())
            filtered_inj = {k: v for k, v in injection_weights.items() if k in inj_param_keys}
            decoder.safe_injection.load_weights(list(filtered_inj.items()))
    else:
        adapter.load_weights(str(checkpoint_path))


def generate_predictions(
    inputs: Union[Sequence[EvaluationInput], Path],
    condition: str,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
    backbone: Optional[PretrainedGemmaBackbone] = None,
    adapter: Optional[Any] = None,
    decoder: Optional[GemmaCausalPrefixDecoder] = None,
    max_tokens: int = 96,
    limit: Optional[int] = None,
    seed: int = 42,
    conditioning_mode: str = "cross_attention",
) -> Tuple[Path, Path, str]:
    """Generate model predictions consuming strictly target-free EvaluationInputs.

    Guarantees:
    - Evidence Rule 1: Throws OracleLeakageError if oracle keys or answer keys are passed.
    - Evidence Rule 5: Throws MissingCheckpointError if checkpoint is missing (no random fallback).
    - Milestone 1: Halts on token 106 (<turn|>).
    - Metal Latency: mx.eval() synchronization before measuring latency.
    - Evidence Rule 10: Atomic JSON serialization with SHA-256 sidecar.

    Returns:
        tuple of (prediction_file_path, sidecar_file_path, sha256_hash)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    # 1. Ingestion and target-free validation
    inputs_sha256 = ""
    inputs_file_str = "in_memory"
    if isinstance(inputs, Path):
        inputs_file_str = str(inputs)
        eval_inputs, inputs_sha256 = load_target_free_inputs(inputs, limit=limit)
    elif isinstance(inputs, Sequence):
        eval_inputs = []
        for idx, item in enumerate(inputs):
            validate_target_free_record(item)
            if isinstance(item, EvaluationInput):
                eval_inputs.append(item)
            elif isinstance(item, dict):
                eval_inputs.append(EvaluationInput.from_dict(item))
            else:
                raise TypeError(
                    f"Inputs sequence must contain EvaluationInput or dict, got {type(item).__name__} at index {idx}"
                )
        if limit:
            eval_inputs = eval_inputs[:limit]
    else:
        raise TypeError(f"Expected inputs to be Path or Sequence[EvaluationInput], got {type(inputs).__name__}")

    # 2. Checkpoint guard for adapter conditions
    adapter_conditions = set(ALL_CHECKPOINT_CONDITIONS)
    cp_sha256 = ""
    if condition in adapter_conditions:
        _, cp_sha256 = verify_adapter_checkpoint(checkpoint_path, condition)

    # 3. Model setup if not injected
    if backbone is None:
        manifest = ModelManifest.gemma_4_12b_it()
        backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
        backbone.freeze()

    tokenizer = backbone.tokenizer
    is_g4 = is_gemma4_tokenizer(tokenizer)
    stop_ids = {1, 106} if is_g4 else {1, 107}
    if hasattr(tokenizer, "eos_token_ids"):
        try:
            tokenizer.eos_token_ids = stop_ids
        except Exception:
            pass

    if condition != "direct_frozen" and decoder is None:
        decoder = GemmaCausalPrefixDecoder(
            backbone=backbone,
            prefix_dim=3840,
            hidden_dim=3840,
            conditioning_mode=conditioning_mode,
        )

    if condition in adapter_conditions and adapter is None:
        if condition == "non_recurrent":
            adapter = GemmaNonRecurrentAdapter(dim=3840, num_slots=16, intermediate_dim=13440)
        else:
            adapter = GemmaRecurrentAdapter(dim=3840, num_slots=16, num_layers=1, deliberation_steps=4)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            load_adapter_and_injection_weights(checkpoint_path, adapter, decoder=decoder)

    # Provenance metadata
    git_meta = get_git_metadata()
    hw_meta = get_hardware_metadata()
    rt_versions = get_runtime_versions()
    model_id = getattr(backbone.manifest, "model_id", "google/gemma-4-12B-it-4bit")
    tok_id = getattr(backbone.manifest, "tokenizer_id", "google/gemma-4-12B-it")

    # 4. Generation Loop
    records: List[PredictionRecord] = []
    recurrence_depth = 0
    if "_t" in condition:
        try:
            recurrence_depth = int(condition.split("_t")[-1])
        except ValueError:
            recurrence_depth = 0
    elif condition in ("adapter_recurrent", "control_zeroed", "control_shuffled", "control_random"):
        recurrence_depth = 4
    elif condition == "non_recurrent":
        recurrence_depth = 1

    for item in eval_inputs:
        prompt_raw = item.prompt
        user_body = extract_user_body(prompt_raw)
        canonical_prompt = format_canonical_prompt(user_body, tokenizer, is_gemma4=is_g4)
        prompt_ids, _ = backbone.encode_prompt_context(canonical_prompt)
        mx.eval(prompt_ids)

        t0 = time.perf_counter()

        if condition == "direct_frozen":
            gen_tokens, decoded_text, lat_ms = generate_direct_frozen(
                backbone=backbone,
                canonical_prompt=canonical_prompt,
                max_tokens=max_tokens,
            )
        elif condition == "repo_decoder":
            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=None,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        elif condition.startswith("adapter_t") or condition == "adapter_recurrent":
            t_steps = 4 if condition == "adapter_recurrent" else int(condition.split("_t")[-1])
            h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
            if t_steps == 0:
                slots = adapter.prelude(h_prompt)
            else:
                slots = adapter(h_prompt, steps=t_steps)
            mx.eval(slots)
            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        elif condition == "control_zeroed":
            h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
            slots = adapter(h_prompt, steps=4)
            slots = mx.zeros_like(slots)
            mx.eval(slots)
            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        elif condition == "control_random":
            # Random nonzero injection control with magnitude matched to learned latent norm
            h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
            learned_slots = adapter(h_prompt, steps=4)
            mx.eval(learned_slots)

            # Compute per-slot L2 norm: shape (B, M, 1)
            slot_norms = mx.sqrt(mx.sum(learned_slots ** 2, axis=-1, keepdims=True) + 1e-8)

            # Deterministic isotropic Gaussian noise per sample
            sample_seed = (abs(hash(item.id)) ^ seed) & 0xFFFFFFFF
            rng_noise = np.random.default_rng(sample_seed).standard_normal(learned_slots.shape)
            noise_arr = mx.array(rng_noise, dtype=learned_slots.dtype)
            noise_norms = mx.sqrt(mx.sum(noise_arr ** 2, axis=-1, keepdims=True) + 1e-8)

            # Directional normalization and exact magnitude matching
            slots = (noise_arr / noise_norms) * slot_norms
            mx.eval(slots)

            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        elif condition == "control_shuffled":
            # Shuffled-slot control using fixed canonical order-reversal derangement
            h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
            slots = adapter(h_prompt, steps=4)
            perm = mx.array(FIXED_SLOT_PERMUTATION)
            slots = slots[:, perm, :]
            mx.eval(slots)
            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        elif condition == "non_recurrent":
            # Genuinely non-recurrent parameter-matched adapter single-pass deliberation
            h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
            slots = adapter(h_prompt)
            mx.eval(slots)
            gen_tokens_arr = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            mx.eval(gen_tokens_arr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tok_list = gen_tokens_arr[0].tolist() if gen_tokens_arr.ndim > 1 else gen_tokens_arr.tolist()
            gen_tokens = tok_list
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)
        else:
            raise ValueError(f"Unknown condition: '{condition}'")

        rec = PredictionRecord(
            sample_id=item.id,
            domain=item.domain,
            condition=condition,
            recurrence_depth=recurrence_depth,
            generated_token_ids=tuple(gen_tokens),
            decoded_text=decoded_text,
            latency_ms=round(lat_ms, 2),
            prompt_sha256=item.prompt_sha256,
            git_commit_sha=git_meta["git_commit_sha"],
            is_dirty=git_meta["is_dirty"],
            checkpoint_sha256=cp_sha256,
            model_id=str(model_id),
            tokenizer_id=str(tok_id),
            hardware_info=hw_meta,
            runtime_versions=rt_versions,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        records.append(rec)

    # 5. Atomic Serialization
    meta_dict = {
        "condition": condition,
        "recurrence_depth": recurrence_depth,
        "sample_count": len(records),
        "git_commit_sha": git_meta["git_commit_sha"],
        "is_dirty": git_meta["is_dirty"],
        "checkpoint_sha256": cp_sha256,
        "model_id": str(model_id),
        "tokenizer_id": str(tok_id),
        "inputs_file": inputs_file_str,
        "inputs_file_sha256": inputs_sha256,
        "hardware_info": hw_meta,
        "runtime_versions": rt_versions,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if condition == "control_shuffled":
        meta_dict["slot_permutation"] = list(FIXED_SLOT_PERMUTATION)
        meta_dict["permutation_type"] = "canonical_derangement_inversion"
    elif condition == "control_random":
        meta_dict["noise_distribution"] = "isotropic_gaussian"
        meta_dict["norm_matching"] = "l2_slot_matched"

    artifact = {
        "schema_version": "prlr.predictions.v1",
        "metadata": meta_dict,
        "predictions": [r.to_dict() for r in records],
    }

    pred_path = output_dir / f"predictions_{condition}.json"
    return atomic_serialize_prediction_artifact(artifact, pred_path)



# ==============================================================================
# 9. Stage 2: Post-Hoc score_predictions()
# ==============================================================================

def score_predictions(
    predictions_path: Path,
    answer_keys_path: Path,
    output_dir: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    verifier: Optional[ProceduralVerifier] = None,
    inputs_file_sha256: str = "",
) -> Tuple[ScoredSummaryArtifact, Path]:
    """Score model predictions against quarantined answer keys post-hoc.

    Guarantees:
    - Evidence Rule 2: Handshake verifies prediction SHA-256 BEFORE opening answer keys.
    - Sample Alignment: Confirms 1-to-1 match on sample_id and prompt_sha256.
    - Anti-Merge: Refuses to merge summaries across mismatched runs.

    Returns:
        tuple of (ScoredSummaryArtifact, summary_file_path)
    """
    pred_path = Path(predictions_path).resolve()
    keys_path = Path(answer_keys_path).resolve()

    # 1. Pre-Scoring Handshake: verify prediction file integrity BEFORE touching answer keys
    artifact, actual_pred_sha = verify_prediction_file(pred_path)

    # 2. Post-handshake: Load quarantined answer keys
    keys_dict, keys_sha256 = load_quarantined_answer_keys(keys_path)

    # 3. Alignment verification & Scoring
    if verifier is None:
        verifier = ProceduralVerifier()

    meta = artifact.get("metadata", {})
    condition = meta.get("condition", "unknown")
    recurrence_depth = int(meta.get("recurrence_depth", 0))
    raw_predictions = artifact.get("predictions", [])

    exact_matches: List[int] = []
    terminal_matches: List[int] = []
    valid_jsons: List[int] = []
    repetitions: List[int] = []
    entropies: List[float] = []
    latencies: List[float] = []

    for pred in raw_predictions:
        sid = pred["sample_id"]
        if sid not in keys_dict:
            raise SampleMismatchError(
                f"Sample ID '{sid}' from predictions not found in quarantined answer keys '{keys_path.name}'!"
            )

        key_rec = keys_dict[sid]

        # Verify prompt hash alignment
        key_p_sha = key_rec.get("prompt_sha256")
        pred_p_sha = pred.get("prompt_sha256")
        if key_p_sha and pred_p_sha and key_p_sha != pred_p_sha:
            raise SampleMismatchError(
                f"Prompt SHA-256 mismatch for sample '{sid}': key '{key_p_sha}' != pred '{pred_p_sha}'!"
            )

        v_cfg = key_rec.get("verifier_config", {})
        expected_route = v_cfg.get("expected_route", [])
        expected_terminal = v_cfg.get("terminal_tool")
        goal = v_cfg.get("target_goal")
        tools = DOMAIN_CATALOGUES.get(pred.get("domain", ""))

        text = pred.get("decoded_text", pred.get("generated_text", ""))
        v_res = verifier.verify(text, tuple(expected_route), tools=tools, goal=goal)

        is_em = bool(v_res.get("exact_match", False))
        pred_term = v_res.get("terminal_tool")
        is_term = bool(pred_term and expected_terminal and pred_term == expected_terminal)
        is_valid = bool(v_res.get("is_valid", False))
        rep = compute_max_ngram_repetition(text, n=4)
        entropy = compute_shannon_entropy(text)

        exact_matches.append(1 if is_em else 0)
        terminal_matches.append(1 if is_term else 0)
        valid_jsons.append(1 if is_valid else 0)
        repetitions.append(rep)
        entropies.append(entropy)
        latencies.append(float(pred.get("latency_ms", 0.0)))

    n_samples = len(raw_predictions)
    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    latency_stats = LatencyStatistics(
        mean_ms=round(float(np.mean(lat_arr)), 2),
        median_ms=round(float(np.median(lat_arr)), 2),
        p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
        min_ms=round(float(np.min(lat_arr)), 2),
        max_ms=round(float(np.max(lat_arr)), 2),
        total_ms=round(float(np.sum(lat_arr)), 2),
    )

    cond_summary = ConditionScoredSummary(
        condition=condition,
        recurrence_depth=recurrence_depth,
        sample_count=n_samples,
        exact_match_count=sum(exact_matches),
        exact_match_pct=round(float(np.mean(exact_matches)) * 100.0, 2) if exact_matches else 0.0,
        terminal_match_count=sum(terminal_matches),
        terminal_match_pct=round(float(np.mean(terminal_matches)) * 100.0, 2) if terminal_matches else 0.0,
        valid_json_count=sum(valid_jsons),
        valid_json_pct=round(float(np.mean(valid_jsons)) * 100.0, 2) if valid_jsons else 0.0,
        max_4gram_repetition=int(max(repetitions)) if repetitions else 0,
        mean_4gram_repetition=round(float(np.mean(repetitions)), 2) if repetitions else 0.0,
        mean_shannon_entropy=round(float(np.mean(entropies)), 2) if entropies else 0.0,
        latency=latency_stats,
        checkpoint_sha256=meta.get("checkpoint_sha256", ""),
        model_id=meta.get("model_id", ""),
    )

    provenance = ProvenanceMetadata(
        git_commit_sha=meta.get("git_commit_sha", ""),
        is_dirty=bool(meta.get("is_dirty", False)),
        dataset_name="prlr_domain_v1",
        dataset_sha256=meta.get("inputs_file_sha256", inputs_file_sha256),
        split="sealed_test",
        sample_count=n_samples,
        inputs_file=meta.get("inputs_file", ""),
        inputs_file_sha256=meta.get("inputs_file_sha256", inputs_file_sha256),
        keys_file=str(keys_path),
        keys_file_sha256=keys_sha256,
        runtime_versions=dict(meta.get("runtime_versions", {})),
        hardware_info=dict(meta.get("hardware_info", {})),
    )

    # 4. Summary consolidation under CrossRunMergeGuard
    if summary_path is None:
        target_dir = output_dir or pred_path.parent
        summary_path = target_dir / "empirical_baselines_summary.json"
    else:
        summary_path = Path(summary_path).resolve()

    composite_conditions: Dict[str, ConditionScoredSummary] = {}
    composite_shas: Dict[str, str] = {}

    if summary_path.exists():
        existing_data = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_can_merge(
            existing=existing_data,
            incoming_provenance=provenance,
            incoming_condition=condition,
            incoming_checkpoint_sha256=meta.get("checkpoint_sha256", ""),
            incoming_sample_count=n_samples,
        )
        existing_summary = ScoredSummaryArtifact.from_dict(existing_data)
        composite_conditions = dict(existing_summary.conditions)
        composite_shas = dict(existing_summary.prediction_files_sha256)

    composite_conditions[condition] = cond_summary
    composite_shas[condition] = actual_pred_sha

    summary_artifact = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,
        prediction_files_sha256=composite_shas,
        conditions=composite_conditions,
    )

    atomic_serialize_prediction_artifact(summary_artifact.to_dict(), summary_path)
    return summary_artifact, summary_path


__all__ = [
    "HarnessError",
    "OracleLeakageError",
    "TargetLeakageException",
    "Rule1ViolationError",
    "MissingCheckpointError",
    "CheckpointMissingError",
    "PredictionIntegrityError",
    "IntegrityError",
    "TamperedPredictionError",
    "MissingSidecarError",
    "EmptySidecarError",
    "MalformedSidecarError",
    "ChecksumMismatchError",
    "CrossRunMergeRejectionError",
    "SummaryMergeConflictError",
    "SampleMismatchError",
    "FORBIDDEN_ORACLE_TERMS",
    "SQUASHED_ORACLE_TERMS",
    "FORBIDDEN_ROOT_TOKENS",
    "FORBIDDEN_SQUASHED_SUBSTRINGS",
    "FORBIDDEN_COMPOUND_PHRASES",
    "FORBIDDEN_KEY_SUBSTRINGS",
    "LEGITIMATE_SOLUTION_WORDS",
    "split_camel_case",
    "normalize_oracle_key",
    "is_forbidden_oracle_key",
    "validate_target_free_dict",
    "validate_target_free_record",
    "EvaluationInput",
    "PredictionRecord",
    "LatencyStatistics",
    "ConditionScoredSummary",
    "ProvenanceMetadata",
    "ScoredSummaryArtifact",
    "get_git_metadata",
    "get_hardware_metadata",
    "get_runtime_versions",
    "load_target_free_inputs",
    "verify_adapter_checkpoint",
    "atomic_serialize_prediction_artifact",
    "verify_prediction_file",
    "load_quarantined_answer_keys",
    "RECURRENT_ADAPTER_CONDITIONS",
    "NON_RECURRENT_CONDITIONS",
    "ALL_CHECKPOINT_CONDITIONS",
    "is_valid_git_commit_sha",
    "validate_can_merge",
    "safe_merge_condition_summary",
    "generate_direct_frozen",
    "generate_predictions",
    "score_predictions",
    "FIXED_SLOT_PERMUTATION",
]
