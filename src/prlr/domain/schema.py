"""Domain dataset schema and sample dataclasses conforming to prlr.domain.v1.

Enforces:
- Strict view decoupling between evaluation inputs (0 ground truth) and answer keys
- Cryptographic SHA-256 fingerprinting of prompts and solutions
- Explicit 5-way split definitions (train, dev, sealed_test, sealed_gate, extrapolation)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

SplitType = Literal["train", "dev", "sealed_test", "sealed_gate", "extrapolation"]


@dataclass(frozen=True)
class EvaluationInput:
    """Isolated problem input for model evaluation. Strictly zero ground-truth (Rule 1)."""
    id: str
    split: SplitType
    domain: str
    difficulty: int
    num_steps: int
    prompt: str
    prompt_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationInput:
        return cls(**data)


@dataclass(frozen=True)
class AnswerKey:
    """Isolated answer key for post-hoc verifier evaluation (Rule 2)."""
    id: str
    split: SplitType
    domain: str
    target_solution: str
    ground_truth: str
    verifier_type: str
    verifier_config: Dict[str, Any]
    solution_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnswerKey:
        return cls(**data)


@dataclass(frozen=True)
class DomainSample:
    """Immutable domain problem sample conforming to prlr.domain.v1 schema."""
    id: str
    split: SplitType
    domain: str
    difficulty: int
    num_steps: int
    prompt: str
    target_solution: str
    ground_truth: str
    verifier_type: str
    verifier_config: Dict[str, Any]
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_sha256: str = ""
    solution_sha256: str = ""

    def __post_init__(self):
        if not self.prompt_sha256:
            object.__setattr__(
                self,
                "prompt_sha256",
                hashlib.sha256(self.prompt.strip().encode("utf-8")).hexdigest(),
            )
        if not self.solution_sha256:
            object.__setattr__(
                self,
                "solution_sha256",
                hashlib.sha256(self.target_solution.strip().encode("utf-8")).hexdigest(),
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainSample:
        return cls(**data)

    def to_evaluation_input(self) -> EvaluationInput:
        """Extracts strictly isolated input containing NO ground truth (Rule 1)."""
        return EvaluationInput(
            id=self.id,
            split=self.split,
            domain=self.domain,
            difficulty=self.difficulty,
            num_steps=self.num_steps,
            prompt=self.prompt,
            prompt_sha256=self.prompt_sha256,
        )

    def to_answer_key(self) -> AnswerKey:
        """Extracts isolated answer key for post-generation verification (Rule 2)."""
        return AnswerKey(
            id=self.id,
            split=self.split,
            domain=self.domain,
            target_solution=self.target_solution,
            ground_truth=self.ground_truth,
            verifier_type=self.verifier_type,
            verifier_config=self.verifier_config,
            solution_sha256=self.solution_sha256,
        )


@dataclass
class DatasetSplits:
    """Container holding in-memory lists of DomainSample across all 5 splits."""
    train: List[DomainSample]
    dev: List[DomainSample]
    sealed_test: List[DomainSample]
    sealed_gate: List[DomainSample]
    extrapolation: List[DomainSample]

    def get_split(self, name: SplitType) -> List[DomainSample]:
        if name == "train":
            return self.train
        elif name == "dev":
            return self.dev
        elif name == "sealed_test":
            return self.sealed_test
        elif name == "sealed_gate":
            return self.sealed_gate
        elif name == "extrapolation":
            return self.extrapolation
        else:
            raise KeyError(f"Unknown split name: {name}")

    def total_count(self) -> int:
        return (
            len(self.train)
            + len(self.dev)
            + len(self.sealed_test)
            + len(self.sealed_gate)
            + len(self.extrapolation)
        )


@dataclass(frozen=True)
class SplitManifestEntry:
    """Metadata and hashes for an individual dataset split on disk."""
    file_name: str
    sample_count: int
    byte_size: int
    sha256: str
    base_seed: int
    inputs_file: str
    inputs_sha256: str
    keys_file: str
    keys_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplitManifestEntry:
        return cls(**data)


@dataclass(frozen=True)
class DatasetManifest:
    """Cryptographic manifest tracking immutable dataset files and audit status."""
    schema_version: str
    created_at_utc: str
    domain_name: str
    source_commit: str
    total_samples: int
    splits: Dict[str, SplitManifestEntry]
    contamination_status: str
    audit_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "domain_name": self.domain_name,
            "source_commit": self.source_commit,
            "total_samples": self.total_samples,
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
            "contamination_status": self.contamination_status,
            "audit_metrics": self.audit_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetManifest:
        splits = {k: SplitManifestEntry.from_dict(v) for k, v in data["splits"].items()}
        return cls(
            schema_version=data["schema_version"],
            created_at_utc=data["created_at_utc"],
            domain_name=data["domain_name"],
            source_commit=data["source_commit"],
            total_samples=data["total_samples"],
            splits=splits,
            contamination_status=data["contamination_status"],
            audit_metrics=data["audit_metrics"],
        )


__all__ = [
    "SplitType",
    "EvaluationInput",
    "AnswerKey",
    "DomainSample",
    "DatasetSplits",
    "SplitManifestEntry",
    "DatasetManifest",
]
