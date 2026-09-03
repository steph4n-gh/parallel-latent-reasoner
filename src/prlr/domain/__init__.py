"""PRLR Procedural Domain Package.

Provides:
- MTR-DAG procedural reasoning domain with deterministic BFS oracle
- Strict 5-way partition schema, sample definitions, and cryptographic manifest
- 4-tier contamination defense and leakage prevention
- MLX-native dataset loader and batching
"""

from prlr.domain.contamination import (
    ContaminationError,
    JaccardOverlapContaminationError,
    KeyLeakageContaminationError,
    PromptCollisionContaminationError,
    canonicalize_prompt,
    check_split_contamination,
    extract_dynamic_8grams,
    verify_manifest_integrity,
)
from prlr.domain.loader import (
    DomainBatch,
    EvaluationBatch,
    PRLRDomainDataLoader,
    PRLRDomainDataset,
)
from prlr.domain.schema import (
    AnswerKey,
    DatasetManifest,
    DatasetSplits,
    DomainSample,
    EvaluationInput,
    SplitManifestEntry,
    SplitType,
)
from prlr.domain.solver_lane import (
    DOMAIN_CATALOGUES,
    DeterministicToolRoutingOracle,
    ExecutionTraceStep,
    ProceduralLaneGenerator,
    ProceduralProblemInstance,
    ProceduralVerifier,
    ToolDefinition,
)

__all__ = [
    "ToolDefinition",
    "ExecutionTraceStep",
    "ProceduralProblemInstance",
    "DOMAIN_CATALOGUES",
    "DeterministicToolRoutingOracle",
    "ProceduralLaneGenerator",
    "ProceduralVerifier",
    "SplitType",
    "EvaluationInput",
    "AnswerKey",
    "DomainSample",
    "DatasetSplits",
    "SplitManifestEntry",
    "DatasetManifest",
    "ContaminationError",
    "PromptCollisionContaminationError",
    "KeyLeakageContaminationError",
    "JaccardOverlapContaminationError",
    "canonicalize_prompt",
    "extract_dynamic_8grams",
    "check_split_contamination",
    "verify_manifest_integrity",
    "DomainBatch",
    "EvaluationBatch",
    "PRLRDomainDataset",
    "PRLRDomainDataLoader",
]
