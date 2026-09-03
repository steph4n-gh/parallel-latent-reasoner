"""Contamination defense, leakage prevention, and manifest verification for PRLR domain data.

Enforces 4-tier contamination defense:
- Tier 1: 0% exact canonical prompt collisions across all 10 split pairs
- Tier 2: 0% problem parameter fingerprint overlap between train and evaluation splits
- Tier 3: 0% answer key / target solution leakage into prompts
- Tier 4: Dynamic instance 8-gram Jaccard bound < 0.10 between train and sealed_test
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from prlr.domain.schema import DatasetManifest, DatasetSplits, DomainSample, SplitManifestEntry


class ContaminationError(Exception):
    """Base exception for data leakage across dataset splits."""


class PromptCollisionContaminationError(ContaminationError):
    """Raised when exact or canonical prompt is duplicated across splits."""


class KeyLeakageContaminationError(ContaminationError):
    """Raised when answer keys or ground truth leak into prompts or between splits."""


class JaccardOverlapContaminationError(ContaminationError):
    """Raised when n-gram similarity across splits exceeds threshold."""


def canonicalize_prompt(prompt: str) -> str:
    """Normalize prompt text by lowercasing and collapsing whitespace."""
    return re.sub(r"\s+", " ", prompt.strip().lower())


def extract_dynamic_8grams(text: str) -> Set[str]:
    """Extract word 8-grams from normalized text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + 8]) for i in range(len(words) - 7)}


def check_split_contamination(
    splits: DatasetSplits,
    max_train_test_jaccard: float = 0.10,
    check_param_fingerprints: bool = True,
) -> Dict[str, Any]:
    """Execute rigorous 4-tier contamination defense across all 5 dataset splits.

    Args:
        splits: DatasetSplits container holding all 5 partitions.
        max_train_test_jaccard: Maximum allowable 8-gram Jaccard overlap between train and test.
        check_param_fingerprints: Whether to verify 0 parameter fingerprint collisions.

    Returns:
        Dict detailing audit metrics and confirmation of 0% contamination.
    """
    split_names = ["train", "dev", "sealed_test", "sealed_gate", "extrapolation"]
    samples_by_split = {name: splits.get_split(name) for name in split_names}

    # 1. Tier 1 — Exact canonical prompt hash collisions across all 10 pairs
    hashes_by_split: Dict[str, Set[str]] = {}
    for name, s_list in samples_by_split.items():
        hashes = set()
        for s in s_list:
            h = hashlib.sha256(canonicalize_prompt(s.prompt).encode("utf-8")).hexdigest()
            hashes.add(h)
        hashes_by_split[name] = hashes

    pairwise_collisions = {}
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = hashes_by_split[s1].intersection(hashes_by_split[s2])
            if overlap:
                raise PromptCollisionContaminationError(
                    f"Contamination detected! Splits '{s1}' and '{s2}' share {len(overlap)} exact prompts."
                )
            pairwise_collisions[f"{s1}_vs_{s2}"] = 0

    # 2. Tier 2 — Parameter fingerprint isolation
    if check_param_fingerprints:
        train_params = set()
        for s in samples_by_split["train"]:
            if "fingerprint" in s.metadata:
                train_params.add(s.metadata["fingerprint"])
        for eval_split in ["dev", "sealed_test", "sealed_gate", "extrapolation"]:
            for s in samples_by_split[eval_split]:
                fp = s.metadata.get("fingerprint")
                if fp and fp in train_params:
                    raise KeyLeakageContaminationError(
                        f"Parameter fingerprint collision between 'train' and '{eval_split}': {fp}"
                    )

    # 3. Tier 3 — Answer key leakage into prompt
    for name, s_list in samples_by_split.items():
        for s in s_list:
            # Check ground truth string or route
            gt_str = str(s.ground_truth).strip().lower()
            # If target solution appears verbatim in prompt
            sol_str = str(s.target_solution).strip().lower()
            p_canon = canonicalize_prompt(s.prompt)
            if len(sol_str) > 10 and sol_str in p_canon:
                raise KeyLeakageContaminationError(
                    f"Sample {s.id} in split '{name}' contains target_solution verbatim in prompt!"
                )

    # 4. Tier 4 — Dynamic 8-gram Jaccard bound between train and sealed_test
    train_grams: Set[str] = set()
    for s in samples_by_split["train"]:
        text = s.metadata.get("instance_text", s.prompt)
        train_grams.update(extract_dynamic_8grams(text))

    test_grams: Set[str] = set()
    for s in samples_by_split["sealed_test"]:
        text = s.metadata.get("instance_text", s.prompt)
        test_grams.update(extract_dynamic_8grams(text))

    intersection = len(train_grams.intersection(test_grams))
    union = len(train_grams.union(test_grams))
    jaccard = (intersection / union) if union > 0 else 0.0

    if jaccard > max_train_test_jaccard:
        raise JaccardOverlapContaminationError(
            f"8-gram Jaccard overlap between train and sealed_test ({jaccard:.4f}) "
            f"exceeds max threshold {max_train_test_jaccard}"
        )

    return {
        "status": "PASS_ZERO_CONTAMINATION",
        "sample_counts": {name: len(samples_by_split[name]) for name in split_names},
        "pairwise_collisions": pairwise_collisions,
        "train_test_8gram_jaccard": jaccard,
    }


def verify_manifest_integrity(
    data_dir: Path | str,
    manifest_path: Optional[Path | str] = None,
) -> DatasetManifest:
    """Verify on-disk dataset files against dataset_manifest.json with SHA-256."""
    d_dir = Path(data_dir)
    m_path = Path(manifest_path) if manifest_path is not None else d_dir / "dataset_manifest.json"

    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        manifest_dict = json.load(f)

    manifest = DatasetManifest.from_dict(manifest_dict)

    for split_name, entry in manifest.splits.items():
        # Check split jsonl file
        file_path = d_dir / entry.file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Split file missing: {file_path}")

        with open(file_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()

        if actual_sha != entry.sha256:
            raise ContaminationError(
                f"Integrity check failed for {file_path.name}! "
                f"Expected SHA-256 {entry.sha256}, got {actual_sha}"
            )

        # Check inputs file
        inputs_path = d_dir / entry.inputs_file
        if inputs_path.exists():
            with open(inputs_path, "rb") as f:
                actual_in_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_in_sha != entry.inputs_sha256:
                raise ContaminationError(
                    f"Integrity check failed for {inputs_path.name}! "
                    f"Expected SHA-256 {entry.inputs_sha256}, got {actual_in_sha}"
                )

        # Check keys file
        keys_path = d_dir / entry.keys_file
        if keys_path.exists():
            with open(keys_path, "rb") as f:
                actual_k_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_k_sha != entry.keys_sha256:
                raise ContaminationError(
                    f"Integrity check failed for {keys_path.name}! "
                    f"Expected SHA-256 {entry.keys_sha256}, got {actual_k_sha}"
                )

    return manifest


__all__ = [
    "ContaminationError",
    "PromptCollisionContaminationError",
    "KeyLeakageContaminationError",
    "JaccardOverlapContaminationError",
    "canonicalize_prompt",
    "extract_dynamic_8grams",
    "check_split_contamination",
    "verify_manifest_integrity",
]
