"""Dataset generator for PRLR 5-way procedural reasoning benchmark conforming to prlr.domain.v1.

Produces:
- 5 Partitioned splits with disjoint PRNG seed allocations:
  - train (512 samples, K in [2, 4], base seed 100_000)
  - dev (128 samples, K in [2, 4], base seed 200_000)
  - sealed_test (256 samples, K in [2, 4], base seed 300_000, includes robotics_control)
  - sealed_gate (128 samples, K in [2, 4], base seed 400_000)
  - extrapolation (256 samples, K in [5, 8], base seed 500_000)
- Isolated views: evaluation_inputs/ (0 ground truth) and answer_keys/
- Cryptographic dataset_manifest.json with SHA-256 hashes and 4-tier contamination verification
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional

from prlr.domain.contamination import check_split_contamination
from prlr.domain.schema import (
    DatasetManifest,
    DatasetSplits,
    DomainSample,
    SplitManifestEntry,
    SplitType,
)
from prlr.domain.solver_lane import ProceduralLaneGenerator


def get_git_commit(cwd: Optional[Path] = None) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "reproducible_pre_release_commit"


def generate_5way_dataset(
    output_dir: Path | str,
    train_count: int = 512,
    dev_count: int = 128,
    sealed_test_count: int = 256,
    sealed_gate_count: int = 128,
    extrapolation_count: int = 256,
) -> DatasetManifest:
    """Generate immutable 5-way procedural dataset and write to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_path / "evaluation_inputs"
    keys_dir = out_path / "answer_keys"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    generator = ProceduralLaneGenerator()

    id_domains = ["api_workflow", "data_pipeline", "security_ops"]
    all_domains = ["api_workflow", "data_pipeline", "security_ops", "robotics_control"]

    split_configs = [
        {
            "name": "train",
            "count": train_count,
            "base_seed": 100_000,
            "domains": id_domains,
            "k_range": (2, 4),
        },
        {
            "name": "dev",
            "count": dev_count,
            "base_seed": 200_000,
            "domains": id_domains,
            "k_range": (2, 4),
        },
        {
            "name": "sealed_test",
            "count": sealed_test_count,
            "base_seed": 300_000,
            "domains": all_domains,  # includes held-out robotics_control
            "k_range": (2, 4),
        },
        {
            "name": "sealed_gate",
            "count": sealed_gate_count,
            "base_seed": 400_000,
            "domains": id_domains,
            "k_range": (2, 4),
        },
        {
            "name": "extrapolation",
            "count": extrapolation_count,
            "base_seed": 500_000,
            "domains": all_domains,
            "k_range": (5, 8),
        },
    ]

    samples_by_split: Dict[str, List[DomainSample]] = {}

    for cfg in split_configs:
        s_name: SplitType = cfg["name"]  # type: ignore
        count: int = cfg["count"]
        base_seed: int = cfg["base_seed"]
        domains: List[str] = cfg["domains"]
        k_min, k_max = cfg["k_range"]

        samples: List[DomainSample] = []
        for i in range(count):
            seed = base_seed + i
            domain = domains[i % len(domains)]
            # Cycle through target depths in range
            k = k_min + (i % (k_max - k_min + 1))
            instance = generator.generate_instance(
                domain=domain,
                seed=seed,
                target_depth_K=k,
                num_distractors=2,
            )

            difficulty = k
            num_steps = k

            sample = DomainSample(
                id=f"{s_name}_{instance.sample_id}",
                split=s_name,
                domain=domain,
                difficulty=difficulty,
                num_steps=num_steps,
                prompt=instance.prompt,
                target_solution=instance.ground_truth_json,
                ground_truth=instance.ground_truth_json,
                verifier_type="mtr_bfs_oracle",
                verifier_config={
                    "expected_route": list(instance.expected_route),
                    "terminal_tool": instance.terminal_tool,
                    "target_goal": instance.target_goal,
                },
                seed=seed,
                metadata=dict(instance.metadata),
            )
            samples.append(sample)

        samples_by_split[s_name] = samples

    splits = DatasetSplits(
        train=samples_by_split["train"],
        dev=samples_by_split["dev"],
        sealed_test=samples_by_split["sealed_test"],
        sealed_gate=samples_by_split["sealed_gate"],
        extrapolation=samples_by_split["extrapolation"],
    )

    # Execute 4-tier contamination check prior to writing
    audit_results = check_split_contamination(splits)

    # Serialize files and compute SHA-256
    split_manifest_entries: Dict[str, SplitManifestEntry] = {}

    for cfg in split_configs:
        s_name = cfg["name"]
        s_samples = samples_by_split[s_name]

        # 1. Master split file
        file_name = f"{s_name}.jsonl"
        file_path = out_path / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            for s in s_samples:
                f.write(json.dumps(s.to_dict()) + "\n")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        byte_size = len(file_bytes)

        # 2. Evaluation inputs (isolated view, zero ground truth)
        in_file_name = f"evaluation_inputs/{s_name}_inputs.jsonl"
        in_file_path = out_path / in_file_name
        with open(in_file_path, "w", encoding="utf-8") as f:
            for s in s_samples:
                f.write(json.dumps(s.to_evaluation_input().to_dict()) + "\n")

        with open(in_file_path, "rb") as f:
            in_sha256 = hashlib.sha256(f.read()).hexdigest()

        # 3. Answer keys (isolated view)
        k_file_name = f"answer_keys/{s_name}_keys.jsonl"
        k_file_path = out_path / k_file_name
        with open(k_file_path, "w", encoding="utf-8") as f:
            for s in s_samples:
                f.write(json.dumps(s.to_answer_key().to_dict()) + "\n")

        with open(k_file_path, "rb") as f:
            k_sha256 = hashlib.sha256(f.read()).hexdigest()

        split_manifest_entries[s_name] = SplitManifestEntry(
            file_name=file_name,
            sample_count=len(s_samples),
            byte_size=byte_size,
            sha256=file_sha256,
            base_seed=cfg["base_seed"],
            inputs_file=in_file_name,
            inputs_sha256=in_sha256,
            keys_file=k_file_name,
            keys_sha256=k_sha256,
        )

    # Build and write dataset_manifest.json
    total_samples = sum(len(s) for s in samples_by_split.values())
    manifest = DatasetManifest(
        schema_version="prlr.domain.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        domain_name="mtr_dag_tool_routing",
        source_commit=get_git_commit(out_path),
        total_samples=total_samples,
        splits=split_manifest_entries,
        contamination_status=audit_results["status"],
        audit_metrics=audit_results,
    )

    manifest_path = out_path / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return manifest


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/prlr_domain_v1"
    manifest = generate_5way_dataset(target)
    print(f"Generated {manifest.total_samples} samples under {target}")
    print(f"Manifest SHA-256 recorded with contamination status: {manifest.contamination_status}")
