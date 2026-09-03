"""Test Suite for Milestone 4 Requirement R6: Procedural Solver Lane, 5-Way Splitting & Contamination Defense.

Verifies:
- 4 Domain Families (api_workflow, data_pipeline, security_ops, robotics_control)
- Deterministic BFS Oracle with canonical lexicographical tie-breaking and execution tracing
- ProceduralLaneGenerator with guaranteed K >= 2 multi-step deliberation
- ProceduralVerifier with independent ground-truth evaluation (Rules 1 & 2)
- Strict 5-way partition structure (train 512, dev 128, sealed_test 256, sealed_gate 128, extrapolation 256)
- 4-Tier Contamination Defense:
  - 0% exact canonical prompt collisions across all 10 split pairs
  - 0% parameter fingerprint collisions between train and eval splits
  - 0% answer key / target solution leakage into prompts
  - Dynamic 8-gram Jaccard bound < 0.10 between train and sealed_test
- PRLRDomainDataset & PRLRDomainDataLoader with MLX batching, right-padding, and strict target_mask
- Dataset manifest SHA-256 cryptographic verification and tampering detection
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

import mlx.core as mx

from prlr.domain.contamination import (
    ContaminationError,
    PromptCollisionContaminationError,
    canonicalize_prompt,
    check_split_contamination,
    extract_dynamic_8grams,
    verify_manifest_integrity,
)
from prlr.domain.loader import DomainBatch, EvaluationBatch, PRLRDomainDataLoader, PRLRDomainDataset
from prlr.domain.schema import (
    AnswerKey,
    DatasetManifest,
    DatasetSplits,
    DomainSample,
    EvaluationInput,
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
from prlr.manifest import ModelManifest, Rule5ViolationError


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "prlr_domain_v1"


def test_tool_catalogues_completeness():
    """Verify all 4 domain catalogues exist with valid core tools and distractors."""
    required_domains = ["api_workflow", "data_pipeline", "security_ops", "robotics_control"]
    for dom in required_domains:
        assert dom in DOMAIN_CATALOGUES, f"Missing domain catalogue: {dom}"
        core = DOMAIN_CATALOGUES[dom]["core"]
        dist = DOMAIN_CATALOGUES[dom]["distractors"]
        assert len(core) >= 6, f"Domain {dom} must have at least 6 core tools"
        assert len(dist) >= 2, f"Domain {dom} must have at least 2 distractors"

        for t in core + dist:
            assert isinstance(t, ToolDefinition)
            assert t.name and len(t.name) > 2
            assert len(t.required_inputs) >= 1
            assert len(t.produced_outputs) >= 1
            assert t.description


def test_deterministic_bfs_oracle_minimal_path_and_trace():
    """Verify BFS oracle finds minimal route, canonical tie-breaking, and execution trace."""
    oracle = DeterministicToolRoutingOracle()

    tools = [
        ToolDefinition("tool_a", ("init_var",), ("step1_var",), "Step 1"),
        ToolDefinition("tool_b", ("step1_var",), ("goal_var",), "Step 2"),
        ToolDefinition("tool_c", ("init_var",), ("step1_var",), "Alternative Step 1"),
    ]

    # Minimal route is K=2
    sol = oracle.solve(tools, {"init_var"}, "goal_var")
    assert sol is not None
    assert sol["depth_K"] == 2
    assert sol["terminal_tool"] == "tool_b"
    # Canonical tie-breaking: 'tool_a' comes before 'tool_c' lexicographically
    assert sol["route"] == ("tool_a", "tool_b")
    assert len(sol["trace"]) == 2
    step1 = sol["trace"][0]
    assert step1.step == 1
    assert step1.tool == "tool_a"
    assert "init_var" in step1.consumed
    assert "step1_var" in step1.produced
    step2 = sol["trace"][1]
    assert step2.step == 2
    assert step2.tool == "tool_b"
    assert "goal_var" in step2.cumulative_state


def test_procedural_lane_generator_k_greater_equal_2():
    """Verify generator enforces K >= 2, valid prompt formatting, and non-trivial deliberation."""
    generator = ProceduralLaneGenerator()

    # Reject K < 2
    with pytest.raises(ValueError, match="target_depth_K must be >= 2"):
        generator.generate_instance("api_workflow", seed=42, target_depth_K=1)

    # Test K in {2, 3, 4} across all 4 domains
    for dom in ["api_workflow", "data_pipeline", "security_ops", "robotics_control"]:
        for k in [2, 3, 4]:
            inst = generator.generate_instance(dom, seed=100 + k, target_depth_K=k, num_distractors=2)
            assert isinstance(inst, ProceduralProblemInstance)
            assert inst.depth_K == k
            assert len(inst.expected_route) == k
            assert inst.terminal_tool == inst.expected_route[-1]
            assert inst.domain == dom
            assert "<start_of_turn>user" in inst.prompt
            assert "<end_of_turn>" in inst.prompt
            assert "<start_of_turn>model" in inst.prompt
            assert "System role not supported" not in inst.prompt
            # Target solution JSON valid
            target_data = json.loads(inst.ground_truth_json)
            assert target_data["route"] == list(inst.expected_route)
            assert target_data["terminal"] == inst.terminal_tool


def test_procedural_verifier_exact_and_operational():
    """Verify independent verifier evaluates exact match and operational semantics."""
    verifier = ProceduralVerifier()

    expected = ("schema_parser", "auth_validator", "permission_checker")
    tools = DOMAIN_CATALOGUES["api_workflow"]["core"]
    init = ("raw_payload",)
    goal = "perm_token"

    # 1. Exact match valid
    pred_text = '<start_of_turn>model\n{"route": ["schema_parser", "auth_validator", "permission_checker"], "terminal": "permission_checker"}<end_of_turn>'
    res = verifier.verify(pred_text, expected, tools=tools, initial_state=init, goal=goal)
    assert res["is_valid"] is True
    assert res["exact_match"] is True
    assert len(res["errors"]) == 0

    # 2. Wrong route
    bad_text = '{"route": ["schema_parser", "auth_validator"], "terminal": "auth_validator"}'
    res_bad = verifier.verify(bad_text, expected, tools=tools, initial_state=init, goal=goal)
    assert res_bad["exact_match"] is False
    assert res_bad["is_valid"] is False
    assert any("not produced" in e for e in res_bad["errors"])

    # 3. Malformed JSON
    malformed = "I believe the answer is schema_parser."
    res_mal = verifier.verify(malformed, expected)
    assert res_mal["is_valid"] is False
    assert res_mal["exact_match"] is False


def test_5way_split_partition_structure():
    """Verify all 5 partitions exist on disk with expected sample counts."""
    assert DATA_DIR.exists(), f"Dataset directory {DATA_DIR} does not exist"

    manifest_path = DATA_DIR / "dataset_manifest.json"
    assert manifest_path.exists(), "dataset_manifest.json missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest = DatasetManifest.from_dict(manifest_data)
    assert manifest.schema_version == "prlr.domain.v1"
    assert manifest.total_samples == 1280

    expected_counts = {
        "train": 512,
        "dev": 128,
        "sealed_test": 256,
        "sealed_gate": 128,
        "extrapolation": 256,
    }

    for s_name, exp_count in expected_counts.items():
        assert s_name in manifest.splits
        entry = manifest.splits[s_name]
        assert entry.sample_count == exp_count

        split_file = DATA_DIR / entry.file_name
        assert split_file.exists()
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == exp_count

        inputs_file = DATA_DIR / entry.inputs_file
        assert inputs_file.exists()
        with open(inputs_file, "r", encoding="utf-8") as f:
            in_lines = [json.loads(line) for line in f if line.strip()]
        assert len(in_lines) == exp_count

        keys_file = DATA_DIR / entry.keys_file
        assert keys_file.exists()
        with open(keys_file, "r", encoding="utf-8") as f:
            k_lines = [json.loads(line) for line in f if line.strip()]
        assert len(k_lines) == exp_count


def test_pairwise_prompt_hash_zero_collision():
    """Tier 1 Defense: 0% exact canonical prompt collisions across all 10 split pairs."""
    splits_data = {}
    for s_name in ["train", "dev", "sealed_test", "sealed_gate", "extrapolation"]:
        with open(DATA_DIR / f"{s_name}.jsonl", "r", encoding="utf-8") as f:
            splits_data[s_name] = [
                canonicalize_prompt(json.loads(line)["prompt"])
                for line in f if line.strip()
            ]

    split_names = list(splits_data.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1 = split_names[i]
            s2 = split_names[j]
            set1 = set(splits_data[s1])
            set2 = set(splits_data[s2])
            overlap = set1.intersection(set2)
            assert len(overlap) == 0, f"Collision detected between {s1} and {s2}: {len(overlap)} shared prompts"


def test_parameter_fingerprint_isolation():
    """Tier 2 Defense: 0% parameter fingerprint overlap between train and eval splits."""
    def get_fingerprints(file_path):
        fps = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    fp = item.get("metadata", {}).get("fingerprint")
                    if fp:
                        fps.add(fp)
        return fps

    train_fps = get_fingerprints(DATA_DIR / "train.jsonl")
    for eval_split in ["dev", "sealed_test", "sealed_gate", "extrapolation"]:
        eval_fps = get_fingerprints(DATA_DIR / f"{eval_split}.jsonl")
        overlap = train_fps.intersection(eval_fps)
        assert len(overlap) == 0, f"Parameter fingerprint overlap between train and {eval_split}: {overlap}"


def test_ground_truth_isolation_contract():
    """Rule 1 Contract: EvaluationInput contains zero ground truth or target solutions."""
    with open(DATA_DIR / "evaluation_inputs" / "sealed_test_inputs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assert "target_solution" not in item
                assert "ground_truth" not in item
                assert "verifier_config" not in item
                assert "expected_route" not in item
                assert "prompt" in item
                assert "id" in item


def test_dynamic_ngram_jaccard_bound():
    """Tier 4 Defense: Dynamic 8-gram Jaccard bound between train and sealed_test < 0.10."""
    def extract_grams(file_path):
        grams = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item.get("metadata", {}).get("instance_text", item["prompt"])
                    grams.update(extract_dynamic_8grams(text))
        return grams

    train_grams = extract_grams(DATA_DIR / "train.jsonl")
    test_grams = extract_grams(DATA_DIR / "sealed_test.jsonl")

    intersection = len(train_grams.intersection(test_grams))
    union = len(train_grams.union(test_grams))
    jaccard = intersection / union if union > 0 else 0.0

    assert jaccard < 0.10, f"Train vs sealed_test 8-gram Jaccard {jaccard:.4f} exceeds 0.10 threshold"


def test_dataset_manifest_sha256_verification_and_tampering():
    """Verify cryptographic manifest validation and assert single-byte tampering is caught."""
    # 1. Clean verification passes
    manifest = verify_manifest_integrity(DATA_DIR)
    assert manifest.contamination_status == "PASS_ZERO_CONTAMINATION"

    # 2. Tampering test in temp dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Copy dev.jsonl and manifest
        with open(DATA_DIR / "dev.jsonl", "r", encoding="utf-8") as f:
            dev_data = f.read()
        with open(DATA_DIR / "dataset_manifest.json", "r", encoding="utf-8") as f:
            man_data = f.read()

        # Write to tmp
        (tmp_path / "evaluation_inputs").mkdir(parents=True)
        (tmp_path / "answer_keys").mkdir(parents=True)
        # Copy input and key
        with open(DATA_DIR / "evaluation_inputs" / "dev_inputs.jsonl", "r") as f:
            (tmp_path / "evaluation_inputs" / "dev_inputs.jsonl").write_text(f.read())
        with open(DATA_DIR / "answer_keys" / "dev_keys.jsonl", "r") as f:
            (tmp_path / "answer_keys" / "dev_keys.jsonl").write_text(f.read())

        # Tamper single byte in dev.jsonl
        tampered_dev = dev_data.replace('dev_', 'tampered_', 1)
        (tmp_path / "dev.jsonl").write_text(tampered_dev)

        # Create mini-manifest with only dev
        man_dict = json.loads(man_data)
        man_dict["splits"] = {"dev": man_dict["splits"]["dev"]}
        (tmp_path / "dataset_manifest.json").write_text(json.dumps(man_dict))

        with pytest.raises(ContaminationError, match="Integrity check failed"):
            verify_manifest_integrity(tmp_path)


def test_dataloader_padding_and_masking():
    """Verify PRLRDomainDataset & DataLoader padding, masking, and Rule 5 enforcement."""
    # Dummy tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=True):
            # Deterministic tokenization
            words = text.split()
            base = [hash(w) % 1000 + 10 for w in words]
            if add_special_tokens:
                return [2] + base  # BOS=2
            return base

    tok = MockTokenizer()

    # Rule 5 check: None tokenizer rejected
    with pytest.raises(Rule5ViolationError):
        PRLRDomainDataset(samples=[], tokenizer=None)

    # Load 8 samples from dev
    samples = []
    with open(DATA_DIR / "dev.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            samples.append(DomainSample.from_dict(json.loads(line)))

    dataset = PRLRDomainDataset(
        samples=samples,
        tokenizer=tok,
        pad_token_id=0,
        eos_token_ids=(1, 107),
    )

    loader = PRLRDomainDataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, DomainBatch)
    assert batch.input_ids.shape[0] == 4
    assert batch.prompt_mask.shape == batch.input_ids.shape
    assert batch.target_ids.shape[0] == 4
    assert batch.target_mask.shape == batch.target_ids.shape

    # Target mask must have 1.0 on valid tokens and 0.0 on padding
    t_mask = batch.target_mask
    assert mx.all((t_mask == 0.0) | (t_mask == 1.0))
