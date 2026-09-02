"""Comprehensive Test Suite for Multi-Domain Dataset Generation and Validation.

Covers Tier 1 (Feature Coverage & Unit Contracts) and Tier 2 (Boundary & Adversarial Stress):
1. Procedural and curated generation across 4 core reasoning domains:
   - Domain 1: Multi-Step Reasoning & Multi-Clue Synthesis
   - Domain 2: Constraint Satisfaction & Combinatorial Optimization
   - Domain 3: Entity Disambiguation & Semantic Binding
   - Domain 4: Arithmetic & Quantitative Reasoning
2. Schema & format validation (JSONL serialization, dataclass contracts, prlr.dataset.v1 schema).
3. 100% verifier pass rate on ground-truth solutions across all 4 domains.
4. Distractor / negative solution rejection guards.
5. 80/10/10 split generation with zero prompt hash collision and <15% 8-gram Jaccard overlap.
6. Token length boundary validation.
7. Adversarial, malformed, and boundary stress tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import mlx.core as mx
import pytest

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    EvaluationResult,
    VerifierType,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.dataset import (
    DistillationSample,
    PRLRDataLoader,
    PRLRDataset,
    ProceduralMultiDomainGenerator,
    check_split_contamination,
    generate_distillation_dataset,
    split_dataset,
    train_prlr_adapter,
)
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.trainer import TrainerConfig


# ============================================================================
# Tier 1 Tests: Unit Contracts, Schema Integrity & Domain Coverage
# ============================================================================

def test_distillation_sample_dataclass_contract():
    """Verify DistillationSample serialization and roundtrip dictionary fidelity."""
    sample = ProceduralMultiDomainGenerator.generate_arithmetic(seed=999)
    d = sample.to_dict()
    assert isinstance(d, dict)
    assert d["id"] == sample.id
    assert d["domain"] == "arithmetic_reasoning"
    assert d["ground_truth"] == sample.ground_truth

    reconstructed = DistillationSample.from_dict(d)
    assert reconstructed.id == sample.id
    assert reconstructed.domain == sample.domain
    assert reconstructed.ground_truth == sample.ground_truth
    assert reconstructed.difficulty == sample.difficulty
    assert reconstructed.seed == sample.seed


def test_four_core_domains_generation():
    """Verify procedural task generation yields valid samples across all 4 core domains."""
    domains_tested = set()

    # 1. Multi-Step Reasoning
    s1 = ProceduralMultiDomainGenerator.generate_multistep_reasoning(1)
    assert s1.domain == "multi_step_reasoning"
    assert len(s1.ground_truth) > 0
    assert len(s1.prompt) > 20
    domains_tested.add(s1.domain)

    # 2. Constraint Satisfaction
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(0)
    assert s2.domain == "constraint_satisfaction"
    parsed_csp = json.loads(s2.ground_truth)
    assert "selected_instruments" in parsed_csp
    assert len(parsed_csp["selected_instruments"]) == 3
    domains_tested.add(s2.domain)

    # 3. Entity Disambiguation
    s3 = ProceduralMultiDomainGenerator.generate_entity_disambiguation(0)
    assert s3.domain == "entity_disambiguation"
    assert s3.ground_truth == "the grand piano"
    domains_tested.add(s3.domain)

    # 4. Arithmetic
    s4 = ProceduralMultiDomainGenerator.generate_arithmetic(0)
    assert s4.domain == "arithmetic_reasoning"
    assert s4.ground_truth == "12"
    domains_tested.add(s4.domain)

    assert len(domains_tested) == 4, f"Expected 4 distinct domains, got {len(domains_tested)}"


def test_verifier_ground_truth_pass_rates():
    """Verify 100% verifier pass rate on ground truth across all 4 domain generators."""
    samples = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning(10),
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction(20),
        ProceduralMultiDomainGenerator.generate_entity_disambiguation(30),
        ProceduralMultiDomainGenerator.generate_arithmetic(40),
    ]

    for s in samples:
        # Construct temporary CognitiveTestCase to evaluate via deterministic verifier engine
        v_type = (
            VerifierType.JSON_SCHEMA if s.verifier_type == "json_schema"
            else (VerifierType.MATHEMATICAL_CONSTRAINT if s.verifier_type == "mathematical_constraint"
                  else VerifierType.EXACT_MATCH)
        )
        case = CognitiveTestCase(
            id=s.id,
            domain=s.domain,
            title=f"Test {s.id}",
            prompt=s.prompt,
            ground_truth=s.ground_truth,
            expected_constraints=[],
            verifier_type=v_type,
            metadata=s.verifier_config,
        )

        res = verify_test_case_result(case, s.ground_truth)
        assert res.passed is True, f"Ground truth for {s.id} failed verification! Details: {res.feedback}"
        assert res.score == 1.0, f"Expected perfect score 1.0 for {s.id}, got {res.score}"


def test_verifier_negative_distractor_rejection():
    """Verify programmatic verifiers correctly reject invalid / distractor answers."""
    # 1. Multi-Step: Distractor suspect Alice should fail
    s1 = ProceduralMultiDomainGenerator.generate_multistep_reasoning(1)
    case1 = CognitiveTestCase(
        id=s1.id, domain=s1.domain, title="Alibi", prompt=s1.prompt,
        ground_truth=s1.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.EXACT_MATCH,
    )
    res_wrong = verify_test_case_result(case1, "Alice")
    assert res_wrong.passed is False
    assert res_wrong.score == 0.0

    # 2. Constraint: Over-budget payload should fail
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(0)
    case2 = CognitiveTestCase(
        id=s2.id, domain=s2.domain, title="Payload", prompt=s2.prompt,
        ground_truth=s2.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata=s2.verifier_config,
    )
    res_over = verify_test_case_result(case2, json.dumps({"selected_instruments": ["Spectrometer", "Radar"]}))
    assert res_over.passed is False

    # 3. Arithmetic: Wrong numerical answer should fail
    s4 = ProceduralMultiDomainGenerator.generate_arithmetic(0)
    case4 = CognitiveTestCase(
        id=s4.id, domain=s4.domain, title="Apples", prompt=s4.prompt,
        ground_truth=s4.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata=s4.verifier_config,
    )
    res_num_wrong = verify_test_case_result(case4, "18")
    assert res_num_wrong.passed is False
    assert res_num_wrong.score == 0.0


# ============================================================================
# Tier 2 Tests: Split Disjointness, Bounds, and Adversarial Resilience
# ============================================================================

def _compute_8gram_set(text: str) -> set[str]:
    """Compute normalized word 8-grams."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return {" ".join(words)}
    return {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}


def test_split_generation_and_zero_contamination():
    """Verify 80/10/10 split generation with zero prompt hash collision and <15% n-gram leakage."""
    # 80 Train samples (seeds 1000..1019)
    train_samples: list[DistillationSample] = []
    for i in range(20):
        train_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=2000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=3000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4000 + i, split="train"))

    # 10 Validation samples (seeds 1020..1022)
    val_samples: list[DistillationSample] = []
    for i in range(2):
        val_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=2020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=3020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4020 + i, split="train"))
    val_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1023, split="train"))
    val_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4023, split="train"))

    # 10 Holdout Test samples (distinct holdout domain tasks, split="test")
    test_samples: list[DistillationSample] = []
    for i in range(2):
        test_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=5000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=6000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=7000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=8000 + i, split="test"))
    test_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=5003, split="test"))
    test_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=8003, split="test"))

    assert len(train_samples) == 80
    assert len(val_samples) == 10
    assert len(test_samples) == 10

    # 1. Exact Prompt Hash Uniqueness across splits
    train_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in train_samples}
    val_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in val_samples}
    test_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in test_samples}

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train and Val splits share identical prompts!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train and Test splits share identical prompts!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val and Test splits share identical prompts!"

    # 2. 8-Gram Jaccard Overlap < 15%
    train_grams: set[str] = set()
    for s in train_samples:
        train_grams.update(_compute_8gram_set(s.prompt))

    test_grams: set[str] = set()
    for s in test_samples:
        test_grams.update(_compute_8gram_set(s.prompt))

    intersection = len(train_grams.intersection(test_grams))
    union = len(train_grams.union(test_grams))
    jaccard = (intersection / union) if union > 0 else 0.0

    assert jaccard < 0.15, f"8-gram Jaccard similarity ({jaccard:.3f}) exceeds 15% maximum allowed leakage!"


def test_token_and_text_length_bounds():
    """Verify prompt, solution, and CoT length bounds across generated records."""
    generators = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning,
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction,
        ProceduralMultiDomainGenerator.generate_entity_disambiguation,
        ProceduralMultiDomainGenerator.generate_arithmetic,
    ]

    for gen in generators:
        sample = gen(seed=42)
        words_prompt = len(sample.prompt.split())
        words_solution = len(sample.target_solution.split())
        words_cot = len(sample.teacher_cot.split()) if sample.teacher_cot else 0

        assert 5 <= words_prompt <= 512, f"Prompt length out of bounds: {words_prompt} words in {sample.id}"
        assert 1 <= words_solution <= 64, f"Solution length out of bounds: {words_solution} words in {sample.id}"
        assert 5 <= words_cot <= 384, f"CoT length out of bounds: {words_cot} words in {sample.id}"


def test_jsonl_batch_io_roundtrip(tmp_path):
    """Verify serialization of dataset records to JSONL file and roundtrip deserialization."""
    samples = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=501),
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=502),
        ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=503),
        ProceduralMultiDomainGenerator.generate_arithmetic(seed=504),
    ]

    jsonl_path = tmp_path / "dataset_train.jsonl"

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")

    # Read JSONL
    loaded_samples: list[DistillationSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded_samples.append(DistillationSample.from_dict(json.loads(line)))

    assert len(loaded_samples) == len(samples)
    for orig, loaded in zip(samples, loaded_samples):
        assert orig.id == loaded.id
        assert orig.domain == loaded.domain
        assert orig.ground_truth == loaded.ground_truth
        assert orig.verifier_type == loaded.verifier_type


def test_adversarial_malformed_records():
    """Verify robust error handling when encountering corrupt or missing dataset fields."""
    # 1. Missing required field 'prompt'
    bad_dict = {"id": "bad_01", "domain": "math", "ground_truth": "42"}
    with pytest.raises(KeyError):
        DistillationSample.from_dict(bad_dict)

    # 2. Missing 'id'
    bad_dict_2 = {"prompt": "What is 1+1?", "domain": "math", "ground_truth": "2"}
    with pytest.raises(KeyError):
        DistillationSample.from_dict(bad_dict_2)


def test_deterministic_seed_reproducibility():
    """Verify identical random seeds produce byte-for-byte identical dataset samples."""
    s1 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=777)
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=777)
    s3 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=888)

    assert s1.to_dict() == s2.to_dict(), "Identical seeds must produce identical data samples."
    assert s1.id != s3.id or s1.prompt != s3.prompt, "Different seeds must produce distinct data samples."


def test_all_sixteen_subdomains_generation_and_verification():
    """Verify all 16 distinct subdomains across 4 domains generate valid verifiable samples."""
    subdomains_found = set()
    for domain_fn in [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning,
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction,
        ProceduralMultiDomainGenerator.generate_entity_disambiguation,
        ProceduralMultiDomainGenerator.generate_arithmetic,
    ]:
        for sub_seed in range(4):
            sample = domain_fn(seed=sub_seed, split="train")
            subdomains_found.add(sample.subdomain)

            # Check verification
            v_type = (
                VerifierType.JSON_SCHEMA if sample.verifier_type == "json_schema"
                else (VerifierType.MATHEMATICAL_CONSTRAINT if sample.verifier_type == "mathematical_constraint"
                      else VerifierType.EXACT_MATCH)
            )
            case = CognitiveTestCase(
                id=sample.id,
                domain=sample.domain,
                title=f"Subdomain {sample.subdomain}",
                prompt=sample.prompt,
                ground_truth=sample.ground_truth,
                expected_constraints=[],
                verifier_type=v_type,
                metadata=sample.verifier_config,
            )
            res = verify_test_case_result(case, sample.ground_truth)
            assert res.passed is True, f"Subdomain {sample.subdomain} failed verification: {res.feedback}"

    assert len(subdomains_found) == 16, f"Expected 16 distinct subdomains, got {len(subdomains_found)}: {subdomains_found}"


def test_mathematical_tolerance_boundaries():
    """Verify mathematical verifier handles exact matches, distinct numbers, and negative numbers."""
    case = CognitiveTestCase(
        id="math_tol_01",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Tolerance Test",
        prompt="Calculate value",
        ground_truth="100.0",
        expected_constraints=[],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"target_value": 100.0, "tolerance": 1e-3},
    )

    # Exact match
    assert verify_test_case_result(case, "100.0").passed is True
    # Different numbers should fail
    assert verify_test_case_result(case, "250.0").passed is False
    assert verify_test_case_result(case, "99.0").passed is False
    # Negative number
    assert verify_test_case_result(case, "-50.0").passed is False


def test_empty_and_whitespace_prompt_handling():
    """Verify behavior on empty or whitespace-only inputs."""
    case = CognitiveTestCase(
        id="ws_01",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Whitespace Test",
        prompt="   \n\t  ",
        ground_truth="answer",
        expected_constraints=[],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"expected_entity": "answer"},
    )
    # Empty response should fail
    assert verify_test_case_result(case, "").passed is False
    assert verify_test_case_result(case, "   ").passed is False
    # Correct response should pass even if prompt was whitespace
    assert verify_test_case_result(case, "answer").passed is True


def test_unicode_and_latex_math_in_prompts():
    """Verify prompts containing Unicode, LaTeX math expressions, and special characters."""
    latex_prompt = r"Calculate $\Delta E = \sum_{i=1}^3 m_i c^2$ where $m_1=2, m_2=3, m_3=5$ and $c=3 \times 10^8$ m/s."
    sample = DistillationSample(
        id="latex_001",
        domain="arithmetic_reasoning",
        subdomain="physics_math",
        prompt=latex_prompt,
        ground_truth="9.0e17",
        teacher_cot=r"1. Sum masses: 2 + 3 + 5 = 10 kg. 2. c^2 = 9e16. 3. 10 * 9e16 = 9e17.",
        target_solution="9.0e17",
        verifier_type="mathematical_constraint",
        verifier_config={"target_value": 9.0e17, "tolerance": 1e12},
    )
    d = sample.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert r"\Delta" in deserialized["prompt"]
    assert r"\sum" in deserialized["prompt"]


# ============================================================================
# Tier 3 Tests: Dataset Loaders, Batch Tensor Shapes, and Metal Training
# ============================================================================

def test_prlr_dataset_and_dataloader_collate():
    """Verify PRLRDataset and PRLRDataLoader collation into padded MLX tensor batches."""
    samples = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=101),
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=102),
        ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=103),
        ProceduralMultiDomainGenerator.generate_arithmetic(seed=104),
        ProceduralMultiDomainGenerator.generate_arithmetic(seed=105),
    ]

    dataset = PRLRDataset(samples, vocab_size=128, max_prompt_len=128, max_target_len=32)
    assert len(dataset) == 5
    assert dataset[0].id == samples[0].id

    loader = PRLRDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        vocab_size=128,
        dim=64,
        synthesize_teacher_latents=True,
    )
    assert len(loader) == 3

    batches = list(loader)
    assert len(batches) == 3

    # Batch 1 (size 2)
    b1 = batches[0]
    assert "input_ids" in b1
    assert "target_tokens" in b1
    assert "teacher_latents" in b1
    assert b1["input_ids"].ndim == 2
    assert b1["input_ids"].shape[0] == 2
    assert b1["target_tokens"].ndim == 2
    assert b1["target_tokens"].shape[0] == 2
    assert b1["teacher_latents"].shape == (2, 64)

    # Batch 3 (size 1)
    b3 = batches[2]
    assert b3["input_ids"].shape[0] == 1
    assert b3["target_tokens"].shape[0] == 1
    assert b3["teacher_latents"].shape == (1, 64)


def test_generate_distillation_dataset_distribution_and_splits():
    """Verify generate_distillation_dataset produces balanced, leak-free splits."""
    train_s, val_s, test_s = generate_distillation_dataset(
        total_samples=40, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    assert len(train_s) == 32
    assert len(val_s) == 4
    assert len(test_s) == 4

    # Verify all samples have 100% verifier compliance on ground truth
    for s in train_s + val_s + test_s:
        res = s.verify_ground_truth()
        assert res.passed is True, f"Sample {s.id} failed verification: {res.feedback}"
        assert res.score == 1.0


def test_split_dataset_helper_ratios():
    """Verify split_dataset correctly partitions datasets according to custom ratios."""
    samples = ProceduralMultiDomainGenerator.generate_dataset(num_samples=20, split="train")
    tr, va, te = split_dataset(samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=123)
    assert len(tr) == 14
    assert len(va) == 4
    assert len(te) == 2


def test_distillation_sample_ground_truth_verification_method():
    """Verify DistillationSample verification methods directly execute programmatic verifiers."""
    s_arith = ProceduralMultiDomainGenerator.generate_arithmetic(seed=555)
    res = s_arith.verify_ground_truth()
    assert res.passed is True
    assert res.score == 1.0

    # Wrong prediction should fail
    res_wrong = s_arith.verify_prediction("999999")
    assert res_wrong.passed is False
    assert res_wrong.score == 0.0


def test_train_prlr_adapter_workflow(tmp_path):
    """Verify train_prlr_adapter trains on Metal GPU and saves reloadable production weights."""
    cfg = GemmaLatentConfig.compact_test(vocab_size=128)
    ckpt_path = tmp_path / "test_prlr_adapter.npz"

    model, summary = train_prlr_adapter(
        config=cfg,
        num_samples=20,
        epochs=2,
        batch_size=4,
        unroll_steps=3,
        checkpoint_path=ckpt_path,
        verbose=False,
    )

    assert ckpt_path.exists(), "Checkpoint file was not created!"
    assert ckpt_path.stat().st_size > 0, "Checkpoint file is empty!"
    assert summary["epochs"] == 2
    assert summary["total_steps"] > 0
    assert "initial_loss" in summary
    assert "final_loss" in summary

    # Verify reloading weights into a fresh model
    fresh_model = MLXCompactGemmaModel(cfg)
    loaded_params = fresh_model.load_adapter_weights(ckpt_path)
    assert len(loaded_params) > 0
    assert mx.allclose(fresh_model.prelude.slot_embeddings, model.prelude.slot_embeddings)

