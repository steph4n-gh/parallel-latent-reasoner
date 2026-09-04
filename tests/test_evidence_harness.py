"""Unit tests for the Two-Stage Target-Free Evidence Harness and Token 106 Halting.

Conforms strictly to:
- Evidence Rule 1: Ground-truth isolation (rejection of oracle terms in evaluation inputs).
- Evidence Rule 2: Post-hoc cryptographic scoring (tamper detection via SHA-256 sidecars).
- Evidence Rule 5: Checkpoint integrity (missing checkpoints raise FileNotFoundError).
- Evidence Rule 10: Cryptographic provenance, commit tracking, and runtime capture.
- Milestone 1 / R1: Direct frozen Gemma 4 halting on token 106 (<turn|>).
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from prlr.domain.schema import AnswerKey, DomainSample
from prlr.eval.harness import (
    FORBIDDEN_KEY_SUBSTRINGS,
    FORBIDDEN_ORACLE_TERMS,
    ChecksumMismatchError,
    ConditionScoredSummary,
    CrossRunMergeRejectionError,
    EmptySidecarError,
    EvaluationInput,
    LatencyStatistics,
    MalformedSidecarError,
    MissingCheckpointError,
    MissingSidecarError,
    OracleLeakageError,
    PredictionIntegrityError,
    PredictionRecord,
    ProvenanceMetadata,
    RECURRENT_ADAPTER_CONDITIONS,
    Rule1ViolationError,
    SampleMismatchError,
    ScoredSummaryArtifact,
    SummaryMergeConflictError,
    TamperedPredictionError,
    TargetLeakageException,
    atomic_serialize_prediction_artifact,
    generate_direct_frozen,
    generate_predictions,
    is_forbidden_oracle_key,
    is_valid_git_commit_sha,
    load_target_free_inputs,
    safe_merge_condition_summary,
    score_predictions,
    validate_can_merge,
    validate_target_free_dict,
    validate_target_free_record,
    verify_adapter_checkpoint,
    verify_prediction_file,
)
from prlr.gemma.backbone import GemmaTokenizerWrapper, PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def gemma4_tokenizer():
    """Load official Gemma 4 tokenizer if cached on disk, or provide a mock."""
    try:
        manifest = ModelManifest.gemma_4_12b_it()
        tok_path = Path(manifest.tokenizer_path)
        tok_dir = tok_path.parent if tok_path.is_file() else tok_path
        if tok_dir.exists():
            return AutoTokenizer.from_pretrained(str(tok_dir), fix_mistral_regex=True)
    except Exception:
        pass

    # Mock tokenizer if huggingface_cache not populated
    mock = MagicMock()
    mock.eos_token_id = 1
    mock.bos_token_id = 2
    mock.vocab_size = 262144
    mock.encode = MagicMock(return_value=[2, 2717, 3723, 107])
    mock.decode = MagicMock(return_value='{"route": ["a", "b"], "terminal": "b"}<turn|>')
    return mock


@pytest.fixture
def valid_input_dict() -> Dict[str, Any]:
    prompt = "You are an execution planner. Plan a valid route from a to b."
    prompt_sha = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
    return {
        "id": "sample_001",
        "split": "sealed_test",
        "domain": "api_workflow",
        "difficulty": 1,
        "num_steps": 2,
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
    }


# ==============================================================================
# Test 1: Direct Frozen Token 106 Halting & No Trailing Turns
# ==============================================================================

def test_direct_frozen_halts_on_token_106_and_no_trailing_turns(gemma4_tokenizer):
    """Verify stream_generate halts on token 106 (<turn|>) without trailing thought runaway."""
    import mlx_lm

    # Simulated generation: model outputs JSON tokens -> token 106 (<turn|>) -> runaway thought channel [108, 109, 101]
    simulated_tokens = [2717, 3723, 107, 106, 108, 109, 101]

    def mock_generate_step(prompt, model, **kwargs):
        for tok in simulated_tokens:
            yield tok, mx.zeros((1,))

    # 1. Unwrapped tokenizer (default eos_token_ids={1}) fails to halt on 106
    unwrapped = TokenizerWrapper(gemma4_tokenizer)
    assert unwrapped.eos_token_ids == {1}

    with patch("mlx_lm.generate.generate_step", side_effect=mock_generate_step):
        unwrapped_responses = list(mlx_lm.stream_generate(nn.Module(), unwrapped, prompt=[2]))
        unwrapped_emitted = [r.token for r in unwrapped_responses]
        # Failure mode: emits past 106 into trailing turns
        assert 108 in unwrapped_emitted, "Unwrapped tokenizer failed to simulate trailing turns"
        assert 109 in unwrapped_emitted

    # 2. Wrapped tokenizer with token 106 (eos_token_ids={1, 106}) halts cleanly
    wrapped = GemmaTokenizerWrapper(gemma4_tokenizer, eos_token_ids={1, 106})
    assert 106 in wrapped.eos_token_ids

    with patch("mlx_lm.generate.generate_step", side_effect=mock_generate_step):
        wrapped_responses = list(mlx_lm.stream_generate(nn.Module(), wrapped, prompt=[2]))
        wrapped_emitted = [r.token for r in wrapped_responses]
        # Verified fix: halts immediately on token 106
        assert 108 not in wrapped_emitted, "Wrapped tokenizer emitted trailing thought channel tokens!"
        assert 109 not in wrapped_emitted
        assert wrapped_emitted[-1] == 106, f"Expected last token to be 106, got {wrapped_emitted[-1]}"


# ==============================================================================
# Test 2: Inference Functions Throw on Targets or Answer Keys (Rule 1)
# ==============================================================================

def test_inference_fails_if_targets_or_answer_keys_passed(valid_input_dict):
    """Verify validate_target_free_record and generate_predictions reject forbidden oracle keys."""
    # 1. Valid input passes cleanly
    validate_target_free_record(valid_input_dict)
    eval_input = EvaluationInput.from_dict(valid_input_dict)
    assert eval_input.id == "sample_001"

    # 2. Rejection of every canonical forbidden oracle term at top level
    for forbidden_key in FORBIDDEN_ORACLE_TERMS:
        leaked_record = dict(valid_input_dict)
        leaked_record[forbidden_key] = "leak_payload"
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            validate_target_free_record(leaked_record)
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            EvaluationInput.from_dict(leaked_record)

    # 3. Rejection of nested oracle terms inside metadata or custom dicts
    nested_leaked = dict(valid_input_dict)
    nested_leaked["metadata"] = {"expected_route": ["policy_gate", "audit_logger"]}
    with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
        validate_target_free_record(nested_leaked)
    with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
        EvaluationInput.from_dict(nested_leaked)

    # 4. Rejection of objects like DomainSample or AnswerKey containing oracle attributes
    with tempfile.TemporaryDirectory() as td:
        leaked_file = Path(td) / "leaked_inputs.jsonl"
        with open(leaked_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({**valid_input_dict, "target_solution": '{"route": ["a"]}'}) + "\n")

        with pytest.raises(OracleLeakageError):
            load_target_free_inputs(leaked_file)

        with pytest.raises(OracleLeakageError):
            generate_predictions(
                inputs=leaked_file,
                condition="direct_frozen",
                output_dir=Path(td),
                checkpoint_path=None,
            )


# ==============================================================================
# Test 3: Missing Checkpoint Raises FileNotFoundError (Rule 5)
# ==============================================================================

def test_missing_checkpoint_raises_filenotfound(valid_input_dict):
    """Verify missing adapter checkpoints abort immediately with FileNotFoundError / MissingCheckpointError."""
    with tempfile.TemporaryDirectory() as td:
        input_file = Path(td) / "valid_inputs.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_input_dict) + "\n")

        missing_checkpoint = Path(td) / "nonexistent_adapter_weights.safetensors"

        # Checkpoint is required for adapter conditions
        for cond in ["adapter_t0", "adapter_t1", "adapter_t2", "adapter_t4", "control_zeroed", "control_shuffled", "non_recurrent"]:
            with pytest.raises(FileNotFoundError):
                verify_adapter_checkpoint(missing_checkpoint, condition=cond)

            with pytest.raises(FileNotFoundError):
                generate_predictions(
                    inputs=input_file,
                    condition=cond,
                    output_dir=Path(td),
                    checkpoint_path=missing_checkpoint,
                )

        # Passing None for adapter condition must also raise MissingCheckpointError
        with pytest.raises(MissingCheckpointError):
            verify_adapter_checkpoint(None, condition="adapter_t1")

        with pytest.raises(MissingCheckpointError):
            generate_predictions(
                inputs=input_file,
                condition="adapter_t1",
                output_dir=Path(td),
                checkpoint_path=None,
            )


# ==============================================================================
# Test 4: Post-Hoc Scoring Fails if Prediction File Modified (Rule 2)
# ==============================================================================

def test_post_hoc_scoring_fails_if_prediction_file_modified():
    """Verify score_predictions raises PredictionIntegrityError if prediction file is tampered."""
    with tempfile.TemporaryDirectory() as td:
        pred_dir = Path(td)
        pred_file = pred_dir / "predictions_direct_frozen.json"
        sidecar_file = pred_dir / "predictions_direct_frozen.json.sha256"
        keys_file = pred_dir / "answer_keys.jsonl"

        prompt_str = "Plan route"
        prompt_sha = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()

        # Create valid prediction file
        valid_predictions = {
            "schema_version": "prlr.predictions.v1",
            "metadata": {
                "condition": "direct_frozen",
                "git_commit_sha": "abc1234",
                "is_dirty": False,
                "sample_count": 1,
            },
            "predictions": [{
                "sample_id": "sample_001",
                "domain": "api_workflow",
                "condition": "direct_frozen",
                "recurrence_depth": 0,
                "generated_token_ids": [2717, 3723, 106],
                "decoded_text": '{"route": ["a", "b"], "terminal": "b"}',
                "latency_ms": 1500.0,
                "prompt_sha256": prompt_sha,
                "git_commit_sha": "abc1234",
                "is_dirty": False,
                "checkpoint_sha256": "",
                "model_id": "gemma_4",
                "tokenizer_id": "gemma_4",
            }],
        }
        pred_bytes = json.dumps(valid_predictions, indent=2).encode("utf-8")
        pred_file.write_bytes(pred_bytes)
        expected_sha = hashlib.sha256(pred_bytes).hexdigest()
        sidecar_file.write_text(f"{expected_sha}  {pred_file.name}\n")

        # Create valid quarantined answer keys
        with open(keys_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "sample_001",
                "prompt_sha256": prompt_sha,
                "target_solution": '{"route": ["a", "b"], "terminal": "b"}',
                "ground_truth": '{"route": ["a", "b"], "terminal": "b"}',
                "verifier_config": {"expected_route": ["a", "b"], "terminal_tool": "b"},
            }) + "\n")

        # 1. Verification of untampered file succeeds
        loaded_artifact, actual_sha = verify_prediction_file(pred_file)
        assert actual_sha == expected_sha

        # 2. Tamper with prediction file (modify 1 character)
        tampered_predictions = dict(valid_predictions)
        tampered_predictions["predictions"][0]["decoded_text"] = '{"route": ["a", "TAMPERED"], "terminal": "b"}'
        pred_file.write_bytes(json.dumps(tampered_predictions, indent=2).encode("utf-8"))

        # 3. score_predictions must immediately raise PredictionIntegrityError BEFORE reading answer keys
        with pytest.raises(PredictionIntegrityError, match="Prediction file tampering detected"):
            score_predictions(
                predictions_path=pred_file,
                answer_keys_path=keys_file,
                output_dir=pred_dir,
            )


# ==============================================================================
# Test 5: Prompt Hash Mismatch Detected During Scoring
# ==============================================================================

def test_sample_hash_mismatch_detected_during_scoring():
    """Verify score_predictions raises SampleMismatchError if prompt_sha256 diverges."""
    with tempfile.TemporaryDirectory() as td:
        pred_dir = Path(td)
        pred_file = pred_dir / "predictions_direct_frozen.json"
        keys_file = pred_dir / "answer_keys.jsonl"

        predictions_data = {
            "schema_version": "prlr.predictions.v1",
            "metadata": {"condition": "direct_frozen", "git_commit_sha": "abc1234", "is_dirty": False},
            "predictions": [{
                "sample_id": "sample_001",
                "domain": "api_workflow",
                "condition": "direct_frozen",
                "recurrence_depth": 0,
                "generated_token_ids": [2717, 3723, 106],
                "decoded_text": '{"route": ["a", "b"], "terminal": "b"}',
                "latency_ms": 1500.0,
                "prompt_sha256": "hash_from_predictions",
                "git_commit_sha": "abc1234",
                "is_dirty": False,
                "checkpoint_sha256": "",
                "model_id": "gemma_4",
                "tokenizer_id": "gemma_4",
            }],
        }
        atomic_serialize_prediction_artifact(predictions_data, pred_file)

        # Create answer key with DIFFERENT prompt_sha256
        with open(keys_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "sample_001",
                "prompt_sha256": "different_hash_in_keys",
                "target_solution": '{"route": ["a", "b"], "terminal": "b"}',
                "verifier_config": {"expected_route": ["a", "b"], "terminal_tool": "b"},
            }) + "\n")

        with pytest.raises(SampleMismatchError, match="Prompt SHA-256 mismatch"):
            score_predictions(
                predictions_path=pred_file,
                answer_keys_path=keys_file,
                output_dir=pred_dir,
            )


# ==============================================================================
# Test 6: Cross-Run Summary Consolidation Rejects Mismatched Runs
# ==============================================================================

def test_cross_run_summary_consolidation_rejects_mismatched_runs():
    """Verify validate_can_merge raises CrossRunMergeRejectionError across 5 dimensions."""
    base_provenance = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="sealed_test_inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="sealed_test_keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )

    base_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_provenance,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={
            "direct_frozen": ConditionScoredSummary(
                condition="direct_frozen",
                recurrence_depth=0,
                sample_count=256,
                exact_match_count=247,
                exact_match_pct=96.48,
                terminal_match_count=248,
                terminal_match_pct=96.88,
                valid_json_count=256,
                valid_json_pct=100.0,
                max_4gram_repetition=1,
                mean_4gram_repetition=1.0,
                mean_shannon_entropy=4.2,
                latency=LatencyStatistics(mean_ms=1850.0, median_ms=1840.0, p95_ms=1920.0, min_ms=1500.0, max_ms=2100.0, total_ms=473600.0),
                checkpoint_sha256="",
                model_id="google/gemma-4-12B-it-4bit",
            ),
            "adapter_t1": ConditionScoredSummary(
                condition="adapter_t1",
                recurrence_depth=1,
                sample_count=256,
                exact_match_count=240,
                exact_match_pct=93.75,
                terminal_match_count=242,
                terminal_match_pct=94.53,
                valid_json_count=256,
                valid_json_pct=100.0,
                max_4gram_repetition=1,
                mean_4gram_repetition=1.0,
                mean_shannon_entropy=4.1,
                latency=LatencyStatistics(mean_ms=1900.0, median_ms=1890.0, p95_ms=1980.0, min_ms=1600.0, max_ms=2200.0, total_ms=486400.0),
                checkpoint_sha256="checkpoint_hash_xxx",
                model_id="google/gemma-4-12B-it-4bit",
            ),
        },
    )

    # 1. Matching incoming provenance succeeds
    validate_can_merge(
        existing=base_summary,
        incoming_provenance=base_provenance,
        incoming_condition="repo_decoder",
    )

    # 2. Dimension 1: Mismatched Git Commit SHA
    mismatched_commit = ProvenanceMetadata(**{**base_provenance.to_dict(), "git_commit_sha": "f1e2d3c4b5a60718293a4b5c6d7e8f9012345678"})
    with pytest.raises(CrossRunMergeRejectionError, match="git_commit_sha mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=mismatched_commit,
            incoming_condition="repo_decoder",
        )

    # 3. Dimension 1b: Mismatched Git Dirty Status
    mismatched_dirty = ProvenanceMetadata(**{**base_provenance.to_dict(), "is_dirty": True})
    with pytest.raises(CrossRunMergeRejectionError, match="is_dirty mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=mismatched_dirty,
            incoming_condition="repo_decoder",
        )

    # 4. Dimension 2: Mismatched Dataset Inputs Hash
    mismatched_dataset = ProvenanceMetadata(**{**base_provenance.to_dict(), "inputs_file_sha256": "tampered_dataset_hash"})
    with pytest.raises(CrossRunMergeRejectionError, match="dataset/inputs hash mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=mismatched_dataset,
            incoming_condition="repo_decoder",
        )

    # 5. Dimension 3: Mismatched Sample Count
    mismatched_samples = ProvenanceMetadata(**{**base_provenance.to_dict(), "sample_count": 128})
    with pytest.raises(CrossRunMergeRejectionError, match="sample_count mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=mismatched_samples,
            incoming_condition="repo_decoder",
        )

    # 6. Dimension 4: Mismatched Checkpoint Hash for Adapter Conditions
    with pytest.raises(CrossRunMergeRejectionError, match="checkpoint_sha256 mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=base_provenance,
            incoming_condition="adapter_t2",
            incoming_checkpoint_sha256="different_checkpoint_hash_yyy",
        )

    # 7. Dimension 5: Mismatched Runtime Versions (e.g. MLX version)
    mismatched_runtime = ProvenanceMetadata(**{
        **base_provenance.to_dict(),
        "runtime_versions": {"python": "3.14.4", "mlx": "0.32.0", "transformers": "5.9.0"},
    })
    with pytest.raises(CrossRunMergeRejectionError, match="Runtime version mismatch"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=mismatched_runtime,
            incoming_condition="repo_decoder",
        )


# ==============================================================================
# Test 7: AST Static Analysis on Inference Signatures & Strings
# ==============================================================================

def test_ast_inference_functions_zero_oracle_parameters():
    """Verify AST / signatures of inference callables accept zero oracle parameters."""
    callables = [
        generate_predictions,
        generate_direct_frozen,
        PretrainedGemmaBackbone.encode_prompt_context,
        PretrainedGemmaBackbone.extract_contextual_hiddens,
        GemmaCausalPrefixDecoder.generate,
    ]

    for fn in callables:
        sig = inspect.signature(fn)
        for param_name in sig.parameters:
            assert param_name not in FORBIDDEN_ORACLE_TERMS, (
                f"Rule 1 Violation: Callable {fn.__qualname__} accepts oracle parameter '{param_name}'!"
            )


def test_ast_generate_predictions_no_oracle_string_constants():
    """Verify generate_predictions source code body does not contain oracle string literals."""
    source = inspect.getsource(generate_predictions)
    parsed = ast.parse(source)

    forbidden_literals = {
        "expected_route", "target_solution", "ground_truth",
        "verifier_config", "answer_keys", "oracle_solution",
    }

    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_lower = node.value.lower()
            for forbidden in forbidden_literals:
                assert forbidden not in val_lower, (
                    f"Rule 1 Violation: generate_predictions body contains oracle literal '{node.value}'!"
                )


# ==============================================================================
# Test 8: Dataclass Invariants & Immutability
# ==============================================================================

def test_prediction_record_immutability():
    """Verify PredictionRecord is frozen and rejects mutation."""
    record = PredictionRecord(
        sample_id="s1",
        domain="api_workflow",
        condition="direct_frozen",
        recurrence_depth=0,
        generated_token_ids=(2717, 3723, 106),
        decoded_text="{}",
        latency_ms=100.0,
        prompt_sha256="abc",
        git_commit_sha="commit_1",
        is_dirty=False,
        checkpoint_sha256="",
        model_id="gemma_4",
        tokenizer_id="gemma_4",
    )
    with pytest.raises(FrozenInstanceError):
        record.latency_ms = 200.0  # type: ignore


def test_evaluation_input_immutability_and_validation(valid_input_dict):
    """Verify EvaluationInput is frozen and validates prompt hash."""
    inp = EvaluationInput.from_dict(valid_input_dict)
    with pytest.raises(FrozenInstanceError):
        inp.prompt = "mutated"  # type: ignore

    # Mismatched prompt hash triggers ValueError
    tampered_dict = dict(valid_input_dict)
    tampered_dict["prompt_sha256"] = "wrong_hash"
    with pytest.raises(ValueError, match="Prompt SHA-256 mismatch"):
        EvaluationInput.from_dict(tampered_dict)


# ==============================================================================
# Test 9: Atomic Prediction Serialization & SHA-256 Sidecars
# ==============================================================================

def test_atomic_prediction_serialization_with_sha256_sidecar():
    """Verify atomic_serialize_prediction_artifact writes JSON and valid sidecar."""
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "predictions_test.json"
        data = {"sample": 1, "status": "ok"}

        p_path, sidecar_path, sha = atomic_serialize_prediction_artifact(data, out_file)
        assert p_path.exists()
        assert sidecar_path.exists()

        file_bytes = p_path.read_bytes()
        computed_sha = hashlib.sha256(file_bytes).hexdigest()
        assert computed_sha == sha

        sidecar_text = sidecar_path.read_text(encoding="utf-8")
        assert sha in sidecar_text


# ==============================================================================
# Test 10: Key Normalization Rejection of CamelCase, Kebab, Stealth & Nested (Defect 1)
# ==============================================================================

def test_key_normalization_rejection_camel_kebab_stealth_and_nested():
    """Verify validate_target_free_dict rejects camelCase, kebab-case, compound stealth, and nested keys."""
    # 1. CamelCase variations
    camel_keys = [
        "targetSolution", "groundTruth", "expectedRoute",
        "verifierConfig", "answerKey", "oracleSolution",
    ]
    for k in camel_keys:
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            validate_target_free_dict({k: "leak"})

    # 2. Kebab-case variations
    kebab_keys = [
        "target-solution", "ground-truth", "expected-route",
        "verifier-config", "answer-key", "oracle-solution",
    ]
    for k in kebab_keys:
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            validate_target_free_dict({k: "leak"})

    # 3. Compound stealth keys containing oracle root words
    stealth_compound_keys = [
        "custom_verifier", "final_solution", "model_answer",
        "user_target", "oracle_path", "gold_answer", "verifier_spec",
    ]
    for k in stealth_compound_keys:
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            validate_target_free_dict({k: "leak"})

    # 4. Squashed delimiter-free stems
    squashed_keys = [
        "targetsolution", "modelanswer", "customverifier", "groundtruth",
    ]
    for k in squashed_keys:
        with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
            validate_target_free_dict({k: "leak"})

    # 5. Deeply nested camelCase and snake_case structures
    nested_structure = {
        "meta": [{"step": 1, "hints": {"expectedRoute": ["auth", "process"]}}]
    }
    with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
        validate_target_free_dict(nested_structure)

    deeply_nested_list = [
        {"pipeline": [{"stage": {"target_solution": [1, 2, 3]}}]}
    ]
    with pytest.raises(OracleLeakageError, match="Evidence Rule 1 Violation"):
        validate_target_free_dict(deeply_nested_list)


# ==============================================================================
# Test 11: Legitimate Keys Pass Validation Without False Positives (Defect 1)
# ==============================================================================

def test_legitimate_keys_pass_validation():
    """Verify legitimate schema keys (including resolution, background) pass cleanly."""
    legitimate_dict = {
        "id": "sample_001",
        "split": "sealed_test",
        "domain": "api_workflow",
        "difficulty": 1,
        "num_steps": 2,
        "prompt": "Legitimate user prompt",
        "prompt_sha256": "abcdef0123456789",
        "metadata": {
            "display_resolution": "1920x1080",
            "screen_resolution": "4k",
            "conflict_resolution": "strict",
            "background": "dark",
            "background_color": "#000000",
            "config": {"timeout": 30},
            "tools": ["tool_a", "tool_b"],
            "operation_id": "op_99",
        },
    }
    validate_target_free_dict(legitimate_dict)
    is_f, _ = is_forbidden_oracle_key("display_resolution")
    assert not is_f
    is_f, _ = is_forbidden_oracle_key("background")
    assert not is_f


# ==============================================================================
# Test 12: Missing Sidecar Rejection (Defect 2)
# ==============================================================================

def test_missing_sidecar_raises_prediction_integrity_error():
    """Verify verify_prediction_file and score_predictions reject missing sidecar (Defect 2)."""
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td)
        pred_file = p_dir / "predictions_direct_frozen.json"
        keys_file = p_dir / "answer_keys.jsonl"

        artifact_data = {
            "schema_version": "prlr.predictions.v1",
            "metadata": {"condition": "direct_frozen"},
            "predictions": [],
        }
        pred_file.write_text(json.dumps(artifact_data))
        keys_file.write_text("{}\n")

        with pytest.raises(PredictionIntegrityError, match="Mandatory SHA-256 sidecar missing"):
            verify_prediction_file(pred_file)

        with pytest.raises(PredictionIntegrityError, match="Mandatory SHA-256 sidecar missing"):
            score_predictions(pred_file, keys_file, output_dir=p_dir)


# ==============================================================================
# Test 13: Empty and Malformed Sidecar Rejection (Defect 5)
# ==============================================================================

def test_empty_and_whitespace_sidecar_raises_prediction_integrity_error():
    """Verify zero-byte and whitespace sidecars raise PredictionIntegrityError instead of IndexError (Defect 5)."""
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td)
        pred_file = p_dir / "predictions_direct_frozen.json"
        sidecar_file = p_dir / "predictions_direct_frozen.json.sha256"
        keys_file = p_dir / "answer_keys.jsonl"

        artifact_data = {
            "schema_version": "prlr.predictions.v1",
            "metadata": {"condition": "direct_frozen"},
            "predictions": [],
        }
        pred_file.write_text(json.dumps(artifact_data))
        keys_file.write_text("{}\n")

        # 1. Zero-byte sidecar
        sidecar_file.write_text("")
        with pytest.raises(PredictionIntegrityError, match="Empty or zero-byte SHA-256 sidecar file"):
            verify_prediction_file(pred_file)
        with pytest.raises(PredictionIntegrityError, match="Empty or zero-byte SHA-256 sidecar file"):
            score_predictions(pred_file, keys_file, output_dir=p_dir)

        # 2. Whitespace-only sidecar
        sidecar_file.write_text("   \n\t  \n  ")
        with pytest.raises(PredictionIntegrityError, match="Empty or zero-byte SHA-256 sidecar file"):
            verify_prediction_file(pred_file)


def test_malformed_sidecar_format_and_filename_mismatch():
    """Verify sidecars with truncated hashes, non-hex characters, or mismatched filenames are rejected."""
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td)
        pred_file = p_dir / "predictions_direct_frozen.json"
        sidecar_file = p_dir / "predictions_direct_frozen.json.sha256"

        pred_bytes = json.dumps({"schema_version": "prlr.predictions.v1", "metadata": {}}).encode("utf-8")
        pred_file.write_bytes(pred_bytes)
        correct_sha = hashlib.sha256(pred_bytes).hexdigest()

        # 1. Truncated hash (length != 64)
        sidecar_file.write_text("a" * 32 + f"  {pred_file.name}\n")
        with pytest.raises(PredictionIntegrityError, match="Malformed SHA-256 sidecar file"):
            verify_prediction_file(pred_file)

        # 2. Non-hex characters
        sidecar_file.write_text("z" * 64 + f"  {pred_file.name}\n")
        with pytest.raises(PredictionIntegrityError, match="Malformed SHA-256 sidecar file"):
            verify_prediction_file(pred_file)

        # 3. Filename mismatch
        sidecar_file.write_text(f"{correct_sha}  predictions_other.json\n")
        with pytest.raises(PredictionIntegrityError, match="Sidecar filename mismatch"):
            verify_prediction_file(pred_file)


# ==============================================================================
# Test 14: Anti-Merge Checkpoint Validation for All Recurrent Conditions (Defect 4)
# ==============================================================================

@pytest.mark.parametrize("condition", [
    "adapter_t0",
    "adapter_t1",
    "adapter_t2",
    "adapter_t4",
    "control_zeroed",
    "control_shuffled",
    "control_random",
])
def test_anti_merge_rejects_mismatched_checkpoint_for_recurrent_conditions(condition):
    """Verify validate_can_merge rejects mismatched checkpoint for ALL recurrent conditions."""
    base_prov = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    base_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_prov,
        prediction_files_sha256={"adapter_t1": "sha_t1"},
        conditions={
            "adapter_t1": ConditionScoredSummary(
                condition="adapter_t1",
                recurrence_depth=1,
                sample_count=256,
                exact_match_count=240,
                exact_match_pct=93.75,
                terminal_match_count=242,
                terminal_match_pct=94.53,
                valid_json_count=256,
                valid_json_pct=100.0,
                max_4gram_repetition=1,
                mean_4gram_repetition=1.0,
                mean_shannon_entropy=4.1,
                latency=LatencyStatistics(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                checkpoint_sha256="1111111111111111111111111111111111111111111111111111111111111111",
                model_id="google/gemma-4-12B-it-4bit",
            ),
        },
    )

    # 1. Matching checkpoint succeeds
    validate_can_merge(
        existing=base_summary,
        incoming_provenance=base_prov,
        incoming_condition=condition,
        incoming_checkpoint_sha256="1111111111111111111111111111111111111111111111111111111111111111",
    )

    # 2. Mismatched checkpoint raises CrossRunMergeRejectionError
    with pytest.raises(CrossRunMergeRejectionError, match="checkpoint_sha256 mismatch for recurrent adapter"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=base_prov,
            incoming_condition=condition,
            incoming_checkpoint_sha256="2222222222222222222222222222222222222222222222222222222222222222",
        )

    # 3. Empty checkpoint raises CrossRunMergeRejectionError
    with pytest.raises(CrossRunMergeRejectionError, match="Missing required checkpoint_sha256"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=base_prov,
            incoming_condition=condition,
            incoming_checkpoint_sha256="",
        )


# ==============================================================================
# Test 15: Anti-Merge Checkpoint Validation for Non-Recurrent Condition (Defect 4)
# ==============================================================================

def test_anti_merge_checkpoint_validation_for_non_recurrent():
    """Verify validate_can_merge enforces checkpoint hash for non_recurrent condition."""
    base_prov = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    non_rec_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    recurrent_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_prov,
        prediction_files_sha256={"adapter_t1": "sha_t1", "non_recurrent": "sha_nr"},
        conditions={
            "adapter_t1": ConditionScoredSummary(
                condition="adapter_t1",
                recurrence_depth=1,
                sample_count=256,
                exact_match_count=240,
                exact_match_pct=93.75,
                terminal_match_count=242,
                terminal_match_pct=94.53,
                valid_json_count=256,
                valid_json_pct=100.0,
                max_4gram_repetition=1,
                mean_4gram_repetition=1.0,
                mean_shannon_entropy=4.1,
                latency=LatencyStatistics(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                checkpoint_sha256=recurrent_sha,
                model_id="google/gemma-4-12B-it-4bit",
            ),
            "non_recurrent": ConditionScoredSummary(
                condition="non_recurrent",
                recurrence_depth=0,
                sample_count=256,
                exact_match_count=235,
                exact_match_pct=91.80,
                terminal_match_count=238,
                terminal_match_pct=92.97,
                valid_json_count=256,
                valid_json_pct=100.0,
                max_4gram_repetition=1,
                mean_4gram_repetition=1.0,
                mean_shannon_entropy=4.0,
                latency=LatencyStatistics(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                checkpoint_sha256=non_rec_sha,
                model_id="google/gemma-4-12B-it-4bit",
            ),
        },
    )

    # 1. Non-recurrent matching its own checkpoint succeeds
    validate_can_merge(
        existing=summary,
        incoming_provenance=base_prov,
        incoming_condition="non_recurrent",
        incoming_checkpoint_sha256=non_rec_sha,
    )

    # 2. Non-recurrent with mismatched checkpoint is rejected
    with pytest.raises(CrossRunMergeRejectionError, match="checkpoint_sha256 mismatch for non-recurrent"):
        validate_can_merge(
            existing=summary,
            incoming_provenance=base_prov,
            incoming_condition="non_recurrent",
            incoming_checkpoint_sha256="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        )

    # 3. Non-recurrent with empty checkpoint is rejected
    with pytest.raises(CrossRunMergeRejectionError, match="Missing required checkpoint_sha256"):
        validate_can_merge(
            existing=summary,
            incoming_provenance=base_prov,
            incoming_condition="non_recurrent",
            incoming_checkpoint_sha256="",
        )


# ==============================================================================
# Test 16: Anti-Merge Rejection of Empty and Whitespace Commit SHAs (Defect 6)
# ==============================================================================

@pytest.mark.parametrize("bad_commit", ["", "   ", "\t\n", "  \n  "])
def test_anti_merge_rejects_empty_and_whitespace_commit_sha(bad_commit):
    """Verify validate_can_merge rejects empty and whitespace-only git_commit_sha."""
    base_prov = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    base_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_prov,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={},
    )

    bad_in_prov = ProvenanceMetadata(**{**base_prov.to_dict(), "git_commit_sha": bad_commit})

    with pytest.raises(CrossRunMergeRejectionError, match="empty or whitespace-only"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=bad_in_prov,
            incoming_condition="repo_decoder",
        )


# ==============================================================================
# Test 17: Anti-Merge Rejection of Unknown Commit SHAs & Offline Bypass (Defect 6)
# ==============================================================================

def test_anti_merge_rejects_unknown_commit_sha_unless_offline_flag(monkeypatch):
    """Verify 'unknown' git_commit_sha is rejected unless offline development flag is active."""
    base_prov = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    base_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_prov,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={},
    )

    unknown_prov = ProvenanceMetadata(**{**base_prov.to_dict(), "git_commit_sha": "unknown"})

    # 1. Standard mode: Rejects 'unknown'
    monkeypatch.delenv("PRLR_ALLOW_OFFLINE_GIT", raising=False)
    with pytest.raises(CrossRunMergeRejectionError, match="unversioned runs cannot be merged"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=unknown_prov,
            incoming_condition="repo_decoder",
            allow_offline_git=False,
        )

    # 2. Offline dev mode with explicit parameter: Permits 'unknown' format
    unknown_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=unknown_prov,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={},
    )
    validate_can_merge(
        existing=unknown_summary,
        incoming_provenance=unknown_prov,
        incoming_condition="repo_decoder",
        allow_offline_git=True,
    )

    # 3. Offline dev mode via environment variable succeeds
    monkeypatch.setenv("PRLR_ALLOW_OFFLINE_GIT", "1")
    validate_can_merge(
        existing=unknown_summary,
        incoming_provenance=unknown_prov,
        incoming_condition="repo_decoder",
    )


# ==============================================================================
# Test 18: Strict Git Commit Format Validation (Defect 6)
# ==============================================================================

@pytest.mark.parametrize("invalid_sha", [
    "123456",            # Too short (< 7 chars)
    "abcdefg",           # Non-hex char ('g')
    "not_a_git_commit!", # Symbols
    "0" * 65,            # Too long (> 64 chars)
])
def test_anti_merge_strict_format_validation(invalid_sha):
    """Verify non-hex or invalid-length commit hashes are rejected under strict mode."""
    base_prov = ProvenanceMetadata(
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    base_summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=base_prov,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={},
    )

    invalid_prov = ProvenanceMetadata(**{**base_prov.to_dict(), "git_commit_sha": invalid_sha})
    with pytest.raises(CrossRunMergeRejectionError, match="invalid format: must be 7-64 hexadecimal"):
        validate_can_merge(
            existing=base_summary,
            incoming_provenance=invalid_prov,
            incoming_condition="repo_decoder",
            allow_offline_git=False,
        )


@pytest.mark.parametrize("valid_sha", [
    "a1b2c3d",                                            # 7 hex chars (abbreviated)
    "0123456789abcdef",                                   # 16 hex chars
    "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",           # 40 hex chars (standard SHA-1)
    "a" * 64,                                             # 64 hex chars (standard SHA-256)
])
def test_anti_merge_accepts_valid_hex_commit_formats(valid_sha):
    """Verify valid short and full hexadecimal commit hashes are accepted."""
    prov = ProvenanceMetadata(
        git_commit_sha=valid_sha,
        is_dirty=False,
        dataset_name="prlr_domain_v1",
        dataset_sha256="dataset_hash_111",
        split="sealed_test",
        sample_count=256,
        inputs_file="inputs.jsonl",
        inputs_file_sha256="dataset_hash_111",
        keys_file="keys.jsonl",
        keys_file_sha256="keys_hash_222",
        runtime_versions={"python": "3.14.4", "mlx": "0.31.2", "transformers": "5.9.0"},
        hardware_info={"platform": "darwin"},
    )
    summary = ScoredSummaryArtifact(
        schema_version="prlr.scored_summary.v1",
        created_at_utc="2026-09-04T12:00:00Z",
        provenance=prov,
        prediction_files_sha256={"direct_frozen": "sha_df"},
        conditions={},
    )
    validate_can_merge(
        existing=summary,
        incoming_provenance=prov,
        incoming_condition="repo_decoder",
        allow_offline_git=False,
    )

