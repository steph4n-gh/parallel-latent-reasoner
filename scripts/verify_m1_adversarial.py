#!/usr/bin/env python3
"""Milestone M1 Adversarial Verification Suite.

Adversarially challenge and stress-test the Milestone 1 target-free isolation,
cryptographic validation, and cross-run merge rejection in projects/parallel_latent_reasoner.

Conforms to:
- Evidence Rule 1: Ground-truth isolation & oracle leakage prevention.
- Evidence Rule 2: Cryptographic tamper detection (SHA-256 sidecars).
- Evidence Rule 5: Checkpoint integrity & fail-fast guards.
- Evidence Rule 10: Provenance tracking & anti-merge guards.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

# Ensure src is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx

from prlr.eval.harness import (
    ConditionScoredSummary,
    CrossRunMergeRejectionError,
    EvaluationInput,
    LatencyStatistics,
    MissingCheckpointError,
    OracleLeakageError,
    PredictionIntegrityError,
    PredictionRecord,
    ProvenanceMetadata,
    SampleMismatchError,
    ScoredSummaryArtifact,
    atomic_serialize_prediction_artifact,
    generate_predictions,
    score_predictions,
    validate_can_merge,
    validate_target_free_dict,
    validate_target_free_record,
    verify_prediction_file,
)


class AdversarialTestRunner:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def record(self, section: str, test_name: str, passed: bool, detail: str, severity: str = "HIGH") -> None:
        self.results.append({
            "section": section,
            "test_name": test_name,
            "passed": passed,
            "detail": detail,
            "severity": severity,
        })
        status_str = "✅ PASS" if passed else f"❌ FAIL ({severity})"
        print(f"[{status_str}] {section} :: {test_name}\n   -> {detail}")

    # ==========================================================================
    # Section 1: Target-Free Isolation & Oracle Leakage
    # ==========================================================================
    def test_section_1_oracle_leakage(self) -> None:
        print("\n=== RUNNING SECTION 1: Oracle Leakage & Stealth Key Stress Tests ===")

        # 1.1 Canonical Snake Case Leaks
        canonical_keys = [
            "target_solution", "ground_truth", "expected_route", "verifier_config",
            "terminal_tool", "target_goal", "answer_key", "oracle_solution",
        ]
        all_canonical_blocked = True
        failed_canonical = []
        for k in canonical_keys:
            try:
                validate_target_free_dict({k: "leak_value"})
                all_canonical_blocked = False
                failed_canonical.append(k)
            except OracleLeakageError:
                pass
        self.record(
            "Section 1",
            "Canonical Snake Case Rejection",
            all_canonical_blocked,
            f"Blocked all {len(canonical_keys)} keys." if all_canonical_blocked else f"Allowed keys: {failed_canonical}",
            severity="CRITICAL",
        )

        # 1.2 CamelCase Variations
        camel_keys = [
            "targetSolution", "groundTruth", "expectedRoute",
            "verifierConfig", "answerKey", "oracleSolution",
        ]
        camel_blocked = []
        camel_leaked = []
        for k in camel_keys:
            try:
                validate_target_free_dict({k: "leak_value"})
                camel_leaked.append(k)
            except OracleLeakageError:
                camel_blocked.append(k)
        self.record(
            "Section 1",
            "CamelCase Oracle Key Variations",
            len(camel_leaked) == 0,
            f"Leaked keys: {camel_leaked}" if camel_leaked else "All camelCase variants blocked.",
            severity="CRITICAL",
        )

        # 1.3 Kebab-Case Variations
        kebab_keys = [
            "target-solution", "ground-truth", "expected-route",
            "verifier-config", "answer-key", "oracle-solution",
        ]
        kebab_blocked = []
        kebab_leaked = []
        for k in kebab_keys:
            try:
                validate_target_free_dict({k: "leak_value"})
                kebab_leaked.append(k)
            except OracleLeakageError:
                kebab_blocked.append(k)
        self.record(
            "Section 1",
            "Kebab-Case Oracle Key Variations",
            len(kebab_leaked) == 0,
            f"Leaked keys: {kebab_leaked}" if kebab_leaked else "All kebab-case variants blocked.",
            severity="CRITICAL",
        )

        # 1.4 Compound Stealth Keys with Generic Oracle Words
        stealth_compound_keys = [
            "custom_verifier", "final_solution", "model_answer",
            "user_target", "oracle_path", "gold_answer", "verifier_spec",
        ]
        stealth_leaked = []
        for k in stealth_compound_keys:
            try:
                validate_target_free_dict({k: "leak_value"})
                stealth_leaked.append(k)
            except OracleLeakageError:
                pass
        self.record(
            "Section 1",
            "Compound Stealth Keys with Oracle Words",
            len(stealth_leaked) == 0,
            f"Leaked stealth keys: {stealth_leaked}" if stealth_leaked else "All compound stealth keys blocked.",
            severity="HIGH",
        )

        # 1.5 Deeply Nested Structures (lists, tuples, nested dicts)
        nested_structure_pass = True
        nested_error = ""
        try:
            deep_leak = {
                "metadata": {
                    "pipeline": {
                        "stages": [
                            {"name": "prefill"},
                            {"name": "deliberation", "debug": {"target_solution": [1, 2, 3]}},
                        ]
                    }
                }
            }
            validate_target_free_dict(deep_leak)
            nested_structure_pass = False
            nested_error = "Deeply nested snake_case target_solution was NOT blocked!"
        except OracleLeakageError:
            nested_structure_pass = True
            nested_error = "Correctly blocked deeply nested snake_case key."
        except Exception as e:
            nested_structure_pass = False
            nested_error = f"Unexpected exception: {e}"
        self.record(
            "Section 1",
            "Deeply Nested Snake Case Oracle Detection",
            nested_structure_pass,
            nested_error,
            severity="HIGH",
        )

        # 1.6 Deeply Nested CamelCase Structure
        deep_camel_pass = True
        deep_camel_detail = ""
        try:
            deep_camel_leak = {
                "meta": [{"step": 1, "hints": {"expectedRoute": ["auth", "process"]}}]
            }
            validate_target_free_dict(deep_camel_leak)
            deep_camel_pass = False
            deep_camel_detail = "Deeply nested camelCase expectedRoute was NOT blocked!"
        except OracleLeakageError:
            deep_camel_pass = True
            deep_camel_detail = "Correctly blocked deeply nested camelCase key."
        self.record(
            "Section 1",
            "Deeply Nested CamelCase Oracle Detection",
            deep_camel_pass,
            deep_camel_detail,
            severity="HIGH",
        )

        # 1.7 generate_predictions() Intake Protection against CamelCase Leaks
        gen_pred_leak_blocked = True
        gen_pred_detail = ""
        with tempfile.TemporaryDirectory() as td:
            inputs = [{
                "id": "sample_adv_01",
                "domain": "api_workflow",
                "difficulty": 1,
                "num_steps": 1,
                "prompt": "Test prompt for adversarial leakage",
                "targetSolution": "leaked_solution_tokens",
            }]
            mock_backbone = MagicMock()
            mock_backbone.encode_prompt_context.return_value = (mx.array([1, 2]), None)
            mock_backbone.tokenizer.decode.return_value = "{}"
            mock_backbone.tokenizer.eos_token_ids = {1, 106}
            mock_backbone.manifest.model_id = "test_model"
            mock_backbone.manifest.tokenizer_id = "test_tok"

            mock_resp = MagicMock()
            mock_resp.token = 106
            mock_resp.text = "{}"

            with patch("mlx_lm.stream_generate", return_value=[mock_resp]):
                try:
                    generate_predictions(
                        inputs=inputs,
                        condition="direct_frozen",
                        output_dir=Path(td),
                        backbone=mock_backbone,
                    )
                    gen_pred_leak_blocked = False
                    gen_pred_detail = "generate_predictions() accepted input containing 'targetSolution' and ran inference without OracleLeakageError!"
                except OracleLeakageError:
                    gen_pred_leak_blocked = True
                    gen_pred_detail = "generate_predictions() rejected input with OracleLeakageError."
                except Exception as e:
                    gen_pred_leak_blocked = False
                    gen_pred_detail = f"generate_predictions() raised unexpected error: {type(e).__name__}: {e}"

        self.record(
            "Section 1",
            "generate_predictions() Intake Filtration",
            gen_pred_leak_blocked,
            gen_pred_detail,
            severity="CRITICAL",
        )

    # ==========================================================================
    # Section 2: Cryptographic Validation & Tamper Detection
    # ==========================================================================
    def test_section_2_tamper_detection(self) -> None:
        print("\n=== RUNNING SECTION 2: Prediction File & Sidecar Tamper Stress Tests ===")

        prompt_str = "Adversarial test prompt"
        prompt_sha = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()

        def make_valid_bundle(td_path: Path):
            pred_file = td_path / "predictions_direct_frozen.json"
            keys_file = td_path / "quarantined_keys.jsonl"
            pred_data = {
                "schema_version": "prlr.predictions.v1",
                "metadata": {
                    "condition": "direct_frozen",
                    "recurrence_depth": 0,
                    "sample_count": 1,
                    "git_commit_sha": "abc1234",
                    "is_dirty": False,
                    "checkpoint_sha256": "",
                    "model_id": "google/gemma-4-12B-it-4bit",
                    "tokenizer_id": "google/gemma-4-12B-it",
                    "inputs_file": "inputs.jsonl",
                    "inputs_file_sha256": "input_sha_xxx",
                    "hardware_info": {},
                    "runtime_versions": {"python": "3.11.0", "mlx": "0.31.0", "transformers": "4.45.0"},
                },
                "predictions": [{
                    "sample_id": "sample_001",
                    "domain": "api_workflow",
                    "condition": "direct_frozen",
                    "recurrence_depth": 0,
                    "generated_token_ids": [2717, 3723, 106],
                    "decoded_text": '{"route": ["a", "b"], "terminal": "b"}',
                    "latency_ms": 100.0,
                    "prompt_sha256": prompt_sha,
                    "git_commit_sha": "abc1234",
                    "is_dirty": False,
                    "checkpoint_sha256": "",
                    "model_id": "google/gemma-4-12B-it-4bit",
                    "tokenizer_id": "google/gemma-4-12B-it",
                }],
            }
            p_path, s_path, sha = atomic_serialize_prediction_artifact(pred_data, pred_file)
            keys_file.write_text(json.dumps({
                "id": "sample_001",
                "prompt_sha256": prompt_sha,
                "target_solution": '{"route": ["a", "b"], "terminal": "b"}',
                "verifier_config": {"expected_route": ["a", "b"], "terminal_tool": "b"},
            }) + "\n")
            return p_path, s_path, keys_file, pred_data

        # 2.1 Untampered Baseline
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, _ = make_valid_bundle(Path(td))
            try:
                summary, _ = score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Untampered Baseline Scoring",
                    True,
                    f"Scored successfully with EM={summary.conditions['direct_frozen'].exact_match_pct}%",
                    severity="LOW",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Untampered Baseline Scoring",
                    False,
                    f"Failed untampered scoring: {e}",
                    severity="CRITICAL",
                )

        # 2.2 Single-Byte Tamper in Prediction File (Sidecar Intact)
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, _ = make_valid_bundle(Path(td))
            raw_bytes = p_path.read_bytes()
            # Flip bytes in sample_id
            tampered_bytes = raw_bytes.replace(b'sample_001', b'sample_999')
            assert tampered_bytes != raw_bytes, "Failed to tamper bytes"
            p_path.write_bytes(tampered_bytes)

            keys_read_attempted = False
            try:
                # Wrap keys file to detect if load_quarantined_answer_keys was called
                score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Single-Byte Tamper Detection",
                    False,
                    "score_predictions() failed to detect single-byte tamper in prediction file!",
                    severity="CRITICAL",
                )
            except PredictionIntegrityError:
                self.record(
                    "Section 2",
                    "Single-Byte Tamper Detection",
                    True,
                    "Detected single-byte tamper and aborted with PredictionIntegrityError before scoring.",
                    severity="CRITICAL",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Single-Byte Tamper Detection",
                    False,
                    f"Raised wrong exception: {type(e).__name__}: {e}",
                    severity="HIGH",
                )

        # 2.3 Modified SHA-256 Sidecar (Prediction File Intact)
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, _ = make_valid_bundle(Path(td))
            # Overwrite sidecar with forged hash
            s_path.write_text("0000000000000000000000000000000000000000000000000000000000000000  predictions_direct_frozen.json\n")
            try:
                score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Modified Sidecar Hash Detection",
                    False,
                    "score_predictions() failed to detect modified SHA-256 sidecar!",
                    severity="CRITICAL",
                )
            except PredictionIntegrityError:
                self.record(
                    "Section 2",
                    "Modified Sidecar Hash Detection",
                    True,
                    "Detected modified SHA-256 sidecar and aborted with PredictionIntegrityError.",
                    severity="CRITICAL",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Modified Sidecar Hash Detection",
                    False,
                    f"Raised wrong exception: {type(e).__name__}: {e}",
                    severity="HIGH",
                )

        # 2.4 Deleted SHA-256 Sidecar
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, _ = make_valid_bundle(Path(td))
            # Delete sidecar
            s_path.unlink()
            assert not s_path.exists()

            try:
                score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Deleted Sidecar Detection & Abort",
                    False,
                    "CRITICAL SECURITY HOLE: score_predictions() proceeded and opened answer keys when SHA-256 sidecar was deleted!",
                    severity="CRITICAL",
                )
            except PredictionIntegrityError:
                self.record(
                    "Section 2",
                    "Deleted Sidecar Detection & Abort",
                    True,
                    "Correctly aborted with PredictionIntegrityError when sidecar was missing.",
                    severity="CRITICAL",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Deleted Sidecar Detection & Abort",
                    False,
                    f"Raised unexpected exception: {type(e).__name__}: {e}",
                    severity="HIGH",
                )

        # 2.5 Empty (Zero-Byte) SHA-256 Sidecar
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, _ = make_valid_bundle(Path(td))
            s_path.write_text("")  # 0 bytes
            try:
                score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Empty Sidecar Handling",
                    False,
                    "Allowed empty sidecar without error.",
                    severity="HIGH",
                )
            except PredictionIntegrityError:
                self.record(
                    "Section 2",
                    "Empty Sidecar Handling",
                    True,
                    "Correctly raised PredictionIntegrityError on empty sidecar.",
                    severity="MEDIUM",
                )
            except IndexError as e:
                self.record(
                    "Section 2",
                    "Empty Sidecar Handling",
                    False,
                    f"Crashed with unhandled IndexError instead of PredictionIntegrityError: {e}",
                    severity="MEDIUM",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Empty Sidecar Handling",
                    False,
                    f"Raised {type(e).__name__}: {e}",
                    severity="MEDIUM",
                )

        # 2.6 Post-Hoc Prompt Hash Mismatch Check
        with tempfile.TemporaryDirectory() as td:
            p_path, s_path, keys_file, pred_data = make_valid_bundle(Path(td))
            # Modify key prompt_sha256
            keys_file.write_text(json.dumps({
                "id": "sample_001",
                "prompt_sha256": "tampered_prompt_sha",
                "target_solution": '{"route": ["a", "b"], "terminal": "b"}',
                "verifier_config": {"expected_route": ["a", "b"], "terminal_tool": "b"},
            }) + "\n")
            try:
                score_predictions(p_path, keys_file, output_dir=Path(td))
                self.record(
                    "Section 2",
                    "Prompt Hash Mismatch Check",
                    False,
                    "score_predictions() accepted mismatched prompt_sha256!",
                    severity="HIGH",
                )
            except SampleMismatchError:
                self.record(
                    "Section 2",
                    "Prompt Hash Mismatch Check",
                    True,
                    "Rejected mismatched prompt_sha256 with SampleMismatchError.",
                    severity="MEDIUM",
                )
            except Exception as e:
                self.record(
                    "Section 2",
                    "Prompt Hash Mismatch Check",
                    False,
                    f"Raised {type(e).__name__}: {e}",
                    severity="MEDIUM",
                )

    # ==========================================================================
    # Section 3: Cross-Run Merge Rejection (Anti-Merge Guards)
    # ==========================================================================
    def test_section_3_cross_run_anti_merge(self) -> None:
        print("\n=== RUNNING SECTION 3: Cross-Run Anti-Merge Guard Stress Tests ===")

        base_prov = ProvenanceMetadata(
            git_commit_sha="commit_111111",
            is_dirty=False,
            dataset_name="prlr_domain_v1",
            dataset_sha256="dataset_hash_aaa",
            split="sealed_test",
            sample_count=256,
            inputs_file="inputs.jsonl",
            inputs_file_sha256="dataset_hash_aaa",
            keys_file="keys.jsonl",
            keys_file_sha256="keys_hash_bbb",
            runtime_versions={"python": "3.11.0", "mlx": "0.31.0", "transformers": "4.45.0"},
            hardware_info={"platform": "darwin", "mlx_device": "gpu"},
        )

        base_summary = ScoredSummaryArtifact(
            schema_version="prlr.scored_summary.v1",
            created_at_utc="2026-09-04T12:00:00Z",
            provenance=base_prov,
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
                    latency=LatencyStatistics(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
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
                    latency=LatencyStatistics(120.0, 120.0, 120.0, 120.0, 120.0, 120.0),
                    checkpoint_sha256="checkpoint_hash_111",
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
                    latency=LatencyStatistics(110.0, 110.0, 110.0, 110.0, 110.0, 110.0),
                    checkpoint_sha256="checkpoint_non_recurrent_111",
                    model_id="google/gemma-4-12B-it-4bit",
                ),
            },
        )

        # 3.1 Git Commit Mismatch
        mismatched_commit = ProvenanceMetadata(**{**base_prov.to_dict(), "git_commit_sha": "commit_222222"})
        try:
            validate_can_merge(base_summary, mismatched_commit, "repo_decoder")
            self.record("Section 3", "Mismatched Git Commit Rejection", False, "Allowed merge with different git_commit_sha!", "CRITICAL")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched Git Commit Rejection", True, "Successfully rejected git_commit_sha mismatch.", "HIGH")

        # 3.2 Git Dirty Status Mismatch
        mismatched_dirty = ProvenanceMetadata(**{**base_prov.to_dict(), "is_dirty": True})
        try:
            validate_can_merge(base_summary, mismatched_dirty, "repo_decoder")
            self.record("Section 3", "Mismatched Git Dirty Status Rejection", False, "Allowed merge with different is_dirty status!", "HIGH")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched Git Dirty Status Rejection", True, "Successfully rejected is_dirty mismatch.", "HIGH")

        # 3.3 Dataset Hash Mismatch
        mismatched_dataset = ProvenanceMetadata(**{
            **base_prov.to_dict(),
            "inputs_file_sha256": "tampered_dataset_sha",
            "dataset_sha256": "tampered_dataset_sha",
        })
        try:
            validate_can_merge(base_summary, mismatched_dataset, "repo_decoder")
            self.record("Section 3", "Mismatched Dataset SHA Rejection", False, "Allowed merge with different dataset SHA!", "CRITICAL")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched Dataset SHA Rejection", True, "Successfully rejected dataset SHA mismatch.", "HIGH")

        # 3.4 Sample Count Mismatch
        mismatched_samples = ProvenanceMetadata(**{**base_prov.to_dict(), "sample_count": 128})
        try:
            validate_can_merge(base_summary, mismatched_samples, "repo_decoder")
            self.record("Section 3", "Mismatched Sample Count Rejection", False, "Allowed merge with different sample_count!", "HIGH")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched Sample Count Rejection", True, "Successfully rejected sample_count mismatch.", "HIGH")

        # 3.5 Runtime Environment Mismatch (MLX Version)
        mismatched_mlx = ProvenanceMetadata(**{
            **base_prov.to_dict(),
            "runtime_versions": {"python": "3.11.0", "mlx": "0.32.0", "transformers": "4.45.0"},
        })
        try:
            validate_can_merge(base_summary, mismatched_mlx, "repo_decoder")
            self.record("Section 3", "Mismatched MLX Version Rejection", False, "Allowed merge with different MLX runtime version!", "HIGH")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched MLX Version Rejection", True, "Successfully rejected MLX version mismatch.", "HIGH")

        # 3.6 Adapter Checkpoint Mismatch for adapter_t2
        try:
            validate_can_merge(
                base_summary,
                base_prov,
                incoming_condition="adapter_t2",
                incoming_checkpoint_sha256="checkpoint_hash_DIFFERENT",
            )
            self.record("Section 3", "Mismatched Checkpoint for adapter_t2 Rejection", False, "Allowed adapter_t2 with mismatched checkpoint_sha256!", "CRITICAL")
        except CrossRunMergeRejectionError:
            self.record("Section 3", "Mismatched Checkpoint for adapter_t2 Rejection", True, "Successfully rejected adapter_t2 checkpoint mismatch.", "HIGH")

        # 3.7 Checkpoint Mismatch for non_recurrent condition
        try:
            validate_can_merge(
                base_summary,
                base_prov,
                incoming_condition="non_recurrent",
                incoming_checkpoint_sha256="checkpoint_non_recurrent_DIFFERENT",
            )
            self.record(
                "Section 3",
                "Mismatched Checkpoint for non_recurrent Rejection",
                False,
                "GAP CONFIRMED: Allowed merge of 'non_recurrent' condition with different checkpoint_sha256!",
                severity="HIGH",
            )
        except CrossRunMergeRejectionError:
            self.record(
                "Section 3",
                "Mismatched Checkpoint for non_recurrent Rejection",
                True,
                "Successfully rejected non_recurrent checkpoint mismatch.",
                severity="HIGH",
            )

        # 3.8 Empty Git Commit Permissiveness Check
        empty_commit_prov = ProvenanceMetadata(**{**base_prov.to_dict(), "git_commit_sha": ""})
        try:
            validate_can_merge(base_summary, empty_commit_prov, "repo_decoder")
            self.record(
                "Section 3",
                "Empty Git Commit SHA Permissiveness",
                False,
                "PERMISSIVE: Empty git_commit_sha is silently merged without rejection against committed summary!",
                severity="MEDIUM",
            )
        except CrossRunMergeRejectionError:
            self.record(
                "Section 3",
                "Empty Git Commit SHA Permissiveness",
                True,
                "Rejected merge when git_commit_sha is empty.",
                severity="MEDIUM",
            )

    # ==========================================================================
    # Summary and Verdict
    # ==========================================================================
    def summarize(self) -> str:
        print("\n" + "=" * 80)
        print("MILITARY-GRADE ADVERSARIAL STRESS TEST SUMMARY")
        print("=" * 80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        critical_failures = [r for r in self.results if not r["passed"] and r["severity"] == "CRITICAL"]
        high_failures = [r for r in self.results if not r["passed"] and r["severity"] == "HIGH"]
        med_failures = [r for r in self.results if not r["passed"] and r["severity"] == "MEDIUM"]

        print(f"Total Tests Executed : {total}")
        print(f"Passed               : {passed}")
        print(f"Failed               : {failed}")
        print(f"  - Critical Failures: {len(critical_failures)}")
        print(f"  - High Failures    : {len(high_failures)}")
        print(f"  - Medium Failures  : {len(med_failures)}")
        print("-" * 80)

        if critical_failures or high_failures:
            verdict = "REQUEST_CHANGES"
            print(f"VERDICT: {verdict} 🚨 (Found {len(critical_failures)} Critical and {len(high_failures)} High vulnerabilities)")
        else:
            verdict = "APPROVE"
            print(f"VERDICT: {verdict} ✅")

        return verdict


def main() -> int:
    runner = AdversarialTestRunner()
    runner.test_section_1_oracle_leakage()
    runner.test_section_2_tamper_detection()
    runner.test_section_3_cross_run_anti_merge()
    verdict = runner.summarize()
    return 1 if verdict == "REQUEST_CHANGES" else 0


if __name__ == "__main__":
    sys.exit(main())
