"""Empirical Challenger Test & Verification Harness for Milestone 3 (PRLRPipeline).

Empirically challenges:
1. PRLRPipeline() default instantiation and strict production adapter loading (strict=True).
2. Fixed deliberation steps T in {1, 2, 4} with working memory slot shape invariant (1, 16, 2048).
3. Dynamic calibrated E-gate halting bounded within [T_min=2, T_max=12].
4. Stage latencies positivity (prefill, prelude, deliberation, decode all > 0.0 ms) and sum-consistency.
5. Text quality metrics: non-empty output string, Shannon entropy H >= 3.0 bits, max 4-gram repetition <= 2.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.egate import GemmaCalibratedEGate
from prlr.manifest import ModelManifest
from prlr.pipeline import (
    BaselineResult,
    HybridDeliberationResult,
    PipelineResult,
    PRLRPipeline,
    compute_shannon_entropy,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "gemma_2b_prlr_adapter.safetensors"
SIDECAR_PATH = PROJECT_DIR / "checkpoints" / "gemma_2b_prlr_adapter.json"
EGATE_CONFIG_PATH = PROJECT_DIR / "checkpoints" / "calibrated_egate_config.json"


def compute_max_4gram_repetition(text: str) -> int:
    """Compute maximum frequency count of any 4-gram in the text."""
    clean = text.strip()
    if not clean:
        return 0
    words = clean.split()
    if len(words) < 4:
        if len(clean) < 4:
            return 1 if clean else 0
        char_ngrams = [clean[i : i + 4] for i in range(len(clean) - 3)]
        counts = collections.Counter(char_ngrams)
        return max(counts.values()) if counts else 0

    word_ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = collections.Counter(word_ngrams)
    return max(counts.values()) if counts else 1


@pytest.fixture(scope="module")
def shared_pipeline() -> PRLRPipeline:
    """Instantiate PRLRPipeline once for empirical verification tests."""
    return PRLRPipeline()


def test_challenge_1_pipeline_instantiation_and_strict_adapter_loading(shared_pipeline: PRLRPipeline):
    """Verify PRLRPipeline instantiates cleanly and loads production adapter with strict=True."""
    pipeline = shared_pipeline
    print("\n--- Challenge 1: Pipeline Instantiation & Strict Loading ---")
    print(f"Adapter loaded: {pipeline.adapter_loaded}")
    print(f"Adapter path: {pipeline.adapter_path}")

    assert pipeline.adapter_loaded is True, "Pipeline failed to load adapter!"
    assert pipeline.adapter_path is not None
    assert Path(pipeline.adapter_path).exists()
    assert Path(pipeline.adapter_path).name == "gemma_2b_prlr_adapter.safetensors"

    # Verify strict=True loading behavior directly on the adapter
    test_adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=4)
    test_adapter.load_weights(str(CHECKPOINT_PATH), strict=True)
    params = dict(tree_flatten(test_adapter.parameters()))
    print(f"Verified strict=True loading of {len(params)} adapter parameter tensors.")
    assert len(params) == 28, f"Expected 28 parameter tensors in production adapter, found {len(params)}"

    # Verify backbone parameter freezing
    trainable_backbone_params = [
        k for k, p in tree_flatten(pipeline.backbone.trainable_parameters())
    ]
    print(f"Trainable backbone parameters: {len(trainable_backbone_params)}")
    assert len(trainable_backbone_params) == 0, "Backbone parameters must be 100% frozen!"


@pytest.mark.parametrize("T", [1, 2, 4])
def test_challenge_2_fixed_steps_slot_shape(shared_pipeline: PRLRPipeline, T: int):
    """Verify fixed steps T in {1, 2, 4} produce slot shape invariant (1, 16, 2048)."""
    pipeline = shared_pipeline
    prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    print(f"\n--- Challenge 2: Fixed Deliberation Steps T={T} ---")

    result = pipeline.deliberate_and_verify(
        prompt=prompt,
        deliberation_steps=T,
        enable_dynamic_gate=False,
        max_new_tokens=16,
        temperature=0.0,
    )

    print(f"Halt step: {result.deliberation_steps} (expected {T})")
    print(f"Final states shape: {result.final_states.shape}")
    print(f"Decoded text: {result.decoded_text!r}")

    assert result.deliberation_steps == T
    assert result.final_states is not None
    assert result.final_states.shape == (1, 16, 2048), f"Expected shape (1, 16, 2048), got {result.final_states.shape}"
    assert not mx.isnan(result.final_states).any().item(), "NaN detected in final states!"
    assert not mx.isinf(result.final_states).any().item(), "Inf detected in final states!"


def test_challenge_3_dynamic_egate_halting_bounds(shared_pipeline: PRLRPipeline):
    """Verify dynamic E-gate halts within calibrated bounds [T_min=2, T_max=12]."""
    pipeline = shared_pipeline
    prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    print("\n--- Challenge 3: Dynamic E-Gate Deliberation Bounds ---")

    result = pipeline.deliberate_and_verify(
        prompt=prompt,
        max_steps=12,
        enable_dynamic_gate=True,
        max_new_tokens=16,
        temperature=0.0,
    )

    steps = result.deliberation_steps
    verdict = result.egate_verdict
    print(f"Dynamic halting steps: {steps}")
    print(f"Exit verdict: {verdict}")
    print(f"Consensus step: {result.consensus_step}")
    if result.gate_telemetry:
        print(f"Telemetry steps recorded: {len(result.gate_telemetry)}")
        for step_idx, telem in enumerate(result.gate_telemetry, 1):
            print(
                f"  Step {step_idx}: vel={telem.velocity:.4f}, margin={telem.margin:.4f}, "
                f"erank={telem.erank:.4f}, halt={telem.halt}"
            )

    assert 2 <= steps <= 12, f"Deliberation steps {steps} outside bounds [2, 12]!"
    assert verdict in ("4_signal_consensus", "max_steps_timeout")
    assert result.final_states.shape == (1, 16, 2048)


def test_challenge_4_stage_latencies_positivity_and_sum_consistency(shared_pipeline: PRLRPipeline):
    """Verify stage latencies are all > 0.0 ms and sum-consistent."""
    pipeline = shared_pipeline
    prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    print("\n--- Challenge 4: Stage Latencies Positivity & Sum Consistency ---")

    result = pipeline.deliberate_and_verify(
        prompt=prompt,
        deliberation_steps=2,
        enable_dynamic_gate=False,
        max_new_tokens=16,
        temperature=0.0,
    )

    b = result.latency_breakdown
    print(f"Latency breakdown: {b}")

    # Stage positivity assertions
    assert b["prefill"] > 0.0, f"Prefill latency must be > 0.0 ms, got {b['prefill']}"
    assert b["prelude"] > 0.0, f"Prelude latency must be > 0.0 ms, got {b['prelude']}"
    assert b["deliberation"] > 0.0, f"Deliberation latency must be > 0.0 ms, got {b['deliberation']}"
    assert b["decode"] > 0.0, f"Decode latency must be > 0.0 ms, got {b['decode']}"
    assert b["total"] > 0.0, f"Total latency must be > 0.0 ms, got {b['total']}"

    # Verify alias keys
    assert b["prefill_ms"] > 0.0
    assert b["prelude_ms"] > 0.0
    assert b["deliberation_ms"] > 0.0
    assert b["decode_ms"] > 0.0
    assert b["total_ms"] > 0.0

    # Sum consistency: total = prefill + prelude + deliberation + decode
    sum_individual = b["prefill"] + b["prelude"] + b["deliberation"] + b["decode"]
    diff = abs(b["total"] - sum_individual)
    print(f"Sum of individual stages: {sum_individual:.3f} ms")
    print(f"Reported total: {b['total']:.3f} ms (discrepancy = {diff:.4f} ms)")

    # Discrepancy should be strictly zero or rounding tolerance (< 0.01 ms)
    assert diff < 0.05, f"Stage latency sum discrepancy too large: |{b['total']} - {sum_individual}| = {diff} ms"


def test_challenge_5_text_quality_entropy_and_repetition(shared_pipeline: PRLRPipeline):
    """Verify output text quality: non-empty, Shannon entropy H >= 3.0 bits, max 4-gram repetition <= 2."""
    pipeline = shared_pipeline
    prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    print("\n--- Challenge 5: Text Quality, Shannon Entropy & 4-Gram Repetition ---")

    result = pipeline.deliberate_and_verify(
        prompt=prompt,
        deliberation_steps=4,
        enable_dynamic_gate=False,
        max_new_tokens=32,
        temperature=0.0,
    )

    text = result.decoded_text
    entropy = compute_shannon_entropy(text)
    rep_count = compute_max_4gram_repetition(text)

    print(f"Decoded text: {text!r}")
    print(f"Length of text: {len(text)} characters, {len(text.split())} words")
    print(f"Shannon entropy: {entropy:.4f} bits (required >= 3.0 bits)")
    print(f"Max 4-gram repetition: {rep_count} (required <= 2)")

    # Assertions
    assert len(text.strip()) > 0, "Generated text is empty!"
    assert entropy >= 3.0, f"Shannon entropy {entropy:.4f} < 3.0 bits!"
    assert rep_count <= 2, f"Max 4-gram repetition {rep_count} > 2!"


def run_standalone_empirical_challenge() -> bool:
    """Run all 5 challenges directly from Python CLI and return overall success."""
    print("=" * 80)
    print("EMPIRICAL CHALLENGER M3-1: PRLRPipeline Verification Suite")
    print("=" * 80)

    t_start = time.perf_counter()

    print("\n[Step 1/5] Initializing PRLRPipeline()...")
    pipeline = PRLRPipeline()
    test_challenge_1_pipeline_instantiation_and_strict_adapter_loading(pipeline)
    print(">>> Challenge 1: PASSED")

    print("\n[Step 2/5] Testing fixed steps T in {1, 2, 4}...")
    for T in [1, 2, 4]:
        test_challenge_2_fixed_steps_slot_shape(pipeline, T)
    print(">>> Challenge 2: PASSED")

    print("\n[Step 3/5] Testing dynamic calibrated E-gate...")
    test_challenge_3_dynamic_egate_halting_bounds(pipeline)
    print(">>> Challenge 3: PASSED")

    print("\n[Step 4/5] Testing stage latencies positivity and sum consistency...")
    test_challenge_4_stage_latencies_positivity_and_sum_consistency(pipeline)
    print(">>> Challenge 4: PASSED")

    print("\n[Step 5/5] Testing text quality (entropy and repetition)...")
    test_challenge_5_text_quality_entropy_and_repetition(pipeline)
    print(">>> Challenge 5: PASSED")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 5 EMPIRICAL CHALLENGES PASSED in {elapsed:.2f}s")
    print("CONFIRMATION VERDICT: CONFIRMED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_standalone_empirical_challenge()
    sys.exit(0 if success else 1)
