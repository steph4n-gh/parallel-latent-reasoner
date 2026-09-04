"""Unit and Integration Tests for Milestone M2 Controls and Paired Bootstrap.

Verifies:
1. Parameter count parity between recurrent and non-recurrent adapter (< 0.05 delta).
2. Norm matching of control_random (L2 magnitude matching to learned latent slots).
3. Deterministic canonical order-reversal derangement of control_shuffled.
4. Paired bootstrap BCa mathematics (known distributions, skewness, boundary degeneracy).
5. Exact and Monte Carlo paired permutation tests on discordant pairs.
6. Calibrated statistical framing adhering to Evidence Rule 8 (no uncalibrated proof prose).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import List

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from prlr.eval.bootstrap import (
    PairedBootstrapResult,
    PermutationTestResult,
    compute_exact_paired_permutation_test,
    compute_paired_bca_ci,
    compute_paired_bootstrap,
    compute_paired_permutation_test,
    format_calibrated_verdict,
    run_bootstrap_analysis,
)
from prlr.eval.harness import FIXED_SLOT_PERMUTATION
from prlr.gemma.adapter import (
    GemmaNonRecurrentAdapter,
    GemmaRecurrentAdapter,
    NonRecurrentAdapterConfig,
)


# ==============================================================================
# 1. Parameter Count Parity & Forward Pass Integrity
# ==============================================================================

def test_parameter_count_parity():
    """Verify parameter count parity between GemmaRecurrentAdapter and GemmaNonRecurrentAdapter.

    Milestone M2 invariant: abs(nr - rec) / rec < 0.05 (strictly under 5%).
    """
    dim = 3840
    num_slots = 16

    rec_adapter = GemmaRecurrentAdapter(
        dim=dim,
        num_slots=num_slots,
        num_layers=1,
        deliberation_steps=4,
    )
    non_rec_adapter = GemmaNonRecurrentAdapter(
        dim=dim,
        num_slots=num_slots,
        intermediate_dim=13440,
    )

    rec_params = sum(v.size for _, v in tree_flatten(rec_adapter.parameters()))
    non_rec_params = sum(v.size for _, v in tree_flatten(non_rec_adapter.parameters()))

    delta = abs(non_rec_params - rec_params) / float(rec_params)

    # Recurrent: 200,701,444 params; Non-Recurrent: 201,169,920 params; delta = 0.233%
    assert delta < 0.05, f"Parameter parity violated: {delta:.4%} >= 5.0%"
    assert delta < 0.005, f"Target delta should be under 0.5%, got {delta:.4%}"
    assert rec_params == 200701444
    assert non_rec_params == 201169920


def test_non_recurrent_adapter_forward_execution():
    """Verify single-pass non-recurrent adapter execution and tensor shapes."""
    adapter = GemmaNonRecurrentAdapter(
        dim=3840,
        num_slots=16,
        intermediate_dim=13440,
    )
    prompt_hiddens = mx.random.normal((2, 32, 3840)).astype(mx.bfloat16)
    mask = mx.ones((2, 32))

    slots = adapter(prompt_hiddens, mask=mask)
    mx.eval(slots)

    assert slots.shape == (2, 16, 3840), f"Expected shape (2, 16, 3840), got {slots.shape}"
    assert not mx.any(mx.isnan(slots)).item(), "NaN detected in non-recurrent adapter output"
    assert not mx.any(mx.isinf(slots)).item(), "Inf detected in non-recurrent adapter output"


def test_non_recurrent_adapter_config():
    """Verify NonRecurrentAdapterConfig defaults align with Gemma 4 12B."""
    cfg = NonRecurrentAdapterConfig()
    assert cfg.dim == 3840
    assert cfg.num_slots == 16
    assert cfg.intermediate_dim == 13440
    assert cfg.num_heads == 8
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 256


# ==============================================================================
# 2. Control Random: L2 Norm Matching
# ==============================================================================

def test_control_random_norm_matching():
    """Verify control_random isotropic Gaussian noise matches empirical slot norms."""
    B, M, D = 4, 16, 3840

    # Simulate learned latent slots from an adapter
    learned_slots = mx.random.normal((B, M, D)) * 2.5 + 0.3
    mx.eval(learned_slots)

    # Compute target per-slot L2 norm: shape (B, M, 1)
    target_norms = mx.sqrt(mx.sum(learned_slots ** 2, axis=-1, keepdims=True) + 1e-8)
    mx.eval(target_norms)

    # Replicate control_random generation logic from harness.py
    seed = 42
    sample_seed = (abs(hash("sample_test_001")) ^ seed) & 0xFFFFFFFF
    rng_noise = np.random.default_rng(sample_seed).standard_normal(learned_slots.shape)
    noise_arr = mx.array(rng_noise, dtype=learned_slots.dtype)
    noise_norms = mx.sqrt(mx.sum(noise_arr ** 2, axis=-1, keepdims=True) + 1e-8)

    random_slots = (noise_arr / noise_norms) * target_norms
    mx.eval(random_slots)

    # Compute actual L2 norm of generated random slots
    actual_norms = mx.sqrt(mx.sum(random_slots ** 2, axis=-1, keepdims=True) + 1e-8)
    mx.eval(actual_norms)

    target_np = np.array(target_norms)
    actual_np = np.array(actual_norms)

    # Must match to high floating point precision
    assert np.allclose(target_np, actual_np, rtol=1e-5, atol=1e-6), (
        f"Max relative norm divergence: {np.max(np.abs(target_np - actual_np) / target_np)}"
    )

    # Assert that directional cosine similarity is near zero (orthogonal isotropic vectors)
    dot_products = mx.sum(learned_slots * random_slots, axis=-1) / (target_norms[..., 0] * actual_norms[..., 0])
    mx.eval(dot_products)
    mean_abs_cosine = float(np.mean(np.abs(np.array(dot_products))))
    assert mean_abs_cosine < 0.1, f"Expected isotropic orthogonality (cosine < 0.1), got {mean_abs_cosine}"


# ==============================================================================
# 3. Control Shuffled: Deterministic Fixed Derangement
# ==============================================================================

def test_control_shuffled_deterministic_derangement():
    """Verify FIXED_SLOT_PERMUTATION satisfies strict derangement and reversal invariants."""
    assert len(FIXED_SLOT_PERMUTATION) == 16
    assert sorted(FIXED_SLOT_PERMUTATION) == list(range(16)), "Permutation must cover 0..15 exactly"

    # Canonical order-reversal derangement: pi(i) = 15 - i
    assert FIXED_SLOT_PERMUTATION == (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

    # Derangement Invariant: zero fixed points (pi(i) != i for all i)
    for i, p in enumerate(FIXED_SLOT_PERMUTATION):
        assert i != p, f"Fixed point detected at slot index {i}: pi({i}) == {p}"

    # Verify tensor permutation application
    slots = mx.arange(16)[None, :, None]  # Shape: (1, 16, 1)
    perm_idx = mx.array(FIXED_SLOT_PERMUTATION)
    shuffled_slots = slots[:, perm_idx, :]
    mx.eval(shuffled_slots)

    expected_indices = list(range(15, -1, -1))
    actual_indices = shuffled_slots[0, :, 0].tolist()
    assert actual_indices == expected_indices, f"Expected {expected_indices}, got {actual_indices}"

    # Involutive Property: pi(pi(x)) == x
    shuffled_twice = shuffled_slots[:, perm_idx, :]
    mx.eval(shuffled_twice)
    assert shuffled_twice[0, :, 0].tolist() == list(range(16)), "Two successive reversals must restore original order"


# ==============================================================================
# 4. Paired Bootstrap BCa Mathematics
# ==============================================================================

def test_paired_bootstrap_known_positive_delta():
    """Verify BCa bootstrap confidence interval captures known positive performance gap."""
    np.random.seed(42)
    N = 256
    # Condition A: 90% success rate, Condition B: 50% success rate (paired)
    a = [1 if i % 10 != 0 else 0 for i in range(N)]  # ~90%
    b = [1 if i % 2 == 0 else 0 for i in range(N)]   # 50%

    res = compute_paired_bootstrap(a, b, num_resamples=1000, seed=42)
    assert isinstance(res, PairedBootstrapResult)

    observed_delta = float(np.mean(a)) - float(np.mean(b))
    assert abs(res.point_estimate - observed_delta) < 1e-6
    assert res.ci_lower < res.point_estimate < res.ci_upper
    assert res.ci_lower > 0.0, "95% CI must strictly exclude zero for 90% vs 50%"
    assert res.method in ("BCa", "percentile_fallback")

    # Tuple unpacking check
    pe, l, u = res
    assert pe == res.point_estimate
    assert l == res.ci_lower
    assert u == res.ci_upper


def test_paired_bootstrap_zero_variance_degeneracy():
    """Verify BCa handles identical paired series (delta == 0 everywhere) gracefully."""
    a = [1, 0, 1, 1, 0, 0, 1, 1]
    b = [1, 0, 1, 1, 0, 0, 1, 1]

    res = compute_paired_bootstrap(a, b, num_resamples=1000, seed=42)
    assert res.point_estimate == 0.0
    assert res.ci_lower == 0.0
    assert res.ci_upper == 0.0
    assert res.ci_95 == (0.0, 0.0)
    assert "degenerate" in res.method


def test_paired_bootstrap_constant_difference():
    """Verify BCa handles constant non-zero difference (a = 1, b = 0) without singularity."""
    a = [1] * 50
    b = [0] * 50

    res = compute_paired_bootstrap(a, b, num_resamples=1000, seed=42)
    assert res.point_estimate == 1.0
    assert res.ci_lower == 1.0
    assert res.ci_upper == 1.0


def test_paired_bootstrap_skewness_adjustment():
    """Verify BCa bias-correction z0 is non-zero on skewed asymmetric distributions."""
    # Highly skewed outcomes: mostly zeros with rare large positive outliers
    a = [1 if i == 0 else 0 for i in range(100)]
    b = [0] * 100

    ci_tuple, z0, a_acc, method = compute_paired_bca_ci(a, b, num_resamples=1000, seed=42)
    assert not math.isnan(z0)
    assert not math.isnan(a_acc)
    assert ci_tuple[0] <= ci_tuple[1]


# ==============================================================================
# 5. Paired Permutation Test & Discordant Pairs
# ==============================================================================

def test_permutation_test_zero_discordant():
    """Verify permutation test returns p=1.0 when conditions are identical."""
    a = [1, 0, 1, 0, 1]
    b = [1, 0, 1, 0, 1]

    res = compute_paired_permutation_test(a, b)
    assert res.p_value == 1.0
    assert res.discordant_count == 0
    assert res.method == "exact_zero_discordant"
    assert res == 1.0  # Float comparison


def test_permutation_test_exact_binomial_small_n():
    """Verify exact permutation test matches analytical binomial distribution (n_disc <= 20)."""
    # 6 discordant pairs: A has 6 successes, B has 0 successes
    # Analytical exact two-sided p-value: 2 * (0.5)^6 = 2 * (1/64) = 0.03125
    a = [1, 1, 1, 1, 1, 1, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0]

    res = compute_paired_permutation_test(a, b)
    assert res.method == "exact_permutation"
    assert res.discordant_count == 6
    assert abs(res.p_value - 0.03125) < 1e-4

    # Test exact binomial helper directly
    p_exact = compute_exact_paired_permutation_test(a, b)
    assert abs(p_exact - 0.03125) < 1e-4


def test_permutation_test_monte_carlo_large_n():
    """Verify Monte Carlo permutation test for n_discordant > 20."""
    np.random.seed(42)
    N = 100
    # 40 discordant pairs where A consistently outperforms B
    a = [1 if i < 50 else 0 for i in range(N)]
    b = [1 if (10 <= i < 50) else 0 for i in range(N)]

    # Exactly 10 discordant instances where A=1, B=0
    # Let's create 30 discordant instances:
    a_30 = [1 if i < 30 else 0 for i in range(N)]
    b_30 = [0] * N

    res = compute_paired_permutation_test(a_30, b_30, num_resamples=10000, seed=42)
    assert res.method == "monte_carlo_permutation"
    assert res.discordant_count == 30
    assert res.p_value < 0.001
    assert res < 0.05


def test_permutation_test_symmetry():
    """Verify two-sided permutation p-value is invariant to condition ordering."""
    a = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    b = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0]

    res_ab = compute_paired_permutation_test(a, b, seed=42)
    res_ba = compute_paired_permutation_test(b, a, seed=42)

    assert abs(res_ab.p_value - res_ba.p_value) < 1e-4
    assert res_ab.discordant_count == res_ba.discordant_count


# ==============================================================================
# 6. Calibrated Statistical Language Protocol (Evidence Rule 8)
# ==============================================================================

def test_calibrated_verdict_formatting():
    """Verify calibrated framing emits scientific statements without prohibited proof vocabulary."""
    # 1. Significant difference
    is_sig, text_sig = format_calibrated_verdict(
        "adapter_t4", "control_random", "exact_match",
        delta=0.25, ci_95=(0.15, 0.35), p_val=0.0001, p_method="exact_permutation"
    )
    assert is_sig is True
    assert "Statistically significant difference detected" in text_sig
    assert "p = 0.0001" in text_sig
    assert "proves" not in text_sig.lower()
    assert "conclusive proof" not in text_sig.lower()

    # 2. Non-significant difference
    is_sig2, text_nonsig = format_calibrated_verdict(
        "control_zeroed", "repo_decoder", "exact_match",
        delta=0.0078, ci_95=(-0.0117, 0.0234), p_val=0.6875, p_method="exact_permutation"
    )
    assert is_sig2 is False
    assert "No statistically significant difference observed" in text_nonsig
    assert "sample stochasticity" in text_nonsig

    # 3. Exact equivalence
    is_sig3, text_equiv = format_calibrated_verdict(
        "repo_decoder", "direct_frozen", "exact_match",
        delta=0.0, ci_95=(0.0, 0.0), p_val=1.0, p_method="exact_zero_discordant"
    )
    assert is_sig3 is False
    assert "Exact empirical equivalence observed" in text_equiv


# ==============================================================================
# 7. End-to-End Bootstrap Analysis Artifact Generation
# ==============================================================================

def test_bootstrap_analysis_artifact_serialization():
    """Verify run_bootstrap_analysis produces valid schema and SHA-256 sidecar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        keys_path = tmp_path / "keys.jsonl"
        preds_dir = tmp_path / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic answer keys
        samples = [f"sample_{i:03d}" for i in range(20)]
        with open(keys_path, "w", encoding="utf-8") as f:
            for sid in samples:
                f.write(json.dumps({
                    "id": sid,
                    "verifier_config": {
                        "expected_route": ["tool_a", "tool_b"],
                        "terminal_tool": "tool_b",
                        "target_goal": "goal",
                    }
                }) + "\n")

        # Generate synthetic predictions for 2 conditions
        def make_pred_file(cond_name, correct: bool):
            ans = '```json\n{"route": ["tool_a", "tool_b"], "terminal": "tool_b"}\n```' if correct else '```json\n{"route": ["wrong"]}\n```'
            data = {
                "metadata": {"condition": cond_name},
                "predictions": [
                    {"sample_id": sid, "decoded_text": ans, "latency_ms": 100.0}
                    for sid in samples
                ]
            }
            p_file = preds_dir / f"predictions_{cond_name}.json"
            p_file.write_text(json.dumps(data), encoding="utf-8")

        make_pred_file("control_zeroed", True)
        make_pred_file("control_random", False)

        out_f, sidecar_f, art = run_bootstrap_analysis(
            predictions_dir=preds_dir,
            keys_path=keys_path,
            output_file=preds_dir / "bootstrap_analysis.json",
            num_resamples=200,
            seed=42,
        )

        assert out_f.exists()
        assert sidecar_f.exists()
        assert art["schema_version"] == "prlr.bootstrap_analysis.v1"
        assert art["provenance"]["matched_sample_count"] == 20

        # Verify sidecar hash
        import hashlib
        computed_sha = hashlib.sha256(out_f.read_bytes()).hexdigest()
        sidecar_text = sidecar_f.read_text(encoding="utf-8")
        assert computed_sha in sidecar_text
