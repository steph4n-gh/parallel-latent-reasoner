"""Empirical Challenger Test Suite for Milestone 1 Production Gemma 2B Recurrent Adapter.

Empirically verifies:
1. Production checkpoint file integrity and SHA-256 cryptographic match with sidecar JSON.
2. Strict weight loading (`adapter.load_weights(filepath, strict=True)`) into `GemmaRecurrentAdapter`.
3. Parameter statistics: non-zero, non-NaN, non-Inf, non-degenerate variance for all weight matrices.
4. Bounded alpha gating parameters (`layers.0.alpha_attn` and `layers.0.alpha_mlp` in [0, 0.5]).
5. Numerical stability across variable deliberation steps (T=1, 2, 4, 8, 12) with dummy prompt inputs.
6. Numerical stability across variable deliberation steps (T=1, 2, 4, 8, 12) with genuine pretrained Gemma contextual hidden states.
7. Trajectory equivalence: `adapter(hiddens, steps=T) == traj[T]` for all T.
8. Adversarial stress testing (extreme input scales, single-token, long-prompt, constant inputs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.manifest import ModelManifest

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_WEIGHTS = CHECKPOINT_DIR / "gemma_2b_prlr_adapter.safetensors"
CHECKPOINT_SIDECAR = CHECKPOINT_DIR / "gemma_2b_prlr_adapter.json"


@pytest.fixture(scope="module")
def loaded_adapter() -> GemmaRecurrentAdapter:
    """Fixture providing a GemmaRecurrentAdapter with strictly loaded production weights."""
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=4)
    adapter.load_weights(str(CHECKPOINT_WEIGHTS), strict=True)
    return adapter


@pytest.fixture(scope="module")
def pretrained_backbone() -> PretrainedGemmaBackbone:
    """Fixture providing real PretrainedGemmaBackbone."""
    manifest = ModelManifest.gemma_2b_it()
    return PretrainedGemmaBackbone(manifest=manifest, load_weights=True)


def test_checkpoint_file_and_sidecar_integrity():
    """Verify checkpoint exists on disk and cryptographic SHA-256 matches sidecar JSON."""
    assert CHECKPOINT_WEIGHTS.exists(), f"Missing weights: {CHECKPOINT_WEIGHTS}"
    assert CHECKPOINT_SIDECAR.exists(), f"Missing sidecar: {CHECKPOINT_SIDECAR}"

    with open(CHECKPOINT_WEIGHTS, "rb") as f:
        file_bytes = f.read()
        computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

    with open(CHECKPOINT_SIDECAR, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert sidecar["weights_sha256"] == computed_sha256, (
        f"SHA-256 mismatch: sidecar has {sidecar['weights_sha256']}, computed {computed_sha256}"
    )
    assert sidecar["weights_file"] == "gemma_2b_prlr_adapter.safetensors"
    assert sidecar["backbone_model_id"] == "google/gemma-2b-it"
    assert sidecar["converged"] is True, "Sidecar indicates training did not converge"
    assert sidecar["final_loss"] < 0.15, f"Final loss {sidecar['final_loss']} not < 0.15"
    assert sidecar["total_parameters"] == 88690692, (
        f"Parameter count mismatch: {sidecar['total_parameters']} != 88690692"
    )


def test_strict_weight_loading(loaded_adapter: GemmaRecurrentAdapter):
    """Verify strict loading restores all 28 parameter arrays without missing or extra keys."""
    params = dict(tree_flatten(loaded_adapter.parameters()))
    assert len(params) == 28, f"Expected 28 parameter tensors, found {len(params)}"

    expected_keys = [
        "prelude.slot_anchors",
        "prelude.slot_role_embed",
        "prelude.context_proj.weight",
        "prelude.norm.weight",
        "layers.0.norm1.weight",
        "layers.0.norm1.mlp_l1.weight",
        "layers.0.norm1.mlp_l1.bias",
        "layers.0.norm1.mlp_l2.weight",
        "layers.0.norm1.mlp_l2.bias",
        "layers.0.attn.q_proj.weight",
        "layers.0.attn.k_proj.weight",
        "layers.0.attn.v_proj.weight",
        "layers.0.attn.o_proj.weight",
        "layers.0.attn.k_cross_proj.weight",
        "layers.0.attn.v_cross_proj.weight",
        "layers.0.alpha_attn",
        "layers.0.raw_alpha_attn",
        "layers.0.norm2.weight",
        "layers.0.norm2.mlp_l1.weight",
        "layers.0.norm2.mlp_l1.bias",
        "layers.0.norm2.mlp_l2.weight",
        "layers.0.norm2.mlp_l2.bias",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.alpha_mlp",
        "layers.0.raw_alpha_mlp",
        "out_norm.weight",
    ]
    for key in expected_keys:
        assert key in params, f"Missing expected key: {key}"


def test_weight_statistics_and_variance(loaded_adapter: GemmaRecurrentAdapter):
    """Verify all weight matrices are non-zero, non-NaN, non-Inf, and have non-degenerate variance."""
    params = dict(tree_flatten(loaded_adapter.parameters()))
    for name, p in params.items():
        assert not mx.isnan(p).any().item(), f"NaN detected in parameter {name}"
        assert not mx.isinf(p).any().item(), f"Inf detected in parameter {name}"
        norm = float(mx.linalg.norm(p.flatten()).item())
        assert norm > 0.0, f"Zero weight array in parameter {name}"

        # Check non-degenerate variance on multidimensional projection matrices
        if p.ndim >= 2:
            std = float(mx.sqrt(mx.var(p)).item())
            assert std > 1e-4, f"Degenerate variance in matrix {name}: std={std}"


def test_alpha_gating_parameters_bounded(loaded_adapter: GemmaRecurrentAdapter):
    """Verify layers.0.alpha_attn and layers.0.alpha_mlp are strictly bounded within [0, 0.5]."""
    alpha_attn = float(loaded_adapter.layers[0].alpha_attn.item())
    alpha_mlp = float(loaded_adapter.layers[0].alpha_mlp.item())

    assert 0.0 <= alpha_attn <= 0.5, f"alpha_attn={alpha_attn} outside [0, 0.5]"
    assert 0.0 <= alpha_mlp <= 0.5, f"alpha_mlp={alpha_mlp} outside [0, 0.5]"

    # Check effective alphas computed from raw_alpha via sigmoid
    eff_attn = float(
        (loaded_adapter.layers[0].alpha_max * mx.sigmoid(loaded_adapter.layers[0].raw_alpha_attn)).item()
    )
    eff_mlp = float(
        (loaded_adapter.layers[0].alpha_max * mx.sigmoid(loaded_adapter.layers[0].raw_alpha_mlp)).item()
    )
    assert 0.0 <= eff_attn <= 0.5, f"effective_alpha_attn={eff_attn} outside [0, 0.5]"
    assert 0.0 <= eff_mlp <= 0.5, f"effective_alpha_mlp={eff_mlp} outside [0, 0.5]"
    assert abs(alpha_attn - eff_attn) < 1e-5, f"alpha_attn ({alpha_attn}) != effective ({eff_attn})"
    assert abs(alpha_mlp - eff_mlp) < 1e-5, f"alpha_mlp ({alpha_mlp}) != effective ({eff_mlp})"


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 12])
def test_variable_deliberation_steps_dummy_input(
    loaded_adapter: GemmaRecurrentAdapter,
    steps: int,
):
    """Verify forward pass across T in {1, 2, 4, 8, 12} with dummy inputs remains bounded."""
    dummy_input = mx.random.normal((2, 16, 2048))
    out = loaded_adapter(dummy_input, steps=steps)
    mx.eval(out)

    assert out.shape == (2, 16, 2048)
    assert not mx.isnan(out).any().item(), f"NaN in output at steps={steps}"
    assert not mx.isinf(out).any().item(), f"Inf in output at steps={steps}"

    # Verify representations remain bounded in standard range
    min_val = float(mx.min(out).item())
    max_val = float(mx.max(out).item())
    norm_val = float(mx.linalg.norm(out).item())

    assert -10.0 < min_val < 0.0, f"Unbounded min value: {min_val}"
    assert 0.0 < max_val < 10.0, f"Unbounded max value: {max_val}"
    assert 100.0 < norm_val < 500.0, f"Unbounded norm value: {norm_val}"


def test_trajectory_equivalence_and_velocity_decay(loaded_adapter: GemmaRecurrentAdapter):
    """Verify adapter(hiddens, steps=T) == traj[T] and velocity decays monotonically over unroll."""
    dummy_input = mx.random.normal((1, 24, 2048))
    traj = loaded_adapter.unroll_trajectory(dummy_input, max_steps=12)

    velocities = []
    for t in range(1, 13):
        direct_out = loaded_adapter(dummy_input, steps=t)
        traj_out = traj[t]
        max_diff = float(mx.max(mx.abs(direct_out - traj_out)).item())
        assert max_diff < 1e-6, f"Trajectory discrepancy at t={t}: max_diff={max_diff}"

        st = traj[t]
        st_prev = traj[t - 1]
        delta_norm = float(mx.linalg.norm(st - st_prev).item())
        prev_norm = float(mx.linalg.norm(st_prev).item())
        rel_vel = delta_norm / max(prev_norm, 1e-8)
        velocities.append(rel_vel)

    # Monotonic velocity decay check from step 2 onwards (step 1 is prelude->recurrent jump)
    assert velocities[1] < velocities[0], "Velocity did not decay from step 1 to step 2"
    assert velocities[3] < velocities[1], "Velocity did not decay from step 2 to step 4"
    assert velocities[7] < velocities[3], "Velocity did not decay from step 4 to step 8"
    assert velocities[11] < velocities[7], "Velocity did not decay from step 8 to step 12"
    assert velocities[-1] < 1e-3, f"Velocity at step 12 ({velocities[-1]}) did not converge"


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 12])
def test_variable_deliberation_steps_real_prompt(
    loaded_adapter: GemmaRecurrentAdapter,
    pretrained_backbone: PretrainedGemmaBackbone,
    steps: int,
):
    """Verify forward pass across T in {1, 2, 4, 8, 12} with real contextual hidden states."""
    prompt = (
        "<start_of_turn>user\n"
        "Given the tool registry, determine the minimal valid sequence: "
        "Available Tools: schema_parser, auth_validator. Target: user_session.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    p_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    prompt_hiddens = pretrained_backbone.extract_contextual_hiddens(p_ids)

    out = loaded_adapter(prompt_hiddens, steps=steps)
    mx.eval(out)

    assert out.shape == (1, 16, 2048)
    assert not mx.isnan(out).any().item(), f"NaN in real prompt output at steps={steps}"
    assert not mx.isinf(out).any().item(), f"Inf in real prompt output at steps={steps}"

    min_val = float(mx.min(out).item())
    max_val = float(mx.max(out).item())
    std_val = float(mx.sqrt(mx.var(out)).item())
    norm_val = float(mx.linalg.norm(out).item())

    # Bound assertions on representations
    assert -8.0 < min_val < 0.0, f"Real prompt min value outside bounds: {min_val}"
    assert 0.0 < max_val < 8.0, f"Real prompt max value outside bounds: {max_val}"
    assert 0.9 < std_val < 1.2, f"Real prompt activation std outside normalized range: {std_val}"
    assert 180.0 < norm_val < 210.0, f"Real prompt norm outside expected range: {norm_val}"


def test_adversarial_stress_extreme_inputs(loaded_adapter: GemmaRecurrentAdapter):
    """Stress test adapter under adversarial extreme inputs to verify robustness."""
    test_cases = {
        "large_magnitude": mx.random.normal((1, 16, 2048)) * 1000.0,
        "tiny_magnitude": mx.random.normal((1, 16, 2048)) * 1e-6,
        "single_token": mx.random.normal((1, 1, 2048)),
        "long_sequence": mx.random.normal((1, 512, 2048)),
        "all_constant": mx.ones((1, 16, 2048)) * 50.0,
    }

    for name, inp in test_cases.items():
        for t in [1, 4, 12]:
            out = loaded_adapter(inp, steps=t)
            mx.eval(out)
            assert not mx.isnan(out).any().item(), f"NaN in stress test '{name}' at t={t}"
            assert not mx.isinf(out).any().item(), f"Inf in stress test '{name}' at t={t}"
            norm = float(mx.linalg.norm(out).item())
            assert 100.0 < norm < 300.0, f"Norm out of bounds in stress test '{name}' at t={t}: {norm}"
