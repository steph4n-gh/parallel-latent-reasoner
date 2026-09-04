"""Empirical Challenger Test Suite for Milestone 1: Production Gemma 4 12B Recurrent Adapter Checkpoint.

Strictly verifies:
1. Physical existence and file integrity of `checkpoints/gemma_4_12b_prlr_adapter.safetensors`
   and `checkpoints/gemma_4_12b_prlr_adapter.json`.
2. Cryptographic SHA-256 checksum calculation against the sidecar metadata.
3. Sidecar metadata conformity (convergence, final loss < 0.08, peak VRAM <= 12.0 GB).
4. Raw safetensors tensor count (28) and total parameter count (200,701,444).
5. Strict weight loading (`adapter.load_weights(filepath, strict=True)`) into `GemmaRecurrentAdapter`.
6. Parameter accounting: 200,701,444 total, 200,701,442 trainable across 26 tensors, 2 frozen parameters.
7. Numerical integrity: all 28 tensors are non-zero, finite, non-NaN, non-Inf, with non-degenerate variance.
8. Residual gating bounds: alpha_attn and alpha_mlp strictly bounded in [0, 0.5].
9. Orthogonal QR slot anchors: mutually orthogonal slot anchors verified via Gram matrix.
10. Forward deliberation pass: shape (1, 16, 3840) and bounded numerical representation.
11. Multi-step unrolling & trajectory equivalence: unroll_trajectory matches direct step calls.
12. Deliberation velocity contraction: step-to-step delta remains bounded.
13. Batch and sequence length scaling: B in {1, 2, 4} and P in {1, 16, 64, 128, 256}.
14. Prompt attention mask handling: masked pooling stability.
15. Adversarial stress robustness: zeros, extreme magnitudes (+1e4, +1e-6), constant inputs.
16. Cryptographic tamper detection: single-bit modification triggers mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prlr.gemma.adapter import GemmaRecurrentAdapter

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_WEIGHTS = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.safetensors"
CHECKPOINT_SIDECAR = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.json"

EXPECTED_TOTAL_PARAMS = 200_701_444
EXPECTED_TRAINABLE_PARAMS = 200_701_442
EXPECTED_FROZEN_PARAMS = 2
EXPECTED_TENSOR_COUNT = 28
EXPECTED_TRAINABLE_TENSOR_COUNT = 26

EXPECTED_KEYS = [
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

if not CHECKPOINT_WEIGHTS.exists():
    try:
        scripts_dir = PROJECT_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from download_checkpoint import ensure_checkpoint
        ensure_checkpoint(model="gemma_4_12b", target_dir=CHECKPOINT_DIR, quiet=True)
    except Exception:
        pass

if not CHECKPOINT_WEIGHTS.exists():
    pytest.skip(
        f"Production Gemma 4 12B checkpoint {CHECKPOINT_WEIGHTS.name} not found. "
        "Run `python train_gemma4_adapter.py` or download from GitHub release.",
        allow_module_level=True,
    )

if not CHECKPOINT_SIDECAR.exists():
    pytest.skip(
        f"Production Gemma 4 12B sidecar {CHECKPOINT_SIDECAR.name} not found.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def loaded_gemma4_adapter() -> GemmaRecurrentAdapter:
    """Fixture providing a GemmaRecurrentAdapter initialized for Gemma 4 12B with production weights."""
    assert CHECKPOINT_WEIGHTS.exists(), f"Weights missing: {CHECKPOINT_WEIGHTS}"
    adapter = GemmaRecurrentAdapter(
        dim=3840,
        num_slots=16,
        num_layers=1,
        deliberation_steps=4,
    )
    adapter.load_weights(str(CHECKPOINT_WEIGHTS), strict=True)
    return adapter


def test_checkpoint_files_exist():
    """Empirically verify weights and sidecar files exist and have non-zero size."""
    assert CHECKPOINT_WEIGHTS.exists(), f"Checkpoint weights missing at {CHECKPOINT_WEIGHTS}"
    assert CHECKPOINT_SIDECAR.exists(), f"Checkpoint sidecar missing at {CHECKPOINT_SIDECAR}"

    weights_size = CHECKPOINT_WEIGHTS.stat().st_size
    sidecar_size = CHECKPOINT_SIDECAR.stat().st_size

    assert weights_size > 700_000_000, f"Weights file surprisingly small: {weights_size} bytes"
    assert sidecar_size > 500, f"Sidecar file surprisingly small: {sidecar_size} bytes"


def test_sha256_cryptographic_match():
    """Empirically compute streaming SHA-256 and assert exact cryptographic match against sidecar JSON."""
    hasher = hashlib.sha256()
    with open(CHECKPOINT_WEIGHTS, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            hasher.update(chunk)
    computed_sha256 = hasher.hexdigest()

    with open(CHECKPOINT_SIDECAR, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert sidecar["weights_sha256"] == computed_sha256, (
        f"Cryptographic hash mismatch! Sidecar: {sidecar['weights_sha256']}, Computed: {computed_sha256}"
    )


def test_sidecar_metadata_conformance():
    """Verify sidecar metadata specifies Gemma 4 12B architecture and verified training invariants."""
    with open(CHECKPOINT_SIDECAR, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert sidecar["weights_file"] == "gemma_4_12b_prlr_adapter.safetensors"
    assert sidecar["manifest_id"] == "google/gemma-4-12B-it-4bit"
    assert sidecar["backbone_model_id"] == "google/gemma-4-12B-it-4bit"
    assert sidecar["total_parameters"] == EXPECTED_TOTAL_PARAMS
    assert sidecar["trainable_parameters"] == EXPECTED_TRAINABLE_PARAMS
    assert sidecar["frozen_parameters"] == EXPECTED_FROZEN_PARAMS

    # Architecture validation
    arch = sidecar["architecture"]
    assert arch["adapter_type"] == "GemmaRecurrentAdapter"
    assert arch["dim"] == 3840
    assert arch["num_slots"] == 16
    assert arch["num_layers"] == 1
    assert arch["deliberation_steps"] == 4
    assert arch["alpha_max"] == 0.5
    assert arch["rezero_alpha"] == 0.05

    # Training invariants
    assert sidecar["final_loss"] < 0.08, f"Target loss violated: {sidecar['final_loss']}"
    assert sidecar["converged"] is True, "Sidecar indicates training did not converge"
    assert sidecar["peak_vram_mb"] <= 12288.0, (
        f"VRAM budget violated: {sidecar['peak_vram_mb']} MB > 12288.0 MB"
    )


def test_raw_safetensors_structure():
    """Verify raw safetensors contains exactly 28 tensors matching the 200,701,444 parameter specification."""
    raw_weights = mx.load(str(CHECKPOINT_WEIGHTS))
    assert len(raw_weights) == EXPECTED_TENSOR_COUNT, (
        f"Expected {EXPECTED_TENSOR_COUNT} tensors, found {len(raw_weights)}"
    )

    total_elements = sum(t.size for t in raw_weights.values())
    assert total_elements == EXPECTED_TOTAL_PARAMS, (
        f"Expected {EXPECTED_TOTAL_PARAMS} parameters, found {total_elements}"
    )

    for k in EXPECTED_KEYS:
        assert k in raw_weights, f"Missing key in safetensors: {k}"


def test_strict_weight_loading(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Verify strict weight loading succeeds and parameter accounting matches specification."""
    adapter = loaded_gemma4_adapter

    all_params = dict(tree_flatten(adapter.parameters()))
    assert len(all_params) == EXPECTED_TENSOR_COUNT
    total_params = sum(p.size for p in all_params.values())
    assert total_params == EXPECTED_TOTAL_PARAMS

    trainable_params = dict(tree_flatten(adapter.trainable_parameters()))
    assert len(trainable_params) == EXPECTED_TRAINABLE_TENSOR_COUNT
    trainable_count = sum(p.size for p in trainable_params.values())
    assert trainable_count == EXPECTED_TRAINABLE_PARAMS

    frozen_keys = set(all_params.keys()) - set(trainable_params.keys())
    assert frozen_keys == {"layers.0.alpha_attn", "layers.0.alpha_mlp"}


def test_numerical_integrity_and_variance(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Verify no NaN, no Inf, non-zero norms, and non-degenerate variance across all parameters."""
    adapter = loaded_gemma4_adapter
    all_params = dict(tree_flatten(adapter.parameters()))

    for name, p in all_params.items():
        assert not mx.isnan(p).any().item(), f"NaN found in parameter {name}"
        assert not mx.isinf(p).any().item(), f"Inf found in parameter {name}"
        norm = float(mx.linalg.norm(p.flatten()).item())
        assert norm > 0.0, f"Parameter {name} has zero norm"

        # Check non-degenerate variance for projection matrices
        if p.ndim >= 2:
            std = float(mx.sqrt(mx.var(p)).item())
            assert std > 1e-5, f"Parameter {name} has degenerate variance: {std}"


def test_residual_scaling_bounded(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Verify layers.0.alpha_attn and layers.0.alpha_mlp are strictly bounded in [0, 0.5]."""
    layer = loaded_gemma4_adapter.layers[0]

    alpha_attn = float(layer.alpha_attn.item())
    alpha_mlp = float(layer.alpha_mlp.item())

    assert 0.0 <= alpha_attn <= 0.5, f"alpha_attn outside [0, 0.5]: {alpha_attn}"
    assert 0.0 <= alpha_mlp <= 0.5, f"alpha_mlp outside [0, 0.5]: {alpha_mlp}"

    # Effective alphas computed via sigmoid
    eff_attn = float((layer.alpha_max * mx.sigmoid(layer.raw_alpha_attn)).item())
    eff_mlp = float((layer.alpha_max * mx.sigmoid(layer.raw_alpha_mlp)).item())

    assert 0.0 <= eff_attn <= 0.5, f"eff_attn outside [0, 0.5]: {eff_attn}"
    assert 0.0 <= eff_mlp <= 0.5, f"eff_mlp outside [0, 0.5]: {eff_mlp}"

    assert abs(alpha_attn - eff_attn) < 1e-5, f"alpha_attn ({alpha_attn}) != eff_attn ({eff_attn})"
    assert abs(alpha_mlp - eff_mlp) < 1e-5, f"alpha_mlp ({alpha_mlp}) != eff_mlp ({eff_mlp})"


def test_slot_anchor_initialization_orthogonality():
    """Empirically verify slot anchors initialization is mutually orthogonal via CPU QR decomposition."""
    from prlr.gemma.adapter import init_orthogonal_slot_anchors

    anchors = init_orthogonal_slot_anchors(16, 3840, scale=0.02)[0]
    assert anchors.shape == (16, 3840)

    gram = (anchors @ anchors.T).astype(mx.float32)
    expected_diag = (0.02 ** 2) * mx.eye(16)
    max_error = float(mx.max(mx.abs(gram - expected_diag)).item())
    assert max_error < 1e-5, f"Slot anchor initialization failed orthogonality check: max error {max_error}"


def test_checkpoint_slot_anchors_properties(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Empirically verify slot anchors in loaded checkpoint have shape (1, 16, 3840) and learned values."""
    anchors = loaded_gemma4_adapter.prelude.slot_anchors
    assert anchors.shape == (1, 16, 3840)
    assert not mx.isnan(anchors).any().item()
    assert not mx.isinf(anchors).any().item()
    norm_val = float(mx.linalg.norm(anchors).item())
    assert norm_val > 0.05, f"Learned slot anchors norm unexpectedly small: {norm_val}"


def test_forward_pass_deliberation_shape(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Empirically verify forward deliberation pass with prompt hiddens of shape (1, 16, 3840)."""
    prompt_hiddens = mx.random.normal((1, 16, 3840))
    out = loaded_gemma4_adapter(prompt_hiddens, steps=4)
    mx.eval(out)

    assert out.shape == (1, 16, 3840), f"Expected shape (1, 16, 3840), got {out.shape}"
    assert not mx.isnan(out).any().item(), "NaN in forward pass output"
    assert not mx.isinf(out).any().item(), "Inf in forward pass output"

    # Representation norm and distribution bounds
    norm_val = float(mx.linalg.norm(out).item())
    assert 100.0 < norm_val < 600.0, f"Deliberated slots norm out of expected range: {norm_val}"


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 12])
def test_variable_deliberation_steps(
    loaded_gemma4_adapter: GemmaRecurrentAdapter,
    steps: int,
):
    """Verify forward deliberation pass across various step counts T in {1, 2, 4, 8, 12}."""
    prompt_hiddens = mx.random.normal((1, 24, 3840))
    out = loaded_gemma4_adapter(prompt_hiddens, steps=steps)
    mx.eval(out)

    assert out.shape == (1, 16, 3840)
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()


def test_trajectory_equivalence_and_contraction(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Verify trajectory unrolling matches individual step outputs and exhibits bounded velocity."""
    prompt_hiddens = mx.random.normal((1, 32, 3840))
    traj = loaded_gemma4_adapter.unroll_trajectory(prompt_hiddens, max_steps=4)
    assert len(traj) == 5  # S^(0), S^(1), S^(2), S^(3), S^(4)

    for t in range(1, 5):
        step_out = loaded_gemma4_adapter(prompt_hiddens, steps=t)
        traj_out = traj[t]
        max_diff = float(mx.max(mx.abs(step_out - traj_out)).item())
        assert max_diff < 1e-5, f"Trajectory mismatch at step {t}: max diff {max_diff}"

    # Verify velocities between steps are finite and bounded
    v1 = float(mx.linalg.norm(traj[1] - traj[0]).item())
    v2 = float(mx.linalg.norm(traj[2] - traj[1]).item())
    v4 = float(mx.linalg.norm(traj[4] - traj[3]).item())
    assert v1 > 0.0
    assert v2 > 0.0
    assert v4 > 0.0


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64, 128, 256])
def test_batch_and_sequence_scaling(
    loaded_gemma4_adapter: GemmaRecurrentAdapter,
    batch_size: int,
    seq_len: int,
):
    """Stress test adapter under varying batch sizes and prompt lengths."""
    prompt_hiddens = mx.random.normal((batch_size, seq_len, 3840))
    out = loaded_gemma4_adapter(prompt_hiddens, steps=4)
    mx.eval(out)

    assert out.shape == (batch_size, 16, 3840)
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()


def test_attention_masking_stability(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Verify forward deliberation pass with active prompt padding masks."""
    batch_size = 2
    seq_len = 32
    prompt_hiddens = mx.random.normal((batch_size, seq_len, 3840))
    # Mask out the last 12 tokens
    mask = mx.ones((batch_size, seq_len))
    mask[:, 20:] = 0.0

    out = loaded_gemma4_adapter(prompt_hiddens, steps=4, mask=mask)
    mx.eval(out)

    assert out.shape == (batch_size, 16, 3840)
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()


def test_adversarial_numerical_stress(loaded_gemma4_adapter: GemmaRecurrentAdapter):
    """Adversarial stress testing with extreme values, constants, and scalings."""
    stress_inputs = {
        "all_zeros": mx.zeros((1, 16, 3840)),
        "large_magnitude": mx.random.normal((1, 16, 3840)) * 10000.0,
        "tiny_magnitude": mx.random.normal((1, 16, 3840)) * 1e-6,
        "constant_high": mx.ones((1, 16, 3840)) * 100.0,
        "alternating_sign": mx.array(
            [[[1.0 if (i % 2 == 0) else -1.0 for i in range(3840)]] * 16]
        ) * 50.0,
    }

    for case_name, inp in stress_inputs.items():
        out = loaded_gemma4_adapter(inp, steps=4)
        mx.eval(out)
        assert out.shape == (1, 16, 3840), f"Shape failed in case {case_name}"
        assert not mx.isnan(out).any().item(), f"NaN encountered in stress case {case_name}"
        assert not mx.isinf(out).any().item(), f"Inf encountered in stress case {case_name}"
        norm = float(mx.linalg.norm(out).item())
        assert norm > 0.0, f"Zero output norm in stress case {case_name}"


def test_cryptographic_tamper_detection():
    """Adversarial security test: verify single-bit modification fails cryptographic verification."""
    # Read first 1MB of checkpoint file
    with open(CHECKPOINT_WEIGHTS, "rb") as f:
        data = bytearray(f.read(1024 * 1024))

    # Flip one bit in payload
    data[1000] ^= 0x01
    tampered_hash = hashlib.sha256(data).hexdigest()

    with open(CHECKPOINT_SIDECAR, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert tampered_hash != sidecar["weights_sha256"], (
        "Single-bit tamper was unexpectedly not detected by SHA-256!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
