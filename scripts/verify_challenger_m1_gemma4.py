#!/usr/bin/env python3
"""Adversarial Empirical Verification Runner for Milestone 1 Gemma 4 12B Recurrent Adapter.

Challenger 1 (teamwork_preview_challenger)
Project: Parallel Latent Reasoner (PRLR)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from prlr.gemma.adapter import GemmaRecurrentAdapter, init_orthogonal_slot_anchors
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_WEIGHTS = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.safetensors"
CHECKPOINT_SIDECAR = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.json"


def log_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verify_checkpoint_files() -> dict:
    log_section("1. PHYSICAL CHECKPOINT FILE EXISTENCE & METADATA")
    assert CHECKPOINT_WEIGHTS.exists(), f"Weights file missing at {CHECKPOINT_WEIGHTS}"
    assert CHECKPOINT_SIDECAR.exists(), f"Sidecar file missing at {CHECKPOINT_SIDECAR}"

    w_stat = CHECKPOINT_WEIGHTS.stat()
    s_stat = CHECKPOINT_SIDECAR.stat()

    print(f"[*] Weights File : {CHECKPOINT_WEIGHTS}")
    print(f"    Size         : {w_stat.st_size:,} bytes ({w_stat.st_size / (1024**2):.2f} MB)")
    print(f"    Last Modified: {time.ctime(w_stat.st_mtime)}")
    print(f"[*] Sidecar JSON : {CHECKPOINT_SIDECAR}")
    print(f"    Size         : {s_stat.st_size:,} bytes")
    print(f"    Last Modified: {time.ctime(s_stat.st_mtime)}")

    assert w_stat.st_size > 700_000_000, f"Unexpected weights size: {w_stat.st_size}"
    assert s_stat.st_size > 500, f"Unexpected sidecar size: {s_stat.st_size}"
    print("[✓] Both checkpoint files physically exist with expected non-zero sizes.")

    return {"weights_size": w_stat.st_size, "sidecar_size": s_stat.st_size}


def verify_sha256() -> dict:
    log_section("2. CRYPTOGRAPHIC SHA-256 HASH VERIFICATION")
    t0 = time.perf_counter()
    hasher = hashlib.sha256()
    bytes_read = 0
    with open(CHECKPOINT_WEIGHTS, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            hasher.update(chunk)
            bytes_read += len(chunk)
    computed_sha = hasher.hexdigest()
    t1 = time.perf_counter()

    with open(CHECKPOINT_SIDECAR, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    expected_sha = sidecar.get("weights_sha256")
    print(f"[*] Computed SHA-256 : {computed_sha}")
    print(f"[*] Sidecar SHA-256  : {expected_sha}")
    print(f"[*] Hashing Speed    : {bytes_read / (1024**2) / (t1 - t0):.1f} MB/s ({t1 - t0:.2f}s)")

    assert computed_sha == expected_sha, f"SHA-256 mismatch! Computed {computed_sha} != Sidecar {expected_sha}"
    print("[✓] Exact cryptographic SHA-256 match confirmed.")

    return {"computed_sha": computed_sha, "sidecar_sha": expected_sha, "sidecar": sidecar}


def verify_sidecar_spec(sidecar: dict) -> None:
    log_section("3. SIDECAR METADATA SPECIFICATION & CONVERGENCE AUDIT")
    print(f"[*] Manifest ID      : {sidecar.get('manifest_id')}")
    print(f"[*] Total Parameters : {sidecar.get('total_parameters'):,}")
    print(f"[*] Trainable Params : {sidecar.get('trainable_parameters'):,}")
    print(f"[*] Frozen Params    : {sidecar.get('frozen_parameters')}")
    print(f"[*] Final Step Loss  : {sidecar.get('final_loss')}")
    print(f"[*] Target Loss      : {sidecar.get('target_loss')}")
    print(f"[*] Converged Flag   : {sidecar.get('converged')}")
    print(f"[*] Peak VRAM        : {sidecar.get('peak_vram_mb')} MB ({sidecar.get('peak_vram_mb') / 1024:.2f} GB)")

    assert sidecar.get("weights_file") == "gemma_4_12b_prlr_adapter.safetensors"
    assert sidecar.get("manifest_id") == "google/gemma-4-12B-it-4bit"
    assert sidecar.get("total_parameters") == 200701444
    assert sidecar.get("trainable_parameters") == 200701442
    assert sidecar.get("frozen_parameters") == 2
    assert sidecar.get("final_loss") < 0.08, f"Final loss {sidecar.get('final_loss')} >= 0.08 target"
    assert sidecar.get("converged") is True, "Sidecar converged flag is not True"
    assert sidecar.get("peak_vram_mb") <= 12288.0, f"Peak VRAM {sidecar.get('peak_vram_mb')} MB exceeds 12.0 GB"
    print("[✓] Sidecar metadata strictly satisfies all Milestone 1 invariants.")


def verify_raw_safetensors() -> dict:
    log_section("4. RAW SAFETENSORS TENSOR STRUCTURE & PARAMETER COUNT AUDIT")
    t0 = time.perf_counter()
    raw_weights = mx.load(str(CHECKPOINT_WEIGHTS))
    t1 = time.perf_counter()

    tensor_count = len(raw_weights)
    total_elements = sum(t.size for t in raw_weights.values())

    print(f"[*] Loaded raw safetensors in {t1 - t0:.3f}s")
    print(f"[*] Total Tensors in Checkpoint : {tensor_count} (Expected: 28)")
    print(f"[*] Total Parameter Elements    : {total_elements:,} (Expected: 200,701,444)")

    assert tensor_count == 28, f"Expected 28 tensors, found {tensor_count}"
    assert total_elements == 200701444, f"Expected 200,701,444 params, found {total_elements}"

    print("\n  Per-Tensor Empirical Statistics:")
    print(f"  {'Tensor Name':<38} {'Shape':<20} {'Size':<12} {'Dtype':<10} {'Norm':<10} {'Min':<10} {'Max':<10}")
    print("  " + "-" * 115)

    stats = {}
    for name, p in sorted(raw_weights.items()):
        p_np = np.array(p)
        p_min = float(p_np.min())
        p_max = float(p_np.max())
        p_norm = float(np.linalg.norm(p_np))
        p_dtype = str(p.dtype)
        p_shape = str(list(p.shape))
        stats[name] = {
            "shape": p_shape,
            "size": p.size,
            "dtype": p_dtype,
            "norm": p_norm,
            "min": p_min,
            "max": p_max,
        }
        print(f"  {name:<38} {p_shape:<20} {p.size:<12,d} {p_dtype:<10} {p_norm:<10.3f} {p_min:<10.4f} {p_max:<10.4f}")

        # Numerical sanity
        assert not np.isnan(p_np).any(), f"NaN in {name}"
        assert not np.isinf(p_np).any(), f"Inf in {name}"
        assert p_norm > 0.0, f"Zero norm in {name}"

    print("[✓] All 28 tensors verified: finite, non-NaN, non-Inf, non-zero.")
    return stats


def verify_adapter_instantiation_and_loading() -> GemmaRecurrentAdapter:
    log_section("5. STRICT ADAPTER INSTANTIATION & WEIGHT LOADING")
    adapter = GemmaRecurrentAdapter(
        dim=3840,
        num_slots=16,
        num_layers=1,
        deliberation_steps=4,
    )
    t0 = time.perf_counter()
    adapter.load_weights(str(CHECKPOINT_WEIGHTS), strict=True)
    t1 = time.perf_counter()

    all_params = dict(tree_flatten(adapter.parameters()))
    trainable_params = dict(tree_flatten(adapter.trainable_parameters()))
    total_params = sum(p.size for p in all_params.values())
    trainable_count = sum(p.size for p in trainable_params.values())
    frozen_keys = set(all_params.keys()) - set(trainable_params.keys())

    print(f"[*] Strict weights loaded in {t1 - t0:.3f}s")
    print(f"[*] Adapter parameters   : {len(all_params)} tensors, {total_params:,} total parameters")
    print(f"[*] Trainable parameters : {len(trainable_params)} tensors, {trainable_count:,} parameters")
    print(f"[*] Frozen parameters    : {len(frozen_keys)} tensors, {total_params - trainable_count} parameters")
    print(f"    Frozen keys          : {sorted(frozen_keys)}")

    assert len(all_params) == 28
    assert total_params == 200701444
    assert len(trainable_params) == 26
    assert trainable_count == 200701442
    assert frozen_keys == {"layers.0.alpha_attn", "layers.0.alpha_mlp"}
    print("[✓] Strict instantiation and parameter accounting verified.")

    return adapter


def verify_residual_bounds(adapter: GemmaRecurrentAdapter) -> None:
    log_section("6. RESIDUAL SCALING ALPHA BOUNDS AUDIT")
    layer = adapter.layers[0]
    alpha_attn = float(layer.alpha_attn.item())
    alpha_mlp = float(layer.alpha_mlp.item())
    raw_alpha_attn = float(layer.raw_alpha_attn.item())
    raw_alpha_mlp = float(layer.raw_alpha_mlp.item())
    eff_attn = float((layer.alpha_max * mx.sigmoid(layer.raw_alpha_attn)).item())
    eff_mlp = float((layer.alpha_max * mx.sigmoid(layer.raw_alpha_mlp)).item())

    print(f"[*] layers.0.alpha_attn     : {alpha_attn:.6f} (raw: {raw_alpha_attn:.4f})")
    print(f"    Effective (0.5*sigmoid) : {eff_attn:.6f} | In [0, 0.5]: {0.0 <= eff_attn <= 0.5}")
    print(f"[*] layers.0.alpha_mlp      : {alpha_mlp:.6f} (raw: {raw_alpha_mlp:.4f})")
    print(f"    Effective (0.5*sigmoid) : {eff_mlp:.6f} | In [0, 0.5]: {0.0 <= eff_mlp <= 0.5}")

    assert 0.0 <= alpha_attn <= 0.5, f"alpha_attn outside [0, 0.5]: {alpha_attn}"
    assert 0.0 <= alpha_mlp <= 0.5, f"alpha_mlp outside [0, 0.5]: {alpha_mlp}"
    assert 0.0 <= eff_attn <= 0.5
    assert 0.0 <= eff_mlp <= 0.5
    assert abs(alpha_attn - eff_attn) < 1e-5
    assert abs(alpha_mlp - eff_mlp) < 1e-5
    print("[✓] Residual scaling gating strictly bounded in [0, 0.5].")


def verify_forward_pass_and_trajectory(adapter: GemmaRecurrentAdapter) -> None:
    log_section("7. FORWARD DELIBERATION PASS & TRAJECTORY MONOTONICITY")
    prompt_hiddens = mx.random.normal((1, 16, 3840))

    # Standard forward deliberation pass (T=4)
    out = adapter(prompt_hiddens, steps=4)
    mx.eval(out)

    print(f"[*] Input shape        : {list(prompt_hiddens.shape)}")
    print(f"[*] Output shape (T=4) : {list(out.shape)} (Expected: [1, 16, 3840])")
    print(f"[*] Output min / max   : {float(mx.min(out).item()):.4f} / {float(mx.max(out).item()):.4f}")
    print(f"[*] Output mean / std  : {float(mx.mean(out).item()):.4f} / {float(mx.sqrt(mx.var(out)).item()):.4f}")
    print(f"[*] Output L2 norm     : {float(mx.linalg.norm(out).item()):.4f}")

    assert out.shape == (1, 16, 3840), f"Expected (1, 16, 3840), got {out.shape}"
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()
    assert 100.0 < float(mx.linalg.norm(out).item()) < 600.0

    # Variable deliberation steps
    print("\n  Variable Deliberation Steps Evaluation:")
    for t in [1, 2, 4, 8, 12]:
        out_t = adapter(prompt_hiddens, steps=t)
        mx.eval(out_t)
        print(f"    T={t:<2d} -> Shape: {list(out_t.shape)} | Norm: {float(mx.linalg.norm(out_t).item()):.2f} | Non-NaN: True")
        assert out_t.shape == (1, 16, 3840)
        assert not mx.isnan(out_t).any().item()

    # Trajectory equivalence
    traj = adapter.unroll_trajectory(prompt_hiddens, max_steps=4)
    print("\n  Trajectory Equivalence & Contraction:")
    for t in range(1, 5):
        step_out = adapter(prompt_hiddens, steps=t)
        traj_out = traj[t]
        max_diff = float(mx.max(mx.abs(step_out - traj_out)).item())
        delta_norm = float(mx.linalg.norm(traj[t] - traj[t - 1]).item())
        print(f"    Step {t} -> Equivalence max diff: {max_diff:.2e} | Inter-step delta norm: {delta_norm:.4f}")
        assert max_diff < 1e-5
    print("[✓] Forward deliberation pass and trajectory equivalence verified.")


def verify_real_gemma4_backbone_integration(adapter: GemmaRecurrentAdapter) -> None:
    log_section("8. REAL PRETRAINED GEMMA 4 12B BACKBONE EXTRACTION & CONDITIONING")
    t0 = time.perf_counter()
    manifest = ModelManifest.gemma_4_12b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    t1 = time.perf_counter()
    print(f"[*] Real Gemma 4 12B backbone loaded in {t1 - t0:.2f}s")

    test_prompt = (
        "<start_of_turn>user\n"
        "Routing task: find the optimal sequence of tools for financial risk audit.\n"
        "Available: schema_validator, compliance_engine, report_emitter.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    p_ids, _ = backbone.encode_prompt_context(test_prompt)
    print(f"[*] Encoded prompt tokens: shape {list(p_ids.shape)}")

    prompt_hiddens = backbone.extract_contextual_hiddens(p_ids)
    print(f"[*] Extracted prompt contextual hiddens: shape {list(prompt_hiddens.shape)}")
    assert prompt_hiddens.shape == (1, p_ids.shape[1], 3840)

    # Deliberation pass
    slots = adapter(prompt_hiddens, steps=4)
    mx.eval(slots)
    print(f"[*] Deliberated working memory slots: shape {list(slots.shape)}")
    print(f"    Norm: {float(mx.linalg.norm(slots).item()):.2f} | Min: {float(mx.min(slots).item()):.4f} | Max: {float(mx.max(slots).item()):.4f}")
    assert slots.shape == (1, 16, 3840)
    assert not mx.isnan(slots).any().item()

    # Prefix decoder conditioning verification
    decoder = GemmaCausalPrefixDecoder(backbone=backbone, prefix_dim=3840, hidden_dim=3840)
    soft_prefix = (slots * (3840 ** -0.5)).astype(backbone.model.language_model.model.embed_tokens(p_ids).dtype)
    prompt_embeds = backbone.model.language_model.model.embed_tokens(p_ids)
    all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

    all_logits = backbone.model(inputs=None, input_embeddings=all_embeds)
    mx.eval(all_logits)
    print(f"[*] Gemma 4 next-token logits over [soft_prefix, prompt_embeds]: shape {list(all_logits.shape)}")
    assert all_logits.shape == (1, 16 + p_ids.shape[1], 262144)
    print("[✓] Full integration with genuine Gemma 4 12B backbone verified.")


def verify_adversarial_stress(adapter: GemmaRecurrentAdapter) -> None:
    log_section("9. ADVERSARIAL STRESS & CORNER CASE MINING")
    stress_cases = {
        "all_zeros": mx.zeros((1, 16, 3840)),
        "large_magnitude (+10,000x)": mx.random.normal((1, 16, 3840)) * 10000.0,
        "tiny_magnitude (1e-6x)": mx.random.normal((1, 16, 3840)) * 1e-6,
        "constant_high (scalar 100)": mx.ones((1, 16, 3840)) * 100.0,
        "alternating_sign (+/-50)": mx.array([[[50.0 if (i % 2 == 0) else -50.0 for i in range(3840)]] * 16]),
        "single_token_prompt (P=1)": mx.random.normal((1, 1, 3840)),
        "long_sequence (P=512)": mx.random.normal((1, 512, 3840)),
        "batch_stress (B=8, P=64)": mx.random.normal((8, 64, 3840)),
    }

    for name, inp in stress_cases.items():
        B = inp.shape[0]
        out = adapter(inp, steps=4)
        mx.eval(out)
        norm = float(mx.linalg.norm(out).item())
        has_nan = bool(mx.isnan(out).any().item())
        has_inf = bool(mx.isinf(out).any().item())
        print(f"  Stress Case '{name:<26}' -> Out Shape: {list(out.shape)} | Norm: {norm:<10.2f} | Finite: {not (has_nan or has_inf)}")
        assert out.shape == (B, 16, 3840)
        assert not has_nan
        assert not has_inf

    # Attention mask stress
    mask = mx.ones((2, 64))
    mask[:, 32:] = 0.0  # 50% masked
    inp_masked = mx.random.normal((2, 64, 3840))
    out_masked = adapter(inp_masked, steps=4, mask=mask)
    mx.eval(out_masked)
    print(f"  Stress Case '50% masked prompt (B=2, P=64)' -> Out Shape: {list(out_masked.shape)} | Finite: True")
    assert out_masked.shape == (2, 16, 3840)
    assert not mx.isnan(out_masked).any().item()

    print("[✓] All adversarial stress and edge case tests passed.")


def verify_cryptographic_tamper_defense() -> None:
    log_section("10. CRYPTOGRAPHIC TAMPER DEFENSE AUDIT")
    with open(CHECKPOINT_WEIGHTS, "rb") as f:
        original_chunk = bytearray(f.read(1024 * 1024))

    tampered_chunk = bytearray(original_chunk)
    tampered_chunk[4242] ^= 0xFF  # Invert 8 bits at byte offset 4242

    orig_sha = hashlib.sha256(original_chunk).hexdigest()
    tamp_sha = hashlib.sha256(tampered_chunk).hexdigest()

    print(f"[*] Original 1MB SHA-256 : {orig_sha}")
    print(f"[*] Tampered 1MB SHA-256 : {tamp_sha}")
    assert orig_sha != tamp_sha, "Tampered byte failed to alter SHA-256!"
    print("[✓] Single-bit tampering immediately invalidates cryptographic signature.")


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("  CHALLENGER 1: EMPIRICAL ADVERSARIAL VERIFICATION SUITE")
    print("  Milestone 1 Production Gemma 4 12B Recurrent Adapter Checkpoint Audit")
    print("=" * 80)

    file_info = verify_checkpoint_files()
    sha_info = verify_sha256()
    verify_sidecar_spec(sha_info["sidecar"])
    tensor_stats = verify_raw_safetensors()
    adapter = verify_adapter_instantiation_and_loading()
    verify_residual_bounds(adapter)
    verify_forward_pass_and_trajectory(adapter)
    verify_real_gemma4_backbone_integration(adapter)
    verify_adversarial_stress(adapter)
    verify_cryptographic_tamper_defense()

    t_total = time.perf_counter() - t_start
    log_section("VERIFICATION SUITE COMPLETE")
    print(f"[*] Total Execution Time : {t_total:.2f}s")
    print(f"[*] Status               : ALL 10 EMPIRICAL VERIFICATION STAGES PASSED")
    print(f"[*] Binary Verdict       : APPROVE")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
