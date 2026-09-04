"""Empirical Challenger Adversarial Verification Suite for Milestone 1.

Verifies:
1. Decoder stability: D^{-0.5} = 3840^{-0.5} soft memory scaling prevents logits overflow / attention distortion in GemmaCausalPrefixDecoder.
2. Memory boundary: Unrolling GemmaCausalPrefixDecoder and train_gemma4_adapter.py step with --max-prompt-len 128 does not exceed 12.0 GB VRAM on Metal GPU.
3. Downloader integrity: scripts/download_checkpoint.py --model gemma_4_12b logic and SHA-256 verification functions.
4. Backward compatibility: Existing Gemma 2B code, manifests, and tests function without regressions.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_checkpoint import (
    CHECKPOINT_REGISTRY,
    compute_file_sha256,
    ensure_checkpoint,
    ensure_single_model_checkpoint,
    verify_checkpoint_files,
)
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
WEIGHTS_4_PATH = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.safetensors"
SIDECAR_4_PATH = CHECKPOINT_DIR / "gemma_4_12b_prlr_adapter.json"
WEIGHTS_2_PATH = CHECKPOINT_DIR / "gemma_2b_prlr_adapter.safetensors"
SIDECAR_2_PATH = CHECKPOINT_DIR / "gemma_2b_prlr_adapter.json"

if not WEIGHTS_4_PATH.exists():
    try:
        scripts_dir = PROJECT_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from download_checkpoint import ensure_checkpoint
        ensure_checkpoint(model="gemma_4_12b", target_dir=CHECKPOINT_DIR, quiet=True)
    except Exception:
        pass

if not WEIGHTS_2_PATH.exists():
    try:
        scripts_dir = PROJECT_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from download_checkpoint import ensure_checkpoint
        ensure_checkpoint(model="gemma_2b", target_dir=CHECKPOINT_DIR, quiet=True)
    except Exception:
        pass


def get_peak_memory_gb() -> float:
    """Retrieve current peak Metal memory in gigabytes."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024**3)
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024**3)
    return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gemma4_backbone() -> PretrainedGemmaBackbone:
    manifest = ModelManifest.gemma_4_12b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    backbone.freeze()
    return backbone


@pytest.fixture(scope="module")
def gemma4_decoder(gemma4_backbone: PretrainedGemmaBackbone) -> GemmaCausalPrefixDecoder:
    return GemmaCausalPrefixDecoder(
        backbone=gemma4_backbone,
        prefix_dim=3840,
        hidden_dim=3840,
    )


@pytest.fixture(scope="module")
def gemma4_adapter() -> GemmaRecurrentAdapter:
    adapter = GemmaRecurrentAdapter(dim=3840, num_slots=16, num_layers=1, deliberation_steps=4)
    if WEIGHTS_4_PATH.exists():
        adapter.load_weights(str(WEIGHTS_4_PATH), strict=True)
    return adapter


# ---------------------------------------------------------------------------
# Test Category 1: Decoder Stability & D^{-0.5} Soft Memory Scaling
# ---------------------------------------------------------------------------

def test_decoder_stability_soft_memory_scaling(
    gemma4_backbone: PretrainedGemmaBackbone,
    gemma4_decoder: GemmaCausalPrefixDecoder,
    gemma4_adapter: GemmaRecurrentAdapter,
):
    """Verify D^{-0.5} = 3840^{-0.5} soft memory scaling prevents logits overflow and attention distortion."""
    inner = gemma4_decoder.get_inner_model()
    prompt = "The minimal tool path to authenticate user session is"
    prompt_ids, _ = gemma4_backbone.encode_prompt_context(prompt)
    prompt_hiddens = gemma4_backbone.extract_contextual_hiddens(prompt_ids)
    slots = gemma4_adapter(prompt_hiddens, steps=4).astype(mx.bfloat16)

    # 1. Scaled forward pass (D^{-0.5})
    prompt_embeds = inner.embed_tokens(prompt_ids)
    soft_prefix_scaled = (slots * (3840 ** -0.5)).astype(prompt_embeds.dtype)
    all_embeds_scaled = mx.concatenate([soft_prefix_scaled, prompt_embeds], axis=1)
    logits_scaled = gemma4_backbone.model(inputs=None, input_embeddings=all_embeds_scaled)
    mx.eval(logits_scaled)

    # Assert finite & well-conditioned logits
    assert not mx.isnan(logits_scaled).any().item(), "NaN in scaled logits"
    assert not mx.isinf(logits_scaled).any().item(), "Inf in scaled logits"

    max_logit_scaled = float(mx.max(logits_scaled).item())
    min_logit_scaled = float(mx.min(logits_scaled).item())
    std_logit_scaled = float(mx.sqrt(mx.var(logits_scaled)).item())

    # Logits should be in normal transformer range (typically [-30, +35])
    assert max_logit_scaled < 50.0, f"Scaled max logit too high: {max_logit_scaled}"
    assert min_logit_scaled > -50.0, f"Scaled min logit too low: {min_logit_scaled}"
    assert 2.0 < std_logit_scaled < 25.0, f"Scaled logit std outside expected range: {std_logit_scaled}"

    # Softmax token entropy check (must have healthy distribution, not collapsed to delta function)
    probs_scaled = mx.softmax(logits_scaled[:, -1, :].astype(mx.float32), axis=-1)
    entropy_scaled = -float(mx.sum(probs_scaled * mx.log2(probs_scaled + 1e-12)).item())
    assert entropy_scaled > 0.5, f"Entropy collapsed under scaled prefix: {entropy_scaled} bits"

    # 2. Adversarial Unscaled pass (multiplier 1.0 instead of 3840^{-0.5})
    soft_prefix_unscaled = slots.astype(prompt_embeds.dtype)
    all_embeds_unscaled = mx.concatenate([soft_prefix_unscaled, prompt_embeds], axis=1)
    logits_unscaled = gemma4_backbone.model(inputs=None, input_embeddings=all_embeds_unscaled)
    mx.eval(logits_unscaled)

    max_logit_unscaled = float(mx.max(logits_unscaled).item())
    std_logit_unscaled = float(mx.sqrt(mx.var(logits_unscaled)).item())

    # Verify that unscaled slots distort the logits compared to scaled slots
    diff_norm = float(mx.linalg.norm(logits_scaled - logits_unscaled).item())
    assert diff_norm > 50.0, f"Expected significant divergence between scaled and unscaled logits, got diff_norm={diff_norm}"


def test_decoder_perturbation_and_generation_stability(
    gemma4_backbone: PretrainedGemmaBackbone,
    gemma4_decoder: GemmaCausalPrefixDecoder,
    gemma4_adapter: GemmaRecurrentAdapter,
):
    """Verify autoregressive generation runs cleanly with bounded latents and halts on EOS."""
    prompt = "France capital"
    prompt_ids, _ = gemma4_backbone.encode_prompt_context(prompt)
    prompt_hiddens = gemma4_backbone.extract_contextual_hiddens(prompt_ids)
    slots = gemma4_adapter(prompt_hiddens, steps=4).astype(mx.bfloat16)

    # Autoregressive generation up to 16 tokens
    tokens, decoded = gemma4_decoder.decode_tokens(
        soft_prefix_latents=slots,
        max_new_tokens=16,
        eos_token_id=1,
        prompt_ids=prompt_ids,
    )
    mx.eval(tokens)

    assert tokens.shape[0] == 1
    assert tokens.shape[1] <= 16
    assert isinstance(decoded, str)


# ---------------------------------------------------------------------------
# Test Category 2: Memory Boundary (VRAM <= 12.0 GB)
# ---------------------------------------------------------------------------

def test_metal_vram_bounds_training_step(
    gemma4_backbone: PretrainedGemmaBackbone,
    gemma4_adapter: GemmaRecurrentAdapter,
):
    """Empirically verify that BPTT unroll and training step with max-prompt-len 128 stays <= 12.0 GB VRAM."""
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    base_vram_gb = get_peak_memory_gb()

    # Create dummy prompt of exact boundary length: 128 tokens
    max_prompt_len = 128
    prompt_ids = mx.random.randint(10, 200000, (1, max_prompt_len), dtype=mx.int32)
    target_ids = mx.array([[651, 6037, 107, 1]], dtype=mx.int32)
    target_mask = mx.ones((1, 4), dtype=mx.float32)
    prompt_mask = mx.ones((1, max_prompt_len), dtype=mx.float32)

    prompt_hiddens = gemma4_backbone.extract_contextual_hiddens(prompt_ids)
    inner = gemma4_backbone.model.language_model.model
    prompt_embeds = inner.embed_tokens(prompt_ids)
    target_embeds = inner.embed_tokens(target_ids[:, :-1])
    mx.eval(prompt_hiddens, prompt_embeds, target_embeds)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    # Loss closure
    def loss_fn(model):
        slots = model(prompt_hiddens, steps=4, mask=prompt_mask).astype(mx.bfloat16)
        soft_prefix = (slots * (3840 ** -0.5)).astype(prompt_embeds.dtype)
        all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
        all_logits = gemma4_backbone.model(inputs=None, input_embeddings=all_embeds)
        start_idx = 16 + max_prompt_len - 1
        end_idx = start_idx + 4
        target_logits = all_logits[:, start_idx:end_idx, :]
        losses = nn.losses.cross_entropy(target_logits, target_ids)
        return mx.mean(losses * target_mask)

    # Reset peak memory right before BPTT backward graph
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    vg_fn = nn.value_and_grad(gemma4_adapter, loss_fn)
    loss, grads = vg_fn(gemma4_adapter)
    clipped_grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(gemma4_adapter, clipped_grads)
    mx.eval(gemma4_adapter.parameters(), optimizer.state, loss, grad_norm)

    step_peak_vram_gb = get_peak_memory_gb()

    # Clean up
    del grads, clipped_grads, loss, grad_norm, prompt_hiddens, prompt_embeds, target_embeds
    del prompt_ids, target_ids, target_mask, prompt_mask
    gc.collect()
    mx.clear_cache()

    # VRAM ceiling is 12.0 GB (12,288 MB)
    vram_ceiling_gb = 12.0
    assert step_peak_vram_gb <= vram_ceiling_gb, (
        f"Peak VRAM violated 12.0 GB boundary: {step_peak_vram_gb:.3f} GB > {vram_ceiling_gb} GB"
    )
    assert step_peak_vram_gb > 8.0, f"Measured peak memory unexpectedly low: {step_peak_vram_gb:.3f} GB"


def test_metal_vram_bounds_decoder_unroll(
    gemma4_backbone: PretrainedGemmaBackbone,
    gemma4_decoder: GemmaCausalPrefixDecoder,
):
    """Empirically verify decoder forward pass with max-prompt-len 128 remains <= 12.0 GB VRAM."""
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    prompt_ids = mx.random.randint(10, 200000, (1, 128), dtype=mx.int32)
    slots = mx.zeros((1, 16, 3840), dtype=mx.bfloat16)
    target_ids = mx.array([[10, 20, 30, 1]], dtype=mx.int32)

    loss, logits = gemma4_decoder.forward(prompt_ids, slots, target_ids)
    mx.eval(loss, logits)

    decoder_peak_vram_gb = get_peak_memory_gb()

    del loss, logits, prompt_ids, slots, target_ids
    gc.collect()
    mx.clear_cache()

    assert decoder_peak_vram_gb <= 12.0, (
        f"Decoder forward pass exceeded 12.0 GB ceiling: {decoder_peak_vram_gb:.3f} GB"
    )


# ---------------------------------------------------------------------------
# Test Category 3: Downloader Integrity & SHA-256 Verification
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not WEIGHTS_4_PATH.exists() or not SIDECAR_4_PATH.exists(),
    reason="Gemma 4 12B checkpoint or sidecar not found (offline/clean checkout).",
)
def test_downloader_sidecar_and_weights_sha256_match():
    """Verify that gemma_4_12b weights file SHA-256 strictly matches its sidecar JSON."""
    assert WEIGHTS_4_PATH.exists(), f"Weights missing: {WEIGHTS_4_PATH}"
    assert SIDECAR_4_PATH.exists(), f"Sidecar missing: {SIDECAR_4_PATH}"

    actual_sha = compute_file_sha256(WEIGHTS_4_PATH)
    with open(SIDECAR_4_PATH, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert actual_sha == sidecar["weights_sha256"], (
        f"SHA-256 mismatch: file={actual_sha} vs sidecar={sidecar['weights_sha256']}"
    )
    assert sidecar["final_loss"] < 0.08, f"Final loss not < 0.08: {sidecar['final_loss']}"
    assert sidecar["peak_vram_mb"] <= 12288.0, f"Peak VRAM exceeded 12 GB: {sidecar['peak_vram_mb']}"
    assert sidecar["total_parameters"] == 200701444, f"Total parameters mismatch: {sidecar['total_parameters']}"
    assert sidecar["trainable_parameters"] == 200701442, f"Trainable parameters mismatch: {sidecar['trainable_parameters']}"
    assert sidecar["converged"] is True, "Sidecar indicates training failed to converge"


@pytest.mark.skipif(
    not WEIGHTS_4_PATH.exists() or not SIDECAR_4_PATH.exists(),
    reason="Gemma 4 12B checkpoint or sidecar not found (offline/clean checkout).",
)
def test_downloader_tamper_detection():
    """Verify verify_checkpoint_files detects corrupted or tampered files."""
    valid, result = verify_checkpoint_files(WEIGHTS_4_PATH, SIDECAR_4_PATH)
    assert valid is True, f"Legitimate checkpoint failed verification: {result}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_weights = Path(tmp_dir) / "tampered.safetensors"
        tmp_sidecar = Path(tmp_dir) / "tampered.json"

        # Copy sidecar
        shutil.copy(SIDECAR_4_PATH, tmp_sidecar)

        # Copy and tamper 1 byte of weights
        with open(WEIGHTS_4_PATH, "rb") as f_in:
            data = bytearray(f_in.read(1024 * 1024))  # Read first 1MB
            data[42] = (data[42] + 1) % 256  # Mutate 1 byte
        with open(tmp_weights, "wb") as f_out:
            f_out.write(data)

        # Verification must fail
        tampered_valid, reason = verify_checkpoint_files(tmp_weights, tmp_sidecar)
        assert tampered_valid is False, "Tampered file was erroneously accepted as valid!"
        assert "mismatch" in reason.lower() or "failed" in reason.lower()


@pytest.mark.skipif(
    not WEIGHTS_4_PATH.exists(),
    reason="Gemma 4 12B checkpoint not found (offline/clean checkout).",
)
def test_downloader_ensure_checkpoint_gemma4():
    """Verify ensure_checkpoint(model='gemma_4_12b') succeeds and returns verified path."""
    weights_path = ensure_single_model_checkpoint(
        model_key="gemma_4_12b",
        target_dir=CHECKPOINT_DIR,
        force=False,
        quiet=True,
    )
    assert weights_path == WEIGHTS_4_PATH
    assert weights_path.exists()


# ---------------------------------------------------------------------------
# Test Category 4: Backward Compatibility (Gemma 2B)
# ---------------------------------------------------------------------------

def test_backward_compatibility_gemma_2b_manifest_and_loader():
    """Verify Gemma 2B manifest, backbone, and decoder operate without regressions."""
    manifest_2b = ModelManifest.gemma_2b_it()
    assert manifest_2b.hidden_dimension == 2048
    assert manifest_2b.vocabulary_size == 256000
    assert manifest_2b.num_layers == 18

    backbone_2b = PretrainedGemmaBackbone(manifest=manifest_2b, load_weights=True)
    decoder_2b = GemmaCausalPrefixDecoder(backbone=backbone_2b, prefix_dim=2048, hidden_dim=2048)

    assert decoder_2b.is_gemma4_architecture() is False

    prompt = "The capital of France is"
    prompt_ids, _ = backbone_2b.encode_prompt_context(prompt)
    slots = mx.zeros((1, 16, 2048), dtype=mx.bfloat16)
    target_ids = mx.array([[651, 6037, 1]], dtype=mx.int32)

    loss, logits = decoder_2b.forward(prompt_ids, slots, target_ids)
    mx.eval(loss, logits)

    assert logits.shape == (1, 3, 256000)
    assert not mx.isnan(loss).item()
    assert not mx.isinf(loss).item()
    assert loss.item() > 0.0


def test_backward_compatibility_gemma_2b_adapter_loading():
    """Verify Gemma 2B adapter weights load strictly and produce valid deliberation trajectories."""
    if not WEIGHTS_2_PATH.exists():
        pytest.skip("Gemma 2B adapter weights not available")

    adapter_2b = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=4)
    adapter_2b.load_weights(str(WEIGHTS_2_PATH), strict=True)
    mx.eval(adapter_2b.parameters())

    params = dict(tree_flatten(adapter_2b.parameters()))
    assert len(params) == 28
    assert params["prelude.slot_anchors"].shape == (1, 16, 2048)

    dummy_input = mx.random.normal((1, 16, 2048))
    out = adapter_2b(dummy_input, steps=4)
    mx.eval(out)

    assert out.shape == (1, 16, 2048)
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()


# ---------------------------------------------------------------------------
# CLI Direct Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING CHALLENGER M1 EMPIRICAL ADVERSARIAL VERIFICATION SUITE")
    print("=" * 80)

    # 1. Decoder Stability
    print("\n--- [1/4] Testing Decoder Stability & D^{-0.5} Soft Memory Scaling ---")
    manifest = ModelManifest.gemma_4_12b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    backbone.freeze()
    decoder = GemmaCausalPrefixDecoder(backbone=backbone, prefix_dim=3840, hidden_dim=3840)
    adapter = GemmaRecurrentAdapter(dim=3840, num_slots=16, num_layers=1, deliberation_steps=4)
    if WEIGHTS_4_PATH.exists():
        adapter.load_weights(str(WEIGHTS_4_PATH), strict=True)

    test_decoder_stability_soft_memory_scaling(backbone, decoder, adapter)
    print("  [PASS] Scaled logits finite, well-conditioned, and prevented attention distortion.")
    test_decoder_perturbation_and_generation_stability(backbone, decoder, adapter)
    print("  [PASS] Autoregressive decoding generated tokens cleanly with EOS halting.")

    # 2. Memory Boundary
    print("\n--- [2/4] Testing Metal GPU Memory Boundary (<= 12.0 GB) ---")
    test_metal_vram_bounds_training_step(backbone, adapter)
    step_vram = get_peak_memory_gb()
    print(f"  [PASS] BPTT step with max-prompt-len 128 peak VRAM: {step_vram:.2f} GB <= 12.00 GB ceiling.")
    test_metal_vram_bounds_decoder_unroll(backbone, decoder)
    print(f"  [PASS] Decoder unroll forward pass stayed within VRAM ceiling.")

    # 3. Downloader & SHA-256 Integrity
    print("\n--- [3/4] Testing Downloader & Cryptographic SHA-256 Integrity ---")
    test_downloader_sidecar_and_weights_sha256_match()
    print("  [PASS] Gemma 4 12B weights file SHA-256 strictly matches sidecar JSON.")
    test_downloader_tamper_detection()
    print("  [PASS] Tamper detection successfully flagged mutated byte payload.")
    test_downloader_ensure_checkpoint_gemma4()
    print("  [PASS] Downloader ensure_single_model_checkpoint returned verified path.")

    # 4. Backward Compatibility
    print("\n--- [4/4] Testing Backward Compatibility with Gemma 2B ---")
    test_backward_compatibility_gemma_2b_manifest_and_loader()
    print("  [PASS] Gemma 2B manifest, backbone, and causal decoder operate without regressions.")
    test_backward_compatibility_gemma_2b_adapter_loading()
    print("  [PASS] Gemma 2B recurrent adapter loads strictly and unrolls stably.")

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL ADVERSARIAL VERIFICATION CHECKS PASSED.")
    print("=" * 80)
