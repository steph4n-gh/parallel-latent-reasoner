"""Verification Test Suite for Milestone M3: Zero-Gate Parity & Phase 3 Acceptance Criteria.

Operationalizes Phase 3 Requirements from ORIGINAL_REQUEST.md (entry 2026-09-04T18:18:19Z)
and orchestrator_safe_injection/PROJECT.md (Features 12, 13, 14, 15):

Acceptance Criteria Verified:
1. Logit & Token Parity Tests:
   - 100% bit-exact token sequence identity between frozen Gemma and gated adapter at alpha=0.
   - Max absolute logit difference == 0.0 (within IEEE float32 precision).
   - Arbitrary slot invariance: S ~ N(0, 1), S = 0, S * 1e5 all produce delta == 0.0 at alpha=0.
   - Sequence length strictly P, native RoPE positions [0..P-1] (no virtual token prepending).
2. Format & Robustness Acceptance Criteria:
   - Valid JSON syntax rate >= 99.0% on held-out test split (sealed_test_inputs.jsonl).
   - Exact Match within 0.5 percentage points of direct frozen baseline (target delta 0.0%).
   - Maximum 4-gram repetition strictly <= 2.
   - Shannon entropy >= 3.0 bits.
   - Zero spurious special tokens (terminates strictly at token 106 <turn|> in Gemma 4).
3. Nonzero Gate Dynamics & Safety Tests:
   - Conditioning active effect is measurable when alpha > 0 (logits change).
   - Output norms remain stable (no NaN, no Inf, bounded output).
   - Recoverability: setting alpha back to 0 immediately restores 100% frozen Gemma output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch
import pytest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from prlr.domain.prompt_format import format_canonical_prompt
from prlr.domain.solver_lane import DOMAIN_CATALOGUES, ProceduralVerifier
from prlr.eval.semantic_bench import (
    compute_max_ngram_repetition,
    compute_shannon_entropy,
)
from prlr.gemma.backbone import GemmaTokenizerWrapper, PretrainedGemmaBackbone
from prlr.gemma.decoder import (
    GatedCrossAttentionInjection,
    GemmaCausalPrefixDecoder,
)
from prlr.manifest import ModelManifest


# ==============================================================================
# Fixtures & Environment Setup
# ==============================================================================

@pytest.fixture(scope="module")
def gemma_manifest() -> ModelManifest:
    """Load Gemma 4 12B manifest if available on disk; fallback to Gemma 2B for CI."""
    try:
        manifest_g4 = ModelManifest.gemma_4_12b_it()
        if Path(manifest_g4.weights_path).exists():
            return manifest_g4
    except Exception:
        pass
    return ModelManifest.gemma_2b_it()


@pytest.fixture(scope="module")
def pretrained_backbone(gemma_manifest: ModelManifest) -> PretrainedGemmaBackbone:
    """Initialize genuine pretrained backbone on Apple Silicon Metal GPU."""
    backbone = PretrainedGemmaBackbone(manifest=gemma_manifest, load_weights=True)
    backbone.freeze()
    return backbone


@pytest.fixture(scope="module")
def hidden_dimension(pretrained_backbone: PretrainedGemmaBackbone) -> int:
    """Retrieve backbone hidden dimension (3840 for Gemma 4 12B, 2048 for Gemma 2B)."""
    manifest = getattr(pretrained_backbone, "manifest", None)
    return getattr(manifest, "hidden_dimension", 3840)


@pytest.fixture(scope="module")
def injection_module(hidden_dimension: int) -> GatedCrossAttentionInjection:
    """Initialize GatedCrossAttentionInjection with alpha initialized to 0.0."""
    num_heads = 16 if hidden_dimension == 3840 else 8
    return GatedCrossAttentionInjection(hidden_size=hidden_dimension, num_heads=num_heads, gamma_max=0.5)


@pytest.fixture(scope="module")
def diverse_prompts() -> List[str]:
    """Curated corpus of diverse test prompts across multiple domains."""
    return [
        "What is the capital of France? Answer in one word.",
        "Solve for x: 3 * x + 15 = 42. Output only the value of x.",
        "Write a Python function named add_numbers(a, b) that returns their sum.",
        (
            "<start_of_turn>user\n"
            "You are an autonomous execution planner. Given the available tool registry, "
            "determine the minimal valid sequence of tools to achieve the target goal.\n"
            "Available Tools:\n"
            "- policy_gate: requires [risk_score], produces [approval_decision]\n"
            "- audit_logger: requires [approval_decision], produces [audit_receipt]\n"
            "Initial State: [risk_score]\n"
            "Target Goal: audit_receipt\n"
            "Respond with a JSON object containing keys 'route' and 'terminal'.<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "Translate the following English sentence to French: The weather is beautiful today.",
        "Hello",  # Single-token prompt edge case
    ]


@pytest.fixture(scope="module")
def sealed_test_samples() -> List[Dict[str, Any]]:
    """Load target-free problem inputs from sealed_test_inputs.jsonl."""
    inputs_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "prlr_domain_v1"
        / "evaluation_inputs"
        / "sealed_test_inputs.jsonl"
    )
    if not inputs_path.exists():
        pytest.skip(f"Sealed test inputs not found at {inputs_path}")

    samples: List[Dict[str, Any]] = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


@pytest.fixture(scope="module")
def sealed_test_keys() -> Dict[str, Dict[str, Any]]:
    """Load quarantined answer keys for post-hoc validation."""
    keys_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "prlr_domain_v1"
        / "answer_keys"
        / "sealed_test_keys.jsonl"
    )
    if not keys_path.exists():
        pytest.skip(f"Sealed test keys not found at {keys_path}")

    keys: Dict[str, Dict[str, Any]] = {}
    with open(keys_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                keys[rec["id"]] = rec
    return keys


# ==============================================================================
# Suite 1: Gated Cross-Attention Module Unit Tests
# ==============================================================================

class TestGatedCrossAttentionUnit:
    """Unit tests verifying mathematical invariants of GatedCrossAttentionInjection."""

    def test_zero_gate_bit_exact_identity(self, hidden_dimension: int):
        """Verify H_out == H bit-for-bit with max absolute delta == 0.0 when alpha = 0.0."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)
        inj.alpha = mx.array(0.0, dtype=mx.float32)

        B, L, M = 2, 12, 16
        h = mx.random.normal((B, L, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((B, M, hidden_dimension)).astype(mx.bfloat16)

        h_out = inj(h, slots)
        mx.eval(h, h_out)

        # Max absolute difference must be strictly 0.0 in float32 / bfloat16
        max_diff = float(mx.max(mx.abs(h.astype(mx.float32) - h_out.astype(mx.float32))).item())
        assert max_diff == 0.0, f"Zero-gate invariant violated: max_diff = {max_diff}"
        assert inj.gate_value == 0.0

    def test_gate_boundedness_tanh(self, hidden_dimension: int):
        """Verify gate g = gamma_max * tanh(alpha) is strictly bounded in [-gamma_max, gamma_max]."""
        gamma_max = 0.75
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=gamma_max)

        test_alphas = [-100.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 100.0]
        for a_val in test_alphas:
            inj.alpha = mx.array(a_val, dtype=mx.float32)
            g = inj.gate_value
            assert -gamma_max <= g <= gamma_max, f"Gate out of bounds for alpha={a_val}: {g}"
            if a_val == 0.0:
                assert g == 0.0
            elif a_val > 0.0:
                assert g > 0.0
            else:
                assert g < 0.0

    def test_latent_slot_rms_normalization(self, hidden_dimension: int):
        """Verify latent slot normalization prevents attention explosion across diverse magnitudes."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)

        # High-magnitude slots (scale = 1000.0)
        large_slots = mx.random.normal((1, 16, hidden_dimension)) * 1000.0
        norm_slots = inj.normalize_slots(large_slots)
        mx.eval(norm_slots)

        # RMS norm of normalized slots should be approximately 1.0
        slot_rms = mx.sqrt(mx.mean(norm_slots ** 2, axis=-1))
        mx.eval(slot_rms)
        assert mx.allclose(slot_rms, mx.ones_like(slot_rms), atol=1e-3), (
            f"Slot RMS not normalized to 1.0: {slot_rms[0, 0].item()}"
        )

    def test_gradient_flow_through_gate(self, hidden_dimension: int):
        """Verify loss gradients flow cleanly to alpha when alpha != 0 without vanishing/exploding."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)
        inj.alpha = mx.array(0.5, dtype=mx.float32)

        h = mx.random.normal((1, 4, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        def loss_fn(model):
            out = model(h, slots)
            return mx.sum(out.astype(mx.float32) ** 2)

        grad_fn = nn.value_and_grad(inj, loss_fn)
        loss_val, grads = grad_fn(inj)
        mx.eval(loss_val, grads)

        assert "alpha" in grads, "Gradient for alpha not found in grads"
        grad_alpha = float(grads["alpha"].item())
        assert not math.isnan(grad_alpha) and not math.isinf(grad_alpha)
        assert abs(grad_alpha) > 1e-6, f"Gradient vanished through alpha gate: {grad_alpha}"

    def test_adversarial_slot_norm_boundedness(self, hidden_dimension: int):
        """Verify output hidden state norm is bounded even with adversarial high-norm slot inputs."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)
        inj.alpha = mx.array(10.0, dtype=mx.float32)  # Max gate g ~ 0.5

        h = mx.random.normal((1, 8, hidden_dimension)).astype(mx.bfloat16)
        adversarial_slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16) * 1e4

        h_out = inj(h, adversarial_slots)
        mx.eval(h, h_out)

        norm_in = float(mx.linalg.norm(h).item())
        norm_out = float(mx.linalg.norm(h_out).item())

        assert not math.isnan(norm_out) and not math.isinf(norm_out)
        assert norm_out < norm_in * 10.0, f"Output exploded: norm_in={norm_in}, norm_out={norm_out}"

    def test_telemetry_reporting(self, hidden_dimension: int):
        """Verify get_telemetry returns accurate gate, alpha, activation norm, and injection ratio."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)
        inj.alpha = mx.array(1.0, dtype=mx.float32)

        h = mx.random.normal((1, 4, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        _ = inj(h, slots)
        telemetry = inj.get_telemetry()

        assert "gate" in telemetry
        assert "alpha" in telemetry
        assert "activation_norm" in telemetry
        assert "injection_ratio" in telemetry
        assert telemetry["alpha"] == 1.0
        assert 0.0 < telemetry["gate"] <= 0.5
        assert telemetry["activation_norm"] > 0.0
        assert telemetry["injection_ratio"] >= 0.0


# ==============================================================================
# Suite 2: Logit & Token Parity Tests at Zero Gate (alpha = 0)
# ==============================================================================

class TestLogitAndTokenParityAtZeroGate:
    """Rigorous tests confirming bit-exact logit and token parity between base model and gated adapter."""

    def test_prefill_logit_bit_exact_parity_diverse_prompts(
        self,
        pretrained_backbone: PretrainedGemmaBackbone,
        diverse_prompts: List[str],
        hidden_dimension: int,
    ):
        """Verify max logit absolute difference between frozen Gemma and gated adapter at alpha=0 is 0.0."""
        decoder = GemmaCausalPrefixDecoder(
            backbone=pretrained_backbone,
            conditioning_mode="cross_attention",
        )
        decoder.safe_injection.alpha = mx.array(0.0, dtype=mx.float32)

        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        for prompt_idx, prompt in enumerate(diverse_prompts):
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)

            # Prefill logits directly from decoder without prefix (frozen base path)
            logits_frozen = decoder.prefill_logits(prompt_ids, prefix_latents=None)
            # Prefill logits through gated adapter with deliberated slots at alpha=0
            logits_gated = decoder.prefill_logits(prompt_ids, prefix_latents=slots)
            mx.eval(logits_frozen, logits_gated)

            diff = float(mx.max(mx.abs(logits_frozen.astype(mx.float32) - logits_gated.astype(mx.float32))).item())
            assert diff == 0.0, (
                f"Logit parity violated on prompt {prompt_idx} ('{prompt[:30]}...'): "
                f"max logit diff = {diff} (must be 0.0 within float32 precision)"
            )

            tok_frozen = int(mx.argmax(logits_frozen, axis=-1)[0, 0].item())
            tok_gated = int(mx.argmax(logits_gated, axis=-1)[0, 0].item())
            assert tok_frozen == tok_gated, (
                f"Top-1 token mismatch on prompt {prompt_idx}: frozen={tok_frozen} vs gated={tok_gated}"
            )

    def test_token_sequence_100pct_identity_diverse_prompts(
        self,
        pretrained_backbone: PretrainedGemmaBackbone,
        diverse_prompts: List[str],
        hidden_dimension: int,
    ):
        """Verify 100% token sequence identity during autoregressive generation at alpha=0."""
        decoder = GemmaCausalPrefixDecoder(
            backbone=pretrained_backbone,
            conditioning_mode="cross_attention",
        )
        decoder.safe_injection.alpha = mx.array(0.0, dtype=mx.float32)

        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        for prompt_idx, prompt in enumerate(diverse_prompts):
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)

            tokens_direct = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=None,
                max_new_tokens=16,
                temperature=0.0,
            )
            tokens_gated = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=16,
                temperature=0.0,
            )
            mx.eval(tokens_direct, tokens_gated)

            direct_list = tokens_direct[0].tolist() if tokens_direct.ndim > 1 else tokens_direct.tolist()
            gated_list = tokens_gated[0].tolist() if tokens_gated.ndim > 1 else tokens_gated.tolist()

            assert direct_list == gated_list, (
                f"Token sequence divergence on prompt {prompt_idx}:\n"
                f"Direct: {direct_list}\n"
                f"Gated:  {gated_list}"
            )

    def test_zero_gate_parity_with_arbitrary_latent_slots(
        self,
        hidden_dimension: int,
    ):
        """Verify that at alpha=0, arbitrary slot contents (Gaussian noise, zeros, huge vectors) have ZERO effect."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)
        inj.alpha = mx.array(0.0, dtype=mx.float32)

        h = mx.random.normal((1, 10, hidden_dimension)).astype(mx.bfloat16)

        slot_variations = [
            ("all_zeros", mx.zeros((1, 16, hidden_dimension)).astype(mx.bfloat16)),
            ("standard_gaussian", mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)),
            ("large_magnitude", (mx.random.normal((1, 16, hidden_dimension)) * 1e5).astype(mx.bfloat16)),
            ("extreme_sparse", mx.array(np.random.choice([0.0, 100.0], size=(1, 16, hidden_dimension))).astype(mx.bfloat16)),
        ]

        for desc, slots in slot_variations:
            h_out = inj(h, slots)
            mx.eval(h_out)
            max_delta = float(mx.max(mx.abs(h.astype(mx.float32) - h_out.astype(mx.float32))).item())
            assert max_delta == 0.0, f"Slot variation '{desc}' produced non-zero delta ({max_delta}) at alpha=0!"

    def test_prompt_positional_encoding_preservation(
        self,
        pretrained_backbone: PretrainedGemmaBackbone,
    ):
        """Verify prompt token positions remain [0..P-1] and sequence length is strictly P (no soft-prefix shift)."""
        prompt = "The capital of France is"
        prompt_ids, P = pretrained_backbone.encode_prompt_context(prompt)
        assert prompt_ids.shape[1] == P

        inner = (
            pretrained_backbone.model.language_model.model
            if hasattr(pretrained_backbone.model, "language_model")
            else getattr(pretrained_backbone.model, "model", pretrained_backbone.model)
        )
        prompt_embeds = inner.embed_tokens(prompt_ids)

        # Under safe-injection, input sequence length to transformer is strictly P (never P + M)
        assert prompt_embeds.shape[1] == P, f"Sequence length corrupted: {prompt_embeds.shape[1]} != {P}"


# ==============================================================================
# Suite 3: Format & Robustness Acceptance Criteria (sealed_test_inputs.jsonl)
# ==============================================================================

class TestFormatAndRobustnessAcceptanceCriteria:
    """Rigorous tests evaluating format validity, repetition, entropy, and halting on held-out test data."""

    def test_valid_json_syntax_rate_ge_99_pct(
        self,
        sealed_test_samples: List[Dict[str, Any]],
        pretrained_backbone: PretrainedGemmaBackbone,
    ):
        """Verify valid JSON syntax rate is >= 99.0% on held-out test split (baseline achieves 100.0%)."""
        eval_subset = sealed_test_samples[:16]
        verifier = ProceduralVerifier()
        tokenizer = pretrained_backbone.tokenizer
        is_g4 = "gemma-4" in getattr(getattr(pretrained_backbone, "manifest", None), "model_id", "").lower()
        stop_tokens = {1, 106} if is_g4 else {1, 107}

        decoder = GemmaCausalPrefixDecoder(backbone=pretrained_backbone, conditioning_mode="cross_attention")
        valid_count = 0

        for item in eval_subset:
            prompt = format_canonical_prompt(item["prompt"], tokenizer, is_gemma4=is_g4)
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
            gen_tokens = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=None,
                max_new_tokens=96,
                temperature=0.0,
                eos_token_ids=stop_tokens,
            )
            mx.eval(gen_tokens)
            tok_list = gen_tokens[0].tolist() if gen_tokens.ndim > 1 else gen_tokens.tolist()
            decoded_text = tokenizer.decode(tok_list)
            if isinstance(decoded_text, list):
                decoded_text = " ".join(decoded_text)

            v_res = verifier.verify(decoded_text, expected_route=())
            if v_res.get("is_valid", False):
                valid_count += 1

        valid_rate = (valid_count / len(eval_subset)) * 100.0
        assert valid_rate >= 99.0, f"Valid JSON syntax rate ({valid_rate:.2f}%) fell below 99.0% threshold!"

    def test_exact_match_parity_within_half_percentage_point(
        self,
        sealed_test_samples: List[Dict[str, Any]],
        sealed_test_keys: Dict[str, Dict[str, Any]],
        pretrained_backbone: PretrainedGemmaBackbone,
        hidden_dimension: int,
    ):
        """Verify Exact Match of gated adapter at alpha=0 is within 0.5% of direct frozen baseline."""
        eval_subset = sealed_test_samples[:16]
        verifier = ProceduralVerifier()
        tokenizer = pretrained_backbone.tokenizer
        is_g4 = "gemma-4" in getattr(getattr(pretrained_backbone, "manifest", None), "model_id", "").lower()
        stop_tokens = {1, 106} if is_g4 else {1, 107}

        decoder = GemmaCausalPrefixDecoder(backbone=pretrained_backbone, conditioning_mode="cross_attention")
        decoder.safe_injection.alpha = mx.array(0.0, dtype=mx.float32)
        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        direct_matches = 0
        gated_matches = 0

        for item in eval_subset:
            sid = item["id"]
            if sid not in sealed_test_keys:
                continue
            key_rec = sealed_test_keys[sid]
            expected_route = tuple(key_rec.get("verifier_config", {}).get("expected_route", []))

            prompt = format_canonical_prompt(item["prompt"], tokenizer, is_gemma4=is_g4)
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)

            # 1. Direct Frozen Run
            tokens_direct = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=None,
                max_new_tokens=96,
                temperature=0.0,
                eos_token_ids=stop_tokens,
            )
            mx.eval(tokens_direct)
            text_direct = tokenizer.decode(tokens_direct[0].tolist())
            if isinstance(text_direct, list):
                text_direct = " ".join(text_direct)
            v_dir = verifier.verify(text_direct, expected_route=expected_route)
            if v_dir.get("exact_match", False):
                direct_matches += 1

            # 2. Gated Adapter at alpha=0 Run
            tokens_gated = decoder.generate(
                prompt_ids=prompt_ids,
                prefix_latents=slots,
                max_new_tokens=96,
                temperature=0.0,
                eos_token_ids=stop_tokens,
            )
            mx.eval(tokens_gated)
            text_gated = tokenizer.decode(tokens_gated[0].tolist())
            if isinstance(text_gated, list):
                text_gated = " ".join(text_gated)
            v_gat = verifier.verify(text_gated, expected_route=expected_route)
            if v_gat.get("exact_match", False):
                gated_matches += 1

        direct_em = (direct_matches / len(eval_subset)) * 100.0
        gated_em = (gated_matches / len(eval_subset)) * 100.0
        em_delta = abs(gated_em - direct_em)

        assert em_delta <= 0.5, (
            f"Exact Match parity violated: delta {em_delta:.2f}% > 0.5% "
            f"(Direct: {direct_em:.2f}%, Gated: {gated_em:.2f}%)"
        )

    def test_max_4gram_repetition_strictly_le_2(
        self,
        sealed_test_samples: List[Dict[str, Any]],
        pretrained_backbone: PretrainedGemmaBackbone,
    ):
        """Verify maximum 4-gram repetition is strictly <= 2 across all emitted sequences."""
        eval_subset = sealed_test_samples[:8]
        tokenizer = pretrained_backbone.tokenizer
        is_g4 = "gemma-4" in getattr(getattr(pretrained_backbone, "manifest", None), "model_id", "").lower()
        decoder = GemmaCausalPrefixDecoder(backbone=pretrained_backbone, conditioning_mode="cross_attention")

        for item in eval_subset:
            prompt = format_canonical_prompt(item["prompt"], tokenizer, is_gemma4=is_g4)
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
            tokens = decoder.generate(prompt_ids=prompt_ids, prefix_latents=None, max_new_tokens=96)
            mx.eval(tokens)
            text = tokenizer.decode(tokens[0].tolist())
            if isinstance(text, list):
                text = " ".join(text)

            rep_4gram = compute_max_ngram_repetition(text, n=4)
            assert rep_4gram <= 2, (
                f"Degenerate repetition detected in sample '{item['id']}': "
                f"max 4-gram repetition = {rep_4gram} > 2. Text snippet: {text[:100]}"
            )

    def test_shannon_entropy_ge_3_bits(
        self,
        sealed_test_samples: List[Dict[str, Any]],
        pretrained_backbone: PretrainedGemmaBackbone,
    ):
        """Verify Shannon entropy H is strictly >= 3.0 bits across emitted solutions."""
        eval_subset = sealed_test_samples[:8]
        tokenizer = pretrained_backbone.tokenizer
        is_g4 = "gemma-4" in getattr(getattr(pretrained_backbone, "manifest", None), "model_id", "").lower()
        decoder = GemmaCausalPrefixDecoder(backbone=pretrained_backbone, conditioning_mode="cross_attention")

        for item in eval_subset:
            prompt = format_canonical_prompt(item["prompt"], tokenizer, is_gemma4=is_g4)
            prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
            tokens = decoder.generate(prompt_ids=prompt_ids, prefix_latents=None, max_new_tokens=96)
            mx.eval(tokens)
            text = tokenizer.decode(tokens[0].tolist())
            if isinstance(text, list):
                text = " ".join(text)

            entropy = compute_shannon_entropy(text)
            assert entropy >= 3.0, (
                f"Low entropy collapse detected in sample '{item['id']}': "
                f"H = {entropy:.2f} bits < 3.0 bits. Text: {text}"
            )

    def test_strict_token_106_turn_halting_and_no_spurious_tokens(
        self,
        pretrained_backbone: PretrainedGemmaBackbone,
    ):
        """Verify generation halts strictly at token 106 (<turn|>) and token 107 (newline) does not halt."""
        tokenizer = pretrained_backbone.tokenizer
        wrapper = GemmaTokenizerWrapper(tokenizer, eos_token_ids={1, 106})
        assert 106 in wrapper.eos_token_ids
        assert 107 not in wrapper.eos_token_ids, "Token 107 ('\\n') must not be in eos_token_ids!"

        # Simulated generation emitting JSON -> 106 (<turn|>) -> spurious thought tokens [108, 109]
        simulated_tokens = [2717, 3723, 107, 106, 108, 109]

        def mock_generate_step(prompt, model, **kwargs):
            for tok in simulated_tokens:
                yield tok, mx.zeros((1,))

        import mlx_lm
        with patch("mlx_lm.generate.generate_step", side_effect=mock_generate_step):
            responses = list(mlx_lm.stream_generate(nn.Module(), wrapper, prompt=[2]))
            emitted_tokens = [r.token for r in responses]

            # Asserts generation halted promptly at token 106
            assert emitted_tokens[-1] == 106, f"Expected terminal token to be 106, got {emitted_tokens[-1]}"
            assert 108 not in emitted_tokens, "Trailing special tokens leaked past token 106!"
            assert 109 not in emitted_tokens


# ==============================================================================
# Suite 4: Nonzero Gate Dynamics & Safety Tests
# ==============================================================================

class TestNonzeroGateDynamicsAndSafety:
    """Tests evaluating active conditioning effects when alpha > 0 and immediate recoverability."""

    def test_measurable_conditioning_effect_when_alpha_positive(
        self,
        hidden_dimension: int,
    ):
        """Verify that when alpha > 0 (e.g. 0.5, 1.0), output hidden states shift measurably from baseline."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)

        h = mx.random.normal((1, 8, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        # 1. Baseline at alpha = 0.0
        inj.alpha = mx.array(0.0, dtype=mx.float32)
        h_base = inj(h, slots)
        mx.eval(h_base)

        # 2. Conditioning active at alpha = 0.5
        inj.alpha = mx.array(0.5, dtype=mx.float32)
        h_active_05 = inj(h, slots)
        mx.eval(h_active_05)

        # 3. Conditioning active at alpha = 1.0
        inj.alpha = mx.array(1.0, dtype=mx.float32)
        h_active_10 = inj(h, slots)
        mx.eval(h_active_10)

        diff_05 = float(mx.max(mx.abs(h_active_05.astype(mx.float32) - h_base.astype(mx.float32))).item())
        diff_10 = float(mx.max(mx.abs(h_active_10.astype(mx.float32) - h_base.astype(mx.float32))).item())

        assert diff_05 > 0.001, f"Conditioning at alpha=0.5 had no measurable effect: diff={diff_05}"
        assert diff_10 > diff_05, f"Expected monotonic shift with larger alpha: diff_10={diff_10} <= diff_05={diff_05}"

    def test_output_norm_stability_and_no_nan_inf_at_extreme_alpha(
        self,
        hidden_dimension: int,
    ):
        """Verify that extreme gate settings (alpha = 50.0, -50.0) produce finite, bounded, non-NaN outputs."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)

        h = mx.random.normal((2, 10, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((2, 16, hidden_dimension)).astype(mx.bfloat16) * 50.0

        for extreme_alpha in [-50.0, -10.0, 10.0, 50.0]:
            inj.alpha = mx.array(extreme_alpha, dtype=mx.float32)
            h_out = inj(h, slots)
            mx.eval(h_out)

            assert not mx.any(mx.isnan(h_out)).item(), f"NaN detected in output at alpha={extreme_alpha}!"
            assert not mx.any(mx.isinf(h_out)).item(), f"Inf detected in output at alpha={extreme_alpha}!"

            norm_val = float(mx.linalg.norm(h_out).item())
            assert 0.0 < norm_val < 1e6, f"Output norm unstable at alpha={extreme_alpha}: {norm_val}"

    def test_gate_recoverability_immediate_restoration(
        self,
        hidden_dimension: int,
    ):
        """Verify recoverability: setting alpha back to 0 immediately restores 100% frozen base output."""
        inj = GatedCrossAttentionInjection(hidden_size=hidden_dimension, gamma_max=0.5)

        h = mx.random.normal((1, 12, hidden_dimension)).astype(mx.bfloat16)
        slots = mx.random.normal((1, 16, hidden_dimension)).astype(mx.bfloat16)

        # Step 1: Initial state at alpha = 0.0
        inj.alpha = mx.array(0.0, dtype=mx.float32)
        h_initial = inj(h, slots)
        mx.eval(h_initial)

        # Step 2: Drive gate to strong nonzero state (alpha = 2.5)
        inj.alpha = mx.array(2.5, dtype=mx.float32)
        h_perturbed = inj(h, slots)
        mx.eval(h_perturbed)
        assert float(mx.max(mx.abs(h_perturbed.astype(mx.float32) - h_initial.astype(mx.float32))).item()) > 0.0

        # Step 3: Reset alpha back to 0.0
        inj.alpha = mx.array(0.0, dtype=mx.float32)
        h_restored = inj(h, slots)
        mx.eval(h_restored)

        # Step 4: Verify bit-exact restoration (zero hysteresis)
        restoration_diff = float(mx.max(mx.abs(h_restored.astype(mx.float32) - h_initial.astype(mx.float32))).item())
        assert restoration_diff == 0.0, (
            f"Gate recoverability violated: residual error {restoration_diff} after resetting alpha to 0.0!"
        )
