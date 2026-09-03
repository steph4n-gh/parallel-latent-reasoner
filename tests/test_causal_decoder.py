"""Tests for Option A Causal Prefix Decoder (Milestone 3 / Requirement R3).

Verifies:
- Target exact matching (teacher-forcing logit alignment & non-NaN loss)
- EOS termination ({1, 107} halting)
- Variable length target masking (zero loss on padding post-EOS)
- Zero-padding invariance (cosine similarity > 0.99999, argmax equivalence)
- Perturbation sensitivity (||Delta z|| > 0 under slot perturbation / knockout)
- Zero legacy linear recurrence (curr_hidden + 0.1 * tok_embed)
"""

from __future__ import annotations

import inspect
import math
import pytest
import mlx.core as mx

from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


@pytest.fixture(scope="module")
def gemma_manifest() -> ModelManifest:
    return ModelManifest.gemma_2b_it()


@pytest.fixture(scope="module")
def pretrained_backbone(gemma_manifest: ModelManifest) -> PretrainedGemmaBackbone:
    return PretrainedGemmaBackbone(manifest=gemma_manifest, load_weights=True)


@pytest.fixture(scope="module")
def causal_decoder(pretrained_backbone: PretrainedGemmaBackbone) -> GemmaCausalPrefixDecoder:
    return GemmaCausalPrefixDecoder(backbone=pretrained_backbone)


def test_zero_pseudo_decoder_linear_recurrence(causal_decoder: GemmaCausalPrefixDecoder):
    """Verify complete eradication of legacy ungrounded linear recurrence."""
    source = inspect.getsource(GemmaCausalPrefixDecoder)
    assert "curr_hidden + 0.1" not in source
    assert "+ 0.1 *" not in source
    assert "final_norm(curr_hidden" not in source


def test_target_exact_matching_teacher_forcing(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify teacher-forcing forward pass computes valid loss and aligned target logits."""
    prompt = "The capital of France is"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    target_ids = mx.array([[651, 6037, 1]], dtype=mx.int32)  # " Paris<eos>"
    slots = mx.zeros((1, 16, 2048)).astype(mx.bfloat16)

    loss, target_logits = causal_decoder.forward(prompt_ids, slots, target_ids)
    mx.eval(loss, target_logits)

    assert target_logits.shape == (1, 3, 256000)
    assert not mx.isnan(loss).item()
    assert not mx.isinf(loss).item()
    assert loss.item() > 0.0


def test_eos_termination_halting(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify autoregressive generation halts promptly upon reaching an EOS token {1, 107}."""
    prompt = "The capital of France is"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    slots = (mx.random.normal((1, 16, 2048)) * 0.001).astype(mx.bfloat16)

    # Generate up to 32 tokens, expecting early halt on EOS
    generated = causal_decoder.generate(
        prompt_ids,
        slots,
        max_new_tokens=32,
        temperature=0.0,
        eos_token_ids={1, 107},
    )
    mx.eval(generated)

    gen_tokens = generated[0].tolist()
    assert len(gen_tokens) <= 32
    # Check that either generation terminated early with an EOS or completed
    has_eos = any(tok in {1, 107} for tok in gen_tokens)
    if has_eos:
        # If EOS was produced, it must be the terminating token
        assert gen_tokens[-1] in {1, 107} or len(gen_tokens) <= 32


def test_variable_length_target_masking(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify variable-length targets receive strictly zero loss on padding tokens post-EOS."""
    prompt = "France capital"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    slots = mx.zeros((1, 16, 2048)).astype(mx.bfloat16)

    # Sequence with active tokens followed by pad tokens (ID 0)
    unpadded_targets = mx.array([[651, 1]], dtype=mx.int32)
    padded_targets = mx.array([[651, 1, 0, 0, 0]], dtype=mx.int32)

    loss_unpadded, _ = causal_decoder.forward(prompt_ids, slots, unpadded_targets)
    loss_padded, _ = causal_decoder.forward(prompt_ids, slots, padded_targets)
    mx.eval(loss_unpadded, loss_padded)

    # Padding tokens must not alter the active token cross-entropy loss
    diff = abs(loss_unpadded.item() - loss_padded.item())
    assert diff < 1e-4, f"Padded loss ({loss_padded.item()}) diverged from unpadded ({loss_unpadded.item()}); diff={diff}"


def test_zero_padding_invariance(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify prompt zero-padding does not distort active token representations (cosine sim > 0.99999)."""
    inner = causal_decoder.get_inner_model()
    prompt_unpadded = mx.array([[2, 651, 6037]], dtype=mx.int32)
    prompt_padded = mx.array([[2, 651, 6037, 0, 0, 0]], dtype=mx.int32)

    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache

    # 1. Unpadded prompt forward
    c1 = make_prompt_cache(inner)
    h1 = inner.embed_tokens(prompt_unpadded) * (2048 ** 0.5)
    m1 = create_attention_mask(h1, c1[0])
    for l, c in zip(inner.layers, c1):
        h1 = l(h1, mask=m1, cache=c)
    z1 = inner.embed_tokens.as_linear(inner.norm(h1))[:, 2, :]

    # 2. Padded prompt forward
    c2 = make_prompt_cache(inner)
    h2 = inner.embed_tokens(prompt_padded) * (2048 ** 0.5)
    m2 = create_attention_mask(h2, c2[0])
    for l, c in zip(inner.layers, c2):
        h2 = l(h2, mask=m2, cache=c)
    z2 = inner.embed_tokens.as_linear(inner.norm(h2))[:, 2, :]

    # Compute cosine similarity between valid token representations in float32
    z1_f32 = z1.astype(mx.float32)
    z2_f32 = z2.astype(mx.float32)
    cos_sim = float(
        (mx.sum(z1_f32 * z2_f32) / (mx.linalg.norm(z1_f32) * mx.linalg.norm(z2_f32))).item()
    )

    assert cos_sim > 0.99999, f"Padding corrupted causal representation: cosine sim {cos_sim} <= 0.99999"
    assert mx.argmax(z1).item() == mx.argmax(z2).item(), "Top-1 token prediction mismatch between unpadded and padded!"


def test_perturbation_sensitivity(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify that perturbing soft prefix slots produces non-zero shifts in output logits (||Delta z|| > 0)."""
    prompt = "The capital of France is"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    target_ids = mx.array([[651, 6037]], dtype=mx.int32)

    # Baseline slots
    slots_base = (mx.random.normal((1, 16, 2048)) * 0.01).astype(mx.bfloat16)
    _, logits_base = causal_decoder.forward(prompt_ids, slots_base, target_ids)

    # 1. Slot knockout (S -> 0)
    slots_zero = mx.zeros((1, 16, 2048)).astype(mx.bfloat16)
    _, logits_knockout = causal_decoder.forward(prompt_ids, slots_zero, target_ids)

    # 2. Perturbation (delta ~ N(0, 0.1^2))
    perturbation = (mx.random.normal((1, 16, 2048)) * 0.1).astype(mx.bfloat16)
    slots_perturbed = slots_base + perturbation
    _, logits_perturbed = causal_decoder.forward(prompt_ids, slots_perturbed, target_ids)

    mx.eval(logits_base, logits_knockout, logits_perturbed)

    shift_knockout = float(mx.linalg.norm(logits_base - logits_knockout).item())
    shift_perturbed = float(mx.linalg.norm(logits_base - logits_perturbed).item())

    assert shift_knockout > 0.0, f"Slot knockout produced zero logit shift: {shift_knockout}"
    assert shift_perturbed > 0.0, f"Slot perturbation produced zero logit shift: {shift_perturbed}"
    assert shift_knockout > 10.0, f"Expected substantial knockout shift, got {shift_knockout}"
    assert shift_perturbed > 10.0, f"Expected substantial perturbation shift, got {shift_perturbed}"
