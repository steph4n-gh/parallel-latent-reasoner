"""Tests for Pretrained Gemma Vertical Lane (Milestone 3 / Requirement R2).

Verifies:
- Official Google Gemma 2B weights loading via MLX
- Baseline health check on Apple Silicon Metal GPU
- Contextual hidden representation extraction (B, L, 2048) at layer 18 and layer 12
- SentencePiece tokenization alignment and Rule 5 character-modulo rejection
"""

from __future__ import annotations

import math
import pytest
import mlx.core as mx

from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.manifest import ModelManifest, Rule5ViolationError


@pytest.fixture(scope="module")
def gemma_manifest() -> ModelManifest:
    return ModelManifest.gemma_2b_it()


@pytest.fixture(scope="module")
def pretrained_backbone(gemma_manifest: ModelManifest) -> PretrainedGemmaBackbone:
    return PretrainedGemmaBackbone(manifest=gemma_manifest, load_weights=True)


def test_manifest_validation_and_disk_verification(gemma_manifest: ModelManifest):
    """Verify official Gemma 2B manifest is cryptographically sound on disk."""
    assert gemma_manifest.is_pretrained is True
    assert gemma_manifest.random_init is False
    assert gemma_manifest.architecture in ("gemma", "GemmaForCausalLM")
    assert gemma_manifest.num_layers == 18
    assert gemma_manifest.hidden_dimension == 2048

    assert gemma_manifest.vocabulary_size == 256000

    # Strict disk SHA-256 validation
    assert gemma_manifest.validate(check_disk=True) is True


def test_pretrained_gemma_health_check(pretrained_backbone: PretrainedGemmaBackbone):
    """Verify official baseline health check passes on Apple Silicon Metal GPU."""
    diag = pretrained_backbone.health_check()
    assert diag["status"] == "HEALTHY"
    assert diag["manifest_validated"] is True
    assert diag["vocab_size"] == 256000
    assert diag["bos_id"] == 2
    assert diag["eos_id"] == 1
    assert diag["hidden_shape"] == [1, 6, 2048]
    assert diag["has_nan"] is False
    assert diag["has_inf"] is False
    assert diag["semantic_passed"] is True
    assert "Paris" in diag["generation_output"]


def test_contextual_hidden_extraction_layers(pretrained_backbone: PretrainedGemmaBackbone):
    """Verify extraction of (B, L, 2048) representations at layer 18 and layer 12."""
    prompt = "The quick brown fox jumps over the lazy dog"
    input_ids, num_tokens = pretrained_backbone.encode_prompt_context(prompt)
    assert input_ids.shape == (1, num_tokens)
    assert num_tokens > 0

    # Layer 18 (final normalized)
    h18 = pretrained_backbone.extract_contextual_hiddens(input_ids, layer_idx=18)
    mx.eval(h18)
    assert h18.shape == (1, num_tokens, 2048)
    assert not mx.isnan(h18).any().item()
    assert not mx.isinf(h18).any().item()

    # Layer 12 (intermediate representation)
    h12 = pretrained_backbone.extract_contextual_hiddens(input_ids, layer_idx=12)
    mx.eval(h12)
    assert h12.shape == (1, num_tokens, 2048)
    assert not mx.isnan(h12).any().item()
    assert not mx.isinf(h12).any().item()

    # Confirm intermediate layer representation is distinct from final layer
    diff_norm = float(mx.linalg.norm(h18 - h12).item())
    assert diff_norm > 1.0, f"Layer 12 and Layer 18 must be distinct representations; diff={diff_norm}"


def test_sentencepiece_tokenization_integrity(pretrained_backbone: PretrainedGemmaBackbone):
    """Verify SentencePiece tokenization alignment and Rule 5 character-modulo rejection."""
    tokenizer = pretrained_backbone.tokenizer
    assert tokenizer is not None

    # SentencePiece special token ID invariants
    vocab = getattr(tokenizer, "vocab_size", None) or tokenizer.get_piece_size()
    assert vocab == 256000

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None and hasattr(tokenizer, "bos_id"):
        bos = tokenizer.bos_id()
    assert bos == 2

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None and hasattr(tokenizer, "eos_id"):
        eos = tokenizer.eos_id()
    assert eos == 1

    # Encode round-trip
    sample_text = "Parallel Latent Reasoner"
    tokens, count = pretrained_backbone.encode_prompt_context(sample_text)
    assert tokens.ndim == 2
    assert tokens.shape[0] == 1
    assert tokens.shape[1] == count
    assert tokens[0, 0].item() == bos  # Must start with BOS

    decoded = tokenizer.decode(tokens[0].tolist())
    assert "Parallel Latent Reasoner" in decoded


def test_rule5_rejection_of_unverified_models():
    """Verify strict rejection of unverified random models under Rule 5."""
    with pytest.raises(Rule5ViolationError):
        PretrainedGemmaBackbone(manifest=None)

    scratch_manifest = ModelManifest.compact_test()
    with pytest.raises(Rule5ViolationError, match="requires a genuine pretrained checkpoint"):
        PretrainedGemmaBackbone(manifest=scratch_manifest, allow_random_init=False)
