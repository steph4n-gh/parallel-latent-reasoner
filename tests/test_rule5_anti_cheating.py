"""Tests for Non-Negotiable Evidence Rule 5: No configuration-shaped random model may be labeled Gemma."""

import pytest

from prlr.compact import CompactConfig, CompactScratchModel
from prlr.manifest import (
    ManifestError,
    ModelManifest,
    Rule5ViolationError,
    RuleViolationError,
)


def test_compact_model_cannot_be_labeled_gemma():
    """Rule 5: prlr.compact scratch models must explicitly reject Gemma labeling."""
    cfg = CompactConfig(dim=256, num_slots=16)
    model = CompactScratchModel(cfg)
    assert "gemma" not in model.name.lower()
    assert model.manifest.is_pretrained is False
    assert model.manifest.random_init is True


def test_random_model_with_gemma_id_rejected_by_manifest():
    """Rule 5: ModelManifest refuses random_init=True when model_id contains 'gemma'."""
    manifest = ModelManifest(
        model_id="gemma-2b-random-scratch",
        revision="test",
        architecture="GemmaForCausalLM",
        is_pretrained=False,
        random_init=True,
    )
    with pytest.raises((Rule5ViolationError, RuleViolationError, ValueError)) as excinfo:
        manifest.validate(allow_gemma_random_init=False)
    assert "Rule 5" in str(excinfo.value) or "gemma" in str(excinfo.value).lower()


def test_gemma_preset_without_weights_rejected():
    """Rule 5: Invoking prlr.gemma without valid pretrained checkpoint must fail."""
    from prlr.gemma import PretrainedGemmaBackbone

    with pytest.raises((ValueError, FileNotFoundError, RuleViolationError)):
        PretrainedGemmaBackbone(manifest=None)


def test_character_modulo_fallback_strictly_prohibited_in_gemma():
    """Rule 5: Character-modulo tokenization (ord(c) % vocab) is prohibited in gemma lane."""
    from prlr.gemma import PretrainedGemmaBackbone

    manifest = ModelManifest.get_gemma_2b_manifest()
    backbone = PretrainedGemmaBackbone(
        manifest=manifest,
        load_weights=False,
        allow_random_init=True,
    )
    with pytest.raises((TypeError, ValueError)) as excinfo:
        backbone.encode_prompt_context("Hello world")
    assert "character-modulo" in str(excinfo.value).lower() or "tokenizer" in str(excinfo.value).lower()


def test_allow_gemma_random_init_explicit_flag():
    """Verify that random init for Gemma is only possible with explicit opt-in flag."""
    manifest = ModelManifest(
        model_id="gemma-test-override",
        revision="test",
        architecture="GemmaForCausalLM",
        is_pretrained=False,
        random_init=True,
    )
    # Default without flag raises error
    with pytest.raises(Rule5ViolationError):
        manifest.validate(allow_gemma_random_init=False)

    # Explicit flag allows it for controlled synthetic unit tests
    assert manifest.validate(allow_gemma_random_init=True) is True


def test_sneaky_rule5_evasion_blocked():
    """Rule 5 regression: is_pretrained=False, random_init=False on Gemma ID raises (Rule5ViolationError, ManifestError)."""
    manifest = ModelManifest(
        model_id="google/gemma-2b-it",
        is_pretrained=False,
        random_init=False,
    )
    with pytest.raises((Rule5ViolationError, ManifestError)):
        manifest.validate(allow_gemma_random_init=False)


def test_pretrained_gemma_backbone_rejects_compact_testbed_without_override():
    """Rule 5 regression: PretrainedGemmaBackbone rejects compact testbed without allow_random_init=True."""
    from prlr.gemma.backbone import PretrainedGemmaBackbone

    compact_manifest = ModelManifest.compact_test()
    with pytest.raises(Rule5ViolationError) as excinfo:
        PretrainedGemmaBackbone(manifest=compact_manifest, allow_random_init=False)
    assert "requires a genuine pretrained checkpoint" in str(excinfo.value)
