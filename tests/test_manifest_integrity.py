"""Tests for ModelManifest cryptographic validation and load_model integrity."""

import hashlib
from pathlib import Path
import pytest

from prlr.manifest import (
    ArchitectureMismatchError,
    IntegrityError,
    ManifestError,
    ModelManifest,
    load_model,
)


@pytest.fixture
def dummy_weight_and_tokenizer(tmp_path: Path):
    """Create miniature valid test files and compute exact SHA256 hashes."""
    weight_file = tmp_path / "model.safetensors"
    weight_bytes = b"SIMULATED_SAFETENSORS_HEADER_AND_TENSORS_12345"
    weight_file.write_bytes(weight_bytes)
    weight_sha = hashlib.sha256(weight_bytes).hexdigest()

    tok_file = tmp_path / "tokenizer.model"
    tok_bytes = b"SIMULATED_SENTENCEPIECE_PROTOBUF_67890"
    tok_file.write_bytes(tok_bytes)
    tok_sha = hashlib.sha256(tok_bytes).hexdigest()

    manifest = ModelManifest(
        model_id="test-pretrained-model",
        revision="main",
        architecture="GemmaForCausalLM",
        weights_path=str(weight_file),
        weights_sha256={"model.safetensors": weight_sha},
        tokenizer_path=str(tok_file),
        tokenizer_sha256=tok_sha,
        vocabulary_size=1000,
        hidden_dimension=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dimension=32,
        intermediate_dimension=128,
        quantization="none",
        runtime_versions={"mlx": "0.31.2"},
        is_pretrained=True,
        random_init=False,
        adapter_hash=None,
        source_commit="abcdef0",
        verification_status="UNVERIFIED",
    )
    return tmp_path, manifest, weight_file, tok_file


def test_corrupted_weight_hash_raises_integrity_error(dummy_weight_and_tokenizer):
    """Test 2a: Tampered byte in weight file -> must raise IntegrityError or ValueError."""
    tmp_path, manifest, weight_file, _ = dummy_weight_and_tokenizer
    data = bytearray(weight_file.read_bytes())
    data[10] ^= 0xFF
    weight_file.write_bytes(bytes(data))

    with pytest.raises((IntegrityError, ValueError)) as excinfo:
        manifest.validate(check_disk=True)
    assert "mismatch" in str(excinfo.value).lower() or "hash" in str(excinfo.value).lower()


def test_corrupted_tokenizer_hash_raises_integrity_error(dummy_weight_and_tokenizer):
    """Test 2b: Tampered byte in tokenizer file -> must raise IntegrityError or ValueError."""
    tmp_path, manifest, _, tok_file = dummy_weight_and_tokenizer
    data = bytearray(tok_file.read_bytes())
    data[5] ^= 0xFF
    tok_file.write_bytes(bytes(data))

    with pytest.raises((IntegrityError, ValueError)) as excinfo:
        manifest.validate(check_disk=True)
    assert "tokenizer" in str(excinfo.value).lower() or "mismatch" in str(excinfo.value).lower()


def test_missing_weight_shard_raises_file_not_found(dummy_weight_and_tokenizer):
    """Test 2c: Missing weight file/shard -> must raise FileNotFoundError."""
    tmp_path, manifest, weight_file, _ = dummy_weight_and_tokenizer
    weight_file.unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        manifest.validate(check_disk=True)
    assert "model.safetensors" in str(excinfo.value) or "weight" in str(excinfo.value).lower()


def test_missing_tokenizer_file_raises_file_not_found(dummy_weight_and_tokenizer):
    """Test 2d: Missing tokenizer file -> must raise FileNotFoundError."""
    tmp_path, manifest, _, tok_file = dummy_weight_and_tokenizer
    tok_file.unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        manifest.validate(check_disk=True)
    assert "tokenizer" in str(excinfo.value).lower()


def test_vocab_mismatch_raises_error(dummy_weight_and_tokenizer):
    """Test 2e: Mismatched tokenizer vs manifest vocab size -> must raise ValueError."""
    tmp_path, manifest, _, _ = dummy_weight_and_tokenizer

    class MockTokenizer:
        vocab_size = 32000
        bos_token_id = 2
        eos_token_id = 1

    with pytest.raises((ArchitectureMismatchError, ValueError)) as excinfo:
        load_model(manifest, tokenizer=MockTokenizer(), verify_hashes=False)
    assert "vocab" in str(excinfo.value).lower()


def test_random_weight_rejection_when_pretrained_claimed(dummy_weight_and_tokenizer):
    """Test 2f: is_pretrained=True with random weights -> rejected."""
    tmp_path, manifest, weight_file, _ = dummy_weight_and_tokenizer
    weight_file.write_bytes(b"RANDOM_WEIGHTS_NOT_PRETRAINED")
    with pytest.raises((IntegrityError, ValueError)):
        manifest.validate(check_disk=True)


def test_gemma_2b_manifest_factory():
    """Verify factory for official gemma-2b-it checkpoint metadata."""
    manifest = ModelManifest.gemma_2b_it()
    assert manifest.is_pretrained is True
    assert manifest.random_init is False
    assert manifest.model_id == "google/gemma-2b-it"
    assert manifest.vocabulary_size == 256000
    assert manifest.hidden_dimension == 2048
    assert manifest.num_layers == 18
    assert len(manifest.weight_hash) == 64
    assert len(manifest.tokenizer_hash) == 64


def test_compact_test_manifest_factory():
    """Verify factory for synthetic prlr-compact-testbed."""
    manifest = ModelManifest.compact_test()
    assert manifest.is_pretrained is False
    assert manifest.random_init is True
    assert manifest.model_id == "prlr-compact-testbed"
    assert manifest.vocabulary_size == 1000
    assert manifest.hidden_dimension == 256
    assert manifest.num_layers == 1


def test_invalid_hash_length_bypass_raises_integrity_error(tmp_path: Path):
    """Test regression: weights_sha256="BYPASS" (len != 64) must raise IntegrityError."""
    fake_weights = tmp_path / "weights.safetensors"
    fake_weights.write_bytes(b"FAKE_RANDOM_WEIGHTS_NOT_GEMMA")
    fake_tok = tmp_path / "tokenizer.model"
    fake_tok.write_bytes(b"FAKE_TOKENIZER")
    tok_sha = hashlib.sha256(b"FAKE_TOKENIZER").hexdigest()

    manifest = ModelManifest(
        model_id="test-bypass-model",
        weights_path=str(fake_weights),
        weights_sha256="BYPASS",
        tokenizer_path=str(fake_tok),
        tokenizer_sha256=tok_sha,
        vocabulary_size=1000,
        is_pretrained=True,
        random_init=False,
    )
    with pytest.raises(IntegrityError) as excinfo:
        manifest.validate(check_disk=False)
    assert "length" in str(excinfo.value).lower()

    with pytest.raises(IntegrityError) as excinfo:
        manifest.validate(check_disk=True)
    assert "length" in str(excinfo.value).lower()

    with pytest.raises(IntegrityError):
        load_model(manifest, verify_hashes=True)


def test_invalid_manifest_state_neither_pretrained_nor_random():
    """Test regression: manifest with neither is_pretrained nor random_init must raise ManifestError."""
    manifest = ModelManifest(
        model_id="test-invalid-init",
        is_pretrained=False,
        random_init=False,
    )
    with pytest.raises(ManifestError) as excinfo:
        manifest.validate()
    assert "must specify either is_pretrained=True or random_init=True" in str(excinfo.value)
