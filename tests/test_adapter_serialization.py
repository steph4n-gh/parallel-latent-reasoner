"""Tests for PRLR Adapter Weight Serialization and Deserialization.

Verifies:
- Extraction of trainable adapter parameters (Prelude, AdaRMSNorm, ReZero alphas, Coda)
- Exclusion of frozen base model backbone weights
- Freezing base model parameters
- Saving and loading adapter weights to/from .npz and .safetensors formats
- Weight fidelity and restoration across model instances
- Error handling for invalid paths
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXCompactGemmaModel


@pytest.fixture
def compact_model() -> MLXCompactGemmaModel:
    """Create a compact test model instance."""
    config = GemmaLatentConfig(
        dim=64,
        intermediate_dim=128,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
        num_layers=2,
        num_memory_slots=4,
        vocab_size=120,
        deliberation_steps=4,
    )
    return MLXCompactGemmaModel(config)


def test_get_trainable_parameters_keys(compact_model: MLXCompactGemmaModel):
    """Verify that get_trainable_parameters returns exactly the expected adapter parameter keys."""
    params = compact_model.get_trainable_parameters()
    assert isinstance(params, dict)

    # Prelude keys
    assert "prelude.slot_embeddings" in params
    assert "prelude.context_proj.weight" in params
    assert "prelude.norm.weight" in params

    # Layer keys for all layers
    for i in range(compact_model.config.num_layers):
        assert f"engine.layers.{i}.norm1.weight" in params
        assert f"engine.layers.{i}.norm1.mlp_l1.weight" in params
        assert f"engine.layers.{i}.norm1.mlp_l1.bias" in params
        assert f"engine.layers.{i}.norm1.mlp_l2.weight" in params
        assert f"engine.layers.{i}.norm1.mlp_l2.bias" in params

        assert f"engine.layers.{i}.norm2.weight" in params
        assert f"engine.layers.{i}.norm2.mlp_l1.weight" in params
        assert f"engine.layers.{i}.norm2.mlp_l1.bias" in params
        assert f"engine.layers.{i}.norm2.mlp_l2.weight" in params
        assert f"engine.layers.{i}.norm2.mlp_l2.bias" in params

        assert f"engine.layers.{i}.alpha_attn" in params
        assert f"engine.layers.{i}.alpha_mlp" in params

    # Coda keys
    assert "coda.final_norm.weight" in params
    assert "coda.readout_proj.weight" in params

    # Base model parameters must NOT be present
    for k in params:
        assert "attn.q_proj" not in k
        assert "attn.k_proj" not in k
        assert "attn.v_proj" not in k
        assert "attn.o_proj" not in k
        assert "mlp.gate_proj" not in k
        assert "mlp.up_proj" not in k
        assert "mlp.down_proj" not in k
        assert "embed_tokens" not in k


def test_freeze_base_model(compact_model: MLXCompactGemmaModel):
    """Verify freeze_base_model leaves only adapter parameters trainable."""
    from mlx.utils import tree_flatten

    compact_model.freeze_base_model()
    trainable = dict(tree_flatten(compact_model.trainable_parameters()))

    # Verify no base attention or base MLP parameters in trainable parameters
    for k in trainable:
        assert "attn." not in k
        assert "mlp.gate_proj" not in k
        assert "mlp.up_proj" not in k
        assert "mlp.down_proj" not in k
        assert "embed_tokens" not in k

    # Verify adapter parameters are trainable
    assert "prelude.slot_embeddings" in trainable
    assert "prelude.context_proj.weight" in trainable
    assert "coda.readout_proj.weight" in trainable
    assert "engine.layers.0.norm1.mlp_l1.weight" in trainable


def test_save_and_load_npz(compact_model: MLXCompactGemmaModel):
    """Verify save_adapter_weights and load_adapter_weights with .npz format."""
    # Perturb adapter weights
    compact_model.prelude.slot_embeddings = mx.ones((1, 4, 64)) * 3.14159
    compact_model.engine.layers[0].alpha_attn = mx.array([0.042])
    compact_model.coda.final_norm.weight = mx.ones((64,)) * 2.718

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "adapter.npz"
        compact_model.save_adapter_weights(save_path)
        assert save_path.exists()

        # Create new model and verify weights are different initially
        new_model = MLXCompactGemmaModel(compact_model.config)
        assert not mx.allclose(
            new_model.prelude.slot_embeddings, compact_model.prelude.slot_embeddings
        )

        # Load weights into new model
        loaded = new_model.load_adapter_weights(save_path)
        assert isinstance(loaded, dict)
        assert len(loaded) == len(compact_model.get_trainable_parameters())

        # Verify exact equality
        assert mx.allclose(
            new_model.prelude.slot_embeddings, compact_model.prelude.slot_embeddings
        )
        assert mx.allclose(
            new_model.engine.layers[0].alpha_attn,
            compact_model.engine.layers[0].alpha_attn,
        )
        assert mx.allclose(
            new_model.coda.final_norm.weight, compact_model.coda.final_norm.weight
        )


def test_save_and_load_safetensors(compact_model: MLXCompactGemmaModel):
    """Verify save_adapter_weights and load_adapter_weights with .safetensors format."""
    compact_model.prelude.context_proj.weight = (
        mx.random.normal((64, 64)) * 0.5
    )
    compact_model.engine.layers[1].norm2.mlp_l1.weight = (
        mx.random.normal((64, 128)) * 0.2
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "adapter.safetensors"
        compact_model.save_adapter_weights(save_path)
        assert save_path.exists()

        new_model = MLXCompactGemmaModel(compact_model.config)
        new_model.load_adapter_weights(save_path)

        assert mx.allclose(
            new_model.prelude.context_proj.weight,
            compact_model.prelude.context_proj.weight,
        )
        assert mx.allclose(
            new_model.engine.layers[1].norm2.mlp_l1.weight,
            compact_model.engine.layers[1].norm2.mlp_l1.weight,
        )


def test_load_nonexistent_file_raises(compact_model: MLXCompactGemmaModel):
    """Verify load_adapter_weights raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        compact_model.load_adapter_weights("/tmp/nonexistent_adapter_file_12345.npz")
