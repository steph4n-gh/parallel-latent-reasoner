"""Comprehensive Unit Tests for PRLR BPTT Latent Distillation Engine (trainer.py).

Verifies:
- TrainerConfig creation and defaults
- Model parameter freezing and adapter unfreezing upon trainer initialization
- Differentiable Jacobi unroll graph through recurrent steps (T=2..8)
- Multi-objective loss formulation (CE + Teacher Latent Alignment + Contractive Velocity)
- Gradient isolation: gradients computed strictly for adapter parameters
- Single step optimization (train_step) with dict and tuple batches
- Multi-step convergence and loss reduction
- Epoch training (train_epoch) and evaluation (evaluate)
- Dynamic gradient norm clipping
- Checkpoint saving and reloading
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.trainer import (
    PRLRBPTTTrainer,
    TrainerConfig,
    _compute_bptt_loss,
)


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


@pytest.fixture
def trainer_config() -> TrainerConfig:
    """Create a test trainer config."""
    return TrainerConfig(
        learning_rate=2e-3,
        min_learning_rate=1e-5,
        warmup_steps=5,
        total_steps=50,
        deliberation_steps=3,
        lambda_align=0.5,
        lambda_aux=0.1,
        max_grad_norm=1.0,
        save_every_epochs=0,
    )


def test_trainer_config_defaults():
    """Verify TrainerConfig default parameters."""
    cfg = TrainerConfig()
    assert cfg.learning_rate == 1e-3
    assert cfg.min_learning_rate == 1e-5
    assert cfg.deliberation_steps == 4
    assert cfg.lambda_align == 0.5
    assert cfg.lambda_aux == 0.1
    assert cfg.rho_contractive == 0.85
    assert cfg.max_grad_norm == 1.0


def test_trainer_initialization_freezes_base_weights(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify that PRLRBPTTTrainer freezes base weights and keeps only adapter weights trainable."""
    trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

    trainable = dict(tree_flatten(compact_model.trainable_parameters()))

    # Ensure base weights are frozen
    for k in trainable:
        assert "attn.q_proj" not in k
        assert "attn.k_proj" not in k
        assert "attn.v_proj" not in k
        assert "attn.o_proj" not in k
        assert "mlp.gate_proj" not in k
        assert "mlp.up_proj" not in k
        assert "mlp.down_proj" not in k
        assert "embed_tokens" not in k

    # Ensure adapter weights are present
    assert "prelude.slot_embeddings" in trainable
    assert "coda.readout_proj.weight" in trainable
    assert "engine.layers.0.norm1.mlp_l1.weight" in trainable
    assert "engine.layers.0.alpha_attn" in trainable


def test_compute_bptt_loss_1d_target(compact_model: MLXCompactGemmaModel):
    """Verify BPTT loss computation with 1D target tokens."""
    B, P = 2, 6
    input_ids = mx.random.randint(0, 100, (B, P))
    target_tokens = mx.random.randint(0, 100, (B,))
    teacher_latents = mx.random.normal((B, 64))

    loss, (ce, align, aux) = _compute_bptt_loss(
        compact_model,
        input_ids,
        target_tokens,
        teacher_latents=teacher_latents,
        steps=3,
        lambda_align=0.5,
        lambda_aux=0.1,
    )

    assert loss.ndim == 0
    assert float(loss) > 0.0
    assert float(ce) > 0.0
    assert float(align) >= 0.0
    assert float(aux) >= 0.0


def test_compute_bptt_loss_2d_target(compact_model: MLXCompactGemmaModel):
    """Verify BPTT loss computation with multi-token 2D target tokens."""
    B, P, S = 2, 6, 4
    input_ids = mx.random.randint(0, 100, (B, P))
    target_tokens = mx.random.randint(0, 100, (B, S))

    loss, (ce, align, aux) = _compute_bptt_loss(
        compact_model,
        input_ids,
        target_tokens,
        teacher_latents=None,
        steps=4,
        lambda_align=0.5,
        lambda_aux=0.1,
    )

    assert loss.ndim == 0
    assert float(loss) > 0.0
    assert float(align) == 0.0  # Teacher latents were None


def test_bptt_gradients_only_for_adapters(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify that gradients flow strictly to adapter parameters across unrolls."""
    trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

    B, P = 2, 8
    input_ids = mx.random.randint(0, 100, (B, P))
    target_tokens = mx.random.randint(0, 100, (B,))
    teacher_latents = mx.random.normal((B, 64))

    (loss, _), grads = trainer._loss_and_grad_fn(
        compact_model,
        input_ids,
        target_tokens,
        teacher_latents=teacher_latents,
        steps=3,
    )

    flat_grads = dict(tree_flatten(grads))

    # Base weights must NOT have gradients
    for k in flat_grads:
        assert "attn.q_proj" not in k
        assert "mlp.gate_proj" not in k
        assert "embed_tokens" not in k

    # Adapter weights must have gradients
    assert "prelude.slot_embeddings" in flat_grads
    assert "prelude.context_proj.weight" in flat_grads
    assert "coda.readout_proj.weight" in flat_grads
    assert "engine.layers.0.norm1.mlp_l1.weight" in flat_grads


def test_train_step_with_dict_batch(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify train_step execution with dictionary batch format."""
    trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

    batch = {
        "input_ids": mx.random.randint(0, 100, (2, 6)),
        "target_tokens": mx.random.randint(0, 100, (2, 2)),
        "teacher_latents": mx.random.normal((2, 64)),
    }

    initial_slot_emb = mx.array(compact_model.prelude.slot_embeddings)

    loss_val, metrics = trainer.train_step(batch)

    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    assert metrics["step"] == 1
    assert "ce_loss" in metrics
    assert "align_loss" in metrics
    assert "aux_loss" in metrics
    assert "grad_norm" in metrics
    assert "learning_rate" in metrics

    # Verify adapter parameters updated
    assert not mx.allclose(compact_model.prelude.slot_embeddings, initial_slot_emb)


def test_train_step_with_tuple_batch(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify train_step execution with tuple batch format."""
    trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

    batch = (
        mx.random.randint(0, 100, (2, 6)),
        mx.random.randint(0, 100, (2,)),
    )

    loss_val, metrics = trainer.train_step(batch)
    assert isinstance(loss_val, float)
    assert metrics["step"] == 1


def test_training_convergence_over_steps(compact_model: MLXCompactGemmaModel):
    """Verify that repeated BPTT training steps reduce loss on a fixed target batch."""
    cfg = TrainerConfig(
        learning_rate=5e-3,
        min_learning_rate=1e-3,
        warmup_steps=2,
        total_steps=25,
        deliberation_steps=2,
        lambda_align=0.5,
        lambda_aux=0.05,
    )
    trainer = PRLRBPTTTrainer(compact_model, config=cfg)

    # Fixed training sample
    mx.random.seed(42)
    batch = {
        "input_ids": mx.random.randint(0, 100, (2, 6)),
        "target_tokens": mx.array([[10, 20], [30, 40]]),
        "teacher_latents": mx.random.normal((2, 64)),
    }

    initial_loss, _ = trainer.train_step(batch)
    for _ in range(15):
        trainer.train_step(batch)
    final_loss, _ = trainer.train_step(batch)

    # Loss must demonstrably decrease
    assert final_loss < initial_loss, f"Expected final loss ({final_loss}) < initial loss ({initial_loss})"


def test_train_epoch_and_evaluate(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify train_epoch and evaluate methods across dataset batches."""
    trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

    dataset = [
        {
            "input_ids": mx.random.randint(0, 100, (2, 6)),
            "target_tokens": mx.random.randint(0, 100, (2,)),
            "teacher_latents": mx.random.normal((2, 64)),
        }
        for _ in range(3)
    ]

    epoch_metrics = trainer.train_epoch(dataset)
    assert epoch_metrics["epoch"] == 1.0
    assert epoch_metrics["step"] == 3.0
    assert epoch_metrics["loss"] > 0.0

    val_metrics = trainer.evaluate(dataset)
    assert "val_loss" in val_metrics
    assert "val_ce_loss" in val_metrics
    assert "val_accuracy" in val_metrics
    assert 0.0 <= val_metrics["val_accuracy"] <= 1.0


def test_checkpoint_saving_and_loading(compact_model: MLXCompactGemmaModel, trainer_config: TrainerConfig):
    """Verify save_checkpoint and load_checkpoint functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config.checkpoint_dir = tmpdir
        trainer = PRLRBPTTTrainer(compact_model, config=trainer_config)

        # Perturb a parameter
        compact_model.coda.final_norm.weight = mx.ones((64,)) * 8.88

        saved_path = trainer.save_checkpoint()
        assert saved_path.exists()

        # Reset model
        new_model = MLXCompactGemmaModel(compact_model.config)
        new_trainer = PRLRBPTTTrainer(new_model, config=trainer_config)
        new_trainer.load_checkpoint(saved_path)

        assert mx.allclose(new_model.coda.final_norm.weight, compact_model.coda.final_norm.weight)
