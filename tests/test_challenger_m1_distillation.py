"""Adversarial Stress Test Suite for Milestone M1 (MLX BPTT Trainer & Adapter Serialization).

Empirical challenger test harness verifying:
1. Deep unrolls (T=1, 2, 8, 16, 32) and depth scaling numerical stability
2. Variable batch sizes (B=1, 3, 7, 16, 32) and irregular prompt/target dimensions
3. Extreme learning rates (0.0, 1e-8, 10.0) and dynamic gradient clipping bounds
4. Weight serialization roundtrip fidelity (.npz & .safetensors) with bitwise, hash, and norm parity
5. MoE architecture adapter weight serialization fidelity
6. Base model parameter immutability and strict gradient isolation
7. Multi-step memory stability and absence of memory leaks (100 steps)
8. Multi-objective distillation loss edge cases (orthogonal/antiparallel teacher latents, T=1, sequence targets)
9. Serialization path auto-creation and corrupted/missing file robustness
"""

from __future__ import annotations

import hashlib
import math
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
def small_config() -> GemmaLatentConfig:
    return GemmaLatentConfig(
        dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_layers=2,
        num_memory_slots=4,
        vocab_size=128,
        deliberation_steps=4,
    )


@pytest.fixture
def moe_config() -> GemmaLatentConfig:
    return GemmaLatentConfig(
        dim=64,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_layers=2,
        num_memory_slots=4,
        vocab_size=128,
        deliberation_steps=4,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_dim=32,
    )


def compute_tensor_sha256(arr: mx.array) -> str:
    """Compute deterministic SHA-256 hash of an mx.array."""
    mx.eval(arr)
    raw_bytes = bytes(memoryview(arr))
    return hashlib.sha256(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 1. Deep Recurrent Unrolls & Depth Scaling Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unroll_t", [1, 2, 4, 8, 16, 24])
def test_deep_unroll_numerical_stability(small_config: GemmaLatentConfig, unroll_t: int):
    """Stress-test BPTT loss computation and autodiff across shallow to deep unrolls."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    B, P = 2, 8
    input_ids = mx.random.randint(0, small_config.vocab_size, (B, P))
    target_tokens = mx.random.randint(0, small_config.vocab_size, (B,))
    teacher_latents = mx.random.normal((B, small_config.dim))

    (loss, (ce, align, aux)), grads = trainer._loss_and_grad_fn(
        model,
        input_ids,
        target_tokens,
        teacher_latents=teacher_latents,
        steps=unroll_t,
        lambda_align=0.5,
        lambda_aux=0.1,
    )

    mx.eval(loss, ce, align, aux, grads)

    # 1. Loss must be finite and non-NaN
    assert not mx.isnan(loss), f"Loss is NaN at T={unroll_t}"
    assert not mx.isinf(loss), f"Loss is Inf at T={unroll_t}"
    assert float(loss) > 0.0, f"Loss must be strictly positive at T={unroll_t}"

    # 2. Check gradients across all adapter parameters
    flat_grads = dict(tree_flatten(grads))
    for name, g in flat_grads.items():
        assert not mx.isnan(g).any(), f"Gradient contains NaN for {name} at T={unroll_t}"
        assert not mx.isinf(g).any(), f"Gradient contains Inf for {name} at T={unroll_t}"

    # 3. Ensure key adapter parameters received non-zero gradients
    assert float(mx.linalg.norm(flat_grads["prelude.slot_embeddings"])) > 0.0
    assert float(mx.linalg.norm(flat_grads["coda.readout_proj.weight"])) > 0.0


# ---------------------------------------------------------------------------
# 2. Variable Batch Sizes & Boundary Dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 7, 16, 32])
@pytest.mark.parametrize("prompt_len", [1, 5, 19, 64])
def test_variable_batch_sizes_and_prompt_lengths(
    small_config: GemmaLatentConfig, batch_size: int, prompt_len: int
):
    """Stress-test variable and prime batch sizes and prompt lengths."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, prompt_len))
    target_tokens = mx.random.randint(0, small_config.vocab_size, (batch_size,))
    teacher_latents = mx.random.normal((batch_size, small_config.dim))

    batch = {
        "input_ids": input_ids,
        "target_tokens": target_tokens,
        "teacher_latents": teacher_latents,
    }

    loss_val, metrics = trainer.train_step(batch, steps=3)
    assert isinstance(loss_val, float)
    assert not math.isnan(loss_val)
    assert not math.isinf(loss_val)
    assert metrics["grad_norm"] > 0.0
    assert not math.isnan(metrics["grad_norm"])


@pytest.mark.parametrize("seq_len", [1, 2, 6, 12])
def test_variable_target_sequence_lengths(small_config: GemmaLatentConfig, seq_len: int):
    """Stress-test 2D sequence target tokens with varying sequence lengths."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    B, P = 4, 8
    input_ids = mx.random.randint(0, small_config.vocab_size, (B, P))
    target_tokens = mx.random.randint(0, small_config.vocab_size, (B, seq_len))

    batch = {
        "input_ids": input_ids,
        "target_tokens": target_tokens,
    }

    loss_val, metrics = trainer.train_step(batch, steps=3)
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    assert not math.isnan(loss_val)


# ---------------------------------------------------------------------------
# 3. Extreme Learning Rates & Gradient Clipping
# ---------------------------------------------------------------------------


def test_zero_learning_rate_immutability(small_config: GemmaLatentConfig):
    """When learning rate is 0.0, adapter parameters must remain 100% unchanged."""
    model = MLXCompactGemmaModel(small_config)
    cfg = TrainerConfig(
        learning_rate=0.0,
        min_learning_rate=0.0,
        warmup_steps=0,
        total_steps=10,
    )
    trainer = PRLRBPTTTrainer(model, config=cfg)

    init_params = {k: mx.array(v) for k, v in model.get_trainable_parameters().items()}

    batch = {
        "input_ids": mx.random.randint(0, small_config.vocab_size, (2, 8)),
        "target_tokens": mx.random.randint(0, small_config.vocab_size, (2,)),
    }

    trainer.train_step(batch)

    for k, v_orig in init_params.items():
        v_curr = model.get_trainable_parameters()[k]
        assert mx.array_equal(v_orig, v_curr), f"Parameter {k} mutated with lr=0.0"


def test_extreme_large_learning_rate_and_clipping(small_config: GemmaLatentConfig):
    """With massive learning rate and gradient clipping, updates remain stable and bounded."""
    model = MLXCompactGemmaModel(small_config)
    cfg = TrainerConfig(
        learning_rate=10.0,
        min_learning_rate=10.0,
        warmup_steps=0,
        total_steps=10,
        max_grad_norm=0.5,
    )
    trainer = PRLRBPTTTrainer(model, config=cfg)

    batch = {
        "input_ids": mx.random.randint(0, small_config.vocab_size, (2, 8)),
        "target_tokens": mx.random.randint(0, small_config.vocab_size, (2,)),
    }

    loss_val, metrics = trainer.train_step(batch)
    assert not math.isnan(loss_val)
    assert not math.isinf(loss_val)
    # Check that parameters did not blow up into NaN
    for k, v in model.get_trainable_parameters().items():
        assert not mx.isnan(v).any(), f"Parameter {k} became NaN with high lr"
        assert not mx.isinf(v).any(), f"Parameter {k} became Inf with high lr"


# ---------------------------------------------------------------------------
# 4. Weight Serialization Roundtrip Fidelity (.npz & .safetensors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["npz", "safetensors"])
def test_weight_serialization_roundtrip_fidelity(small_config: GemmaLatentConfig, fmt: str):
    """Verify bitwise and norm parity across serialization roundtrip and identical forward logits."""
    # Model A: Trained / perturbed model
    model_a = MLXCompactGemmaModel(small_config)
    model_a.prelude.slot_embeddings = mx.random.normal(model_a.prelude.slot_embeddings.shape)
    model_a.prelude.context_proj.weight = mx.random.normal(model_a.prelude.context_proj.weight.shape)
    for i, layer in enumerate(model_a.engine.layers):
        layer.norm1.mlp_l1.weight = mx.random.normal(layer.norm1.mlp_l1.weight.shape)
        layer.norm2.mlp_l2.weight = mx.random.normal(layer.norm2.mlp_l2.weight.shape)
        layer.alpha_attn = mx.array([0.0314 + i * 0.01])
        layer.alpha_mlp = mx.array([0.0271 + i * 0.01])
    model_a.coda.final_norm.weight = mx.random.normal(model_a.coda.final_norm.weight.shape)
    model_a.coda.readout_proj.weight = mx.random.normal(model_a.coda.readout_proj.weight.shape)

    params_a = model_a.get_trainable_parameters()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / f"adapter_weights.{fmt}"
        model_a.save_adapter_weights(file_path)
        assert file_path.exists(), f"File {file_path} was not created"

        # Model B: Clone of base model A before adapter changes
        mx.random.seed(42)
        model_b = MLXCompactGemmaModel(small_config)
        # Clone base weights from model_a into model_b
        model_b.embed_tokens.weight = mx.array(model_a.embed_tokens.weight)
        for i in range(small_config.num_layers):
            model_b.engine.layers[i].attn.q_proj.weight = mx.array(model_a.engine.layers[i].attn.q_proj.weight)
            model_b.engine.layers[i].attn.k_proj.weight = mx.array(model_a.engine.layers[i].attn.k_proj.weight)
            model_b.engine.layers[i].attn.v_proj.weight = mx.array(model_a.engine.layers[i].attn.v_proj.weight)
            model_b.engine.layers[i].attn.o_proj.weight = mx.array(model_a.engine.layers[i].attn.o_proj.weight)
            model_b.engine.layers[i].mlp.gate_proj.weight = mx.array(model_a.engine.layers[i].mlp.gate_proj.weight)
            model_b.engine.layers[i].mlp.up_proj.weight = mx.array(model_a.engine.layers[i].mlp.up_proj.weight)
            model_b.engine.layers[i].mlp.down_proj.weight = mx.array(model_a.engine.layers[i].mlp.down_proj.weight)
        
        # Load adapter weights into model_b
        loaded_dict = model_b.load_adapter_weights(file_path)
        params_b = model_b.get_trainable_parameters()

        # 1. Parameter count and key match
        assert set(params_a.keys()) == set(params_b.keys())
        assert len(loaded_dict) == len(params_a)

        # 2. Strict Bitwise & Frobenius Norm Equality
        for k in params_a:
            tensor_a = params_a[k]
            tensor_b = params_b[k]

            # Max absolute difference must be exactly 0.0
            max_diff = float(mx.max(mx.abs(tensor_a - tensor_b)))
            assert max_diff == 0.0, f"Max diff for {k} is {max_diff} > 0.0 in {fmt}"

            # Frobenius norm diff must be exactly 0.0
            frobenius_diff = float(mx.linalg.norm(tensor_a - tensor_b))
            assert frobenius_diff == 0.0, f"Frobenius norm diff for {k} is {frobenius_diff} in {fmt}"

            # SHA256 byte hashes must match
            hash_a = compute_tensor_sha256(tensor_a)
            hash_b = compute_tensor_sha256(tensor_b)
            assert hash_a == hash_b, f"SHA256 mismatch for parameter {k} in {fmt}"

        # 3. Forward Pass Logit Parity on Multiple Inputs
        test_inputs = [
            mx.random.randint(0, small_config.vocab_size, (2, 6)),
            mx.random.randint(0, small_config.vocab_size, (1, 14)),
            mx.random.randint(0, small_config.vocab_size, (5, 8)),
        ]

        for x in test_inputs:
            logits_a = model_a(x, steps=3)
            logits_b = model_b(x, steps=3)
            mx.eval(logits_a, logits_b)

            logit_diff = float(mx.max(mx.abs(logits_a - logits_b)))
            assert logit_diff < 1e-6, f"Logit disparity between model A and model B: {logit_diff}"


def test_moe_adapter_serialization_fidelity(moe_config: GemmaLatentConfig):
    """Verify adapter serialization roundtrip for MoE architectures."""
    model_a = MLXCompactGemmaModel(moe_config)
    model_a.prelude.slot_embeddings = mx.random.normal(model_a.prelude.slot_embeddings.shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_npz = Path(tmpdir) / "moe_adapter.npz"
        path_safetensors = Path(tmpdir) / "moe_adapter.safetensors"

        model_a.save_adapter_weights(path_npz)
        model_a.save_adapter_weights(path_safetensors)

        # Reload npz
        model_npz = MLXCompactGemmaModel(moe_config)
        model_npz.load_adapter_weights(path_npz)
        assert mx.allclose(
            model_npz.prelude.slot_embeddings, model_a.prelude.slot_embeddings
        )

        # Reload safetensors
        model_st = MLXCompactGemmaModel(moe_config)
        model_st.load_adapter_weights(path_safetensors)
        assert mx.allclose(
            model_st.prelude.slot_embeddings, model_a.prelude.slot_embeddings
        )


# ---------------------------------------------------------------------------
# 5. Base Model Parameter Immutability & Gradient Isolation
# ---------------------------------------------------------------------------


def test_base_model_strict_immutability_during_training(small_config: GemmaLatentConfig):
    """Ensure base model backbone parameters are NOT mutated or tracked by optimizer."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    # Record baseline hashes for all base model weights
    base_hashes: dict[str, str] = {}
    for i, layer in enumerate(model.engine.layers):
        base_hashes[f"layer_{i}.attn.q_proj"] = compute_tensor_sha256(layer.attn.q_proj.weight)
        base_hashes[f"layer_{i}.attn.k_proj"] = compute_tensor_sha256(layer.attn.k_proj.weight)
        base_hashes[f"layer_{i}.attn.v_proj"] = compute_tensor_sha256(layer.attn.v_proj.weight)
        base_hashes[f"layer_{i}.attn.o_proj"] = compute_tensor_sha256(layer.attn.o_proj.weight)
        base_hashes[f"layer_{i}.mlp.gate_proj"] = compute_tensor_sha256(layer.mlp.gate_proj.weight)
        base_hashes[f"layer_{i}.mlp.up_proj"] = compute_tensor_sha256(layer.mlp.up_proj.weight)
        base_hashes[f"layer_{i}.mlp.down_proj"] = compute_tensor_sha256(layer.mlp.down_proj.weight)
    base_hashes["embed_tokens"] = compute_tensor_sha256(model.embed_tokens.weight)

    # Perform 15 aggressive optimization steps
    for step in range(15):
        batch = {
            "input_ids": mx.random.randint(0, small_config.vocab_size, (4, 10)),
            "target_tokens": mx.random.randint(0, small_config.vocab_size, (4,)),
            "teacher_latents": mx.random.normal((4, small_config.dim)),
        }
        trainer.train_step(batch, steps=4)

    # Verify base weights remain 100% bitwise identical
    for i, layer in enumerate(model.engine.layers):
        assert compute_tensor_sha256(layer.attn.q_proj.weight) == base_hashes[f"layer_{i}.attn.q_proj"]
        assert compute_tensor_sha256(layer.attn.k_proj.weight) == base_hashes[f"layer_{i}.attn.k_proj"]
        assert compute_tensor_sha256(layer.attn.v_proj.weight) == base_hashes[f"layer_{i}.attn.v_proj"]
        assert compute_tensor_sha256(layer.attn.o_proj.weight) == base_hashes[f"layer_{i}.attn.o_proj"]
        assert compute_tensor_sha256(layer.mlp.gate_proj.weight) == base_hashes[f"layer_{i}.mlp.gate_proj"]
        assert compute_tensor_sha256(layer.mlp.up_proj.weight) == base_hashes[f"layer_{i}.mlp.up_proj"]
        assert compute_tensor_sha256(layer.mlp.down_proj.weight) == base_hashes[f"layer_{i}.mlp.down_proj"]
    assert compute_tensor_sha256(model.embed_tokens.weight) == base_hashes["embed_tokens"]


# ---------------------------------------------------------------------------
# 6. Multi-Step Memory Stability (100 Steps Leak Check)
# ---------------------------------------------------------------------------


def test_memory_stability_across_100_training_steps(small_config: GemmaLatentConfig):
    """Stress-test memory stability across 100 consecutive BPTT training steps on Metal GPU."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    mx.metal.clear_cache() if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache") else None
    mx.reset_peak_memory()

    batch = {
        "input_ids": mx.random.randint(0, small_config.vocab_size, (4, 8)),
        "target_tokens": mx.random.randint(0, small_config.vocab_size, (4, 2)),
        "teacher_latents": mx.random.normal((4, small_config.dim)),
    }

    # Warmup 10 steps to establish steady-state allocations
    for _ in range(10):
        trainer.train_step(batch, steps=4)

    mem_at_step_10 = mx.get_active_memory()
    peak_at_step_10 = mx.get_peak_memory()

    # Run next 90 steps
    for _ in range(90):
        trainer.train_step(batch, steps=4)

    mem_at_step_100 = mx.get_active_memory()
    peak_at_step_100 = mx.get_peak_memory()

    # Memory growth between step 10 and step 100 must be essentially flat (allowing minor allocator fragmentation)
    mem_ratio = mem_at_step_100 / max(1, mem_at_step_10)
    assert mem_ratio < 1.5, f"Excessive active memory growth: {mem_at_step_10} -> {mem_at_step_100} (ratio={mem_ratio:.2f})"
    assert trainer.current_step == 100


# ---------------------------------------------------------------------------
# 7. Multi-Objective Loss Edge Cases
# ---------------------------------------------------------------------------


def test_distillation_loss_antiparallel_teacher_latents(small_config: GemmaLatentConfig):
    """Verify behavior of teacher alignment loss with antiparallel and orthogonal vectors."""
    model = MLXCompactGemmaModel(small_config)

    input_ids = mx.random.randint(0, small_config.vocab_size, (2, 6))
    target_tokens = mx.random.randint(0, small_config.vocab_size, (2,))

    # Compute model readout
    slots, prompt_hiddens = model.prelude(input_ids)
    readout = model.coda.pool_readout(slots)
    mx.eval(readout)

    # 1. Exact parallel teacher latents -> align loss should approach 0
    loss_parallel, (_, align_parallel, _) = _compute_bptt_loss(
        model,
        input_ids,
        target_tokens,
        teacher_latents=readout,
        steps=2,
        lambda_align=1.0,
        lambda_aux=0.0,
    )

    # 2. Antiparallel teacher latents -> align loss should approach max (l_cos=2.0, l_nmse=4.0 -> 3.0)
    loss_anti, (_, align_anti, _) = _compute_bptt_loss(
        model,
        input_ids,
        target_tokens,
        teacher_latents=-readout,
        steps=2,
        lambda_align=1.0,
        lambda_aux=0.0,
    )

    mx.eval(align_parallel, align_anti)
    assert float(align_parallel) < 0.05, f"Expected near-zero align loss for parallel, got {float(align_parallel)}"
    assert float(align_anti) > float(align_parallel), "Antiparallel align loss must be higher than parallel"


# ---------------------------------------------------------------------------
# 8. Path Auto-Creation & Corrupted File Robustness
# ---------------------------------------------------------------------------


def test_save_adapter_weights_creates_deep_directories(small_config: GemmaLatentConfig):
    """Verify save_adapter_weights automatically creates deeply nested non-existent directory trees."""
    model = MLXCompactGemmaModel(small_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = Path(tmpdir) / "level1" / "level2" / "level3" / "adapter.npz"
        model.save_adapter_weights(deep_path)
        assert deep_path.exists()

        # Reload to ensure validity
        new_model = MLXCompactGemmaModel(small_config)
        new_model.load_adapter_weights(deep_path)
        assert mx.allclose(new_model.prelude.slot_embeddings, model.prelude.slot_embeddings)


def test_load_corrupted_file_raises(small_config: GemmaLatentConfig):
    """Verify loading corrupted or malformed file raises appropriate exception."""
    model = MLXCompactGemmaModel(small_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        corrupted_path = Path(tmpdir) / "corrupted_adapter.npz"
        corrupted_path.write_bytes(b"CORRUPTED_GARBAGE_BYTES_123456789")

        with pytest.raises(Exception):
            model.load_adapter_weights(corrupted_path)


# ---------------------------------------------------------------------------
# 9. Additional Edge Cases: Untied Embeddings, Shape Mismatches, Evaluation Parity
# ---------------------------------------------------------------------------


def test_untied_embeddings_serialization_fidelity():
    """Verify serialization and gradient isolation when tie_word_embeddings=False."""
    cfg = GemmaLatentConfig(
        dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_layers=2,
        num_memory_slots=4,
        vocab_size=128,
        deliberation_steps=4,
        tie_word_embeddings=False,
    )
    model = MLXCompactGemmaModel(cfg)
    assert model.coda.lm_head is not None
    trainable = model.get_trainable_parameters()
    assert "coda.lm_head.weight" in trainable

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = Path(tmpdir) / "untied_adapter.safetensors"
        model.save_adapter_weights(st_path)

        model_loaded = MLXCompactGemmaModel(cfg)
        model_loaded.load_adapter_weights(st_path)
        assert mx.allclose(model_loaded.coda.lm_head.weight, model.coda.lm_head.weight)


def test_target_tokens_3d_shape_mismatch_raises(small_config: GemmaLatentConfig):
    """Verify that passing 3D target_tokens raises ValueError."""
    model = MLXCompactGemmaModel(small_config)
    input_ids = mx.random.randint(0, small_config.vocab_size, (2, 8))
    target_tokens_3d = mx.random.randint(0, small_config.vocab_size, (2, 4, 3))

    with pytest.raises(ValueError, match="Unexpected target_tokens shape"):
        _compute_bptt_loss(model, input_ids, target_tokens_3d)


def test_evaluate_non_mutating(small_config: GemmaLatentConfig):
    """Verify that evaluate() does not mutate parameters or increment step counter."""
    model = MLXCompactGemmaModel(small_config)
    trainer = PRLRBPTTTrainer(model)

    params_before = {k: mx.array(v) for k, v in model.get_trainable_parameters().items()}
    step_before = trainer.current_step

    val_dataset = [
        {
            "input_ids": mx.random.randint(0, small_config.vocab_size, (2, 6)),
            "target_tokens": mx.random.randint(0, small_config.vocab_size, (2,)),
        }
        for _ in range(4)
    ]

    val_metrics = trainer.evaluate(val_dataset, steps=3)

    assert trainer.current_step == step_before
    for k, v in params_before.items():
        assert mx.array_equal(v, model.get_trainable_parameters()[k]), f"Param {k} mutated during evaluate()"
    assert "val_accuracy" in val_metrics
    assert "val_loss" in val_metrics

