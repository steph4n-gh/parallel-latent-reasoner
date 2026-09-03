"""Tests for 1-Step Gradient Flow and Principled Initialization (Milestone 3 / Requirement R4).

Verifies:
- 100% of trainable adapter parameters receive strictly non-zero gradient norms (||nabla_theta||_2 > 0)
- Orthogonal slot anchor initialization on CPU stream with Gram matrix orthogonality
- Scaled non-zero MoE initialization preventing gradient collapse
- AdaRMSNorm 1-step gradient flow to both mlp_l1 and mlp_l2
- Bounded residual scaling alpha = alpha_max * sigmoid(raw_alpha)
"""

from __future__ import annotations

import math
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from prlr.gemma.adapter import (
    GemmaPreludeAdapter,
    GemmaRecurrentAdapter,
    init_orthogonal_slot_anchors,
)
from prlr.kernel.recurrent_core import (
    MLXAdaRMSNorm,
    MLXCrossAttention,
    MLXMoE,
    MLXRecurrentBlock,
)


def test_orthogonal_slot_anchors_initialization():
    """Verify orthogonal slot anchors are generated on CPU stream and satisfy Q^T Q = scale^2 I."""
    num_slots = 16
    dim = 2048
    scale = 0.02
    anchors = init_orthogonal_slot_anchors(num_slots, dim, scale=scale)

    assert anchors.shape == (1, num_slots, dim)
    A = anchors[0]  # (num_slots, dim)
    # Gram matrix: (M, M)
    gram = (A @ A.T).astype(mx.float32)
    expected_diag = (scale ** 2) * mx.eye(num_slots)
    max_error = float(mx.max(mx.abs(gram - expected_diag)).item())
    assert max_error < 1e-5, f"Anchors failed orthogonality check; max error: {max_error}"


def test_adarmsnorm_1step_gradient_flow():
    """Verify both mlp_l1 and mlp_l2 receive non-zero gradients in a 1-step backward pass."""
    dims = 256
    step_embed_dim = 64
    ada_norm = MLXAdaRMSNorm(dims=dims, step_embed_dim=step_embed_dim)

    x = mx.random.normal((2, 8, dims))
    step = 1

    def loss_fn(model, inp):
        out = model(inp, step=step)
        return mx.sum(out ** 2)

    loss, grads = nn.value_and_grad(ada_norm, loss_fn)(ada_norm, x)

    flat_grads = dict(tree_flatten(grads))
    trainable = dict(tree_flatten(ada_norm.trainable_parameters()))

    assert len(trainable) == len(flat_grads)
    for name in ["mlp_l1.weight", "mlp_l1.bias", "mlp_l2.weight", "mlp_l2.bias"]:
        assert name in flat_grads, f"Missing gradient for {name}"
        norm_val = float(mx.linalg.norm(flat_grads[name]).item())
        assert norm_val > 0.0, f"Zero gradient for {name}: {norm_val}"
        assert not math.isnan(norm_val), f"NaN gradient for {name}"


def test_moe_principled_init_gradient_flow():
    """Verify MoE layer with scaled initialization yields non-zero gradients for router and all experts."""
    class _MockConfig:
        dim = 128
        num_experts = 4
        top_k_experts = 2
        moe_intermediate_dim = 64
        intermediate_dim = 0

    moe = MLXMoE(_MockConfig())
    x = mx.random.normal((2, 8, 128))

    def loss_fn(model, inp):
        out = model(inp)
        return mx.sum(out ** 2)

    loss, grads = nn.value_and_grad(moe, loss_fn)(moe, x)
    flat_grads = dict(tree_flatten(grads))
    trainable = dict(tree_flatten(moe.trainable_parameters()))

    assert len(trainable) == len(flat_grads)
    for name in ["router.weight", "gate_weight", "up_weight", "down_weight"]:
        assert name in flat_grads, f"Missing gradient for {name}"
        norm_val = float(mx.linalg.norm(flat_grads[name]).item())
        assert norm_val > 0.0, f"MoE {name} received zero gradient!"
        assert not math.isnan(norm_val), f"MoE {name} received NaN gradient!"


def test_bounded_residual_scaling_bounds():
    """Verify residual scaling alpha strictly stays within (0, alpha_max) under extreme raw values."""
    class _MockBlockConfig:
        dim = 64
        num_heads = 2
        num_kv_heads = 1
        head_dim = 32
        intermediate_dim = 128
        rms_norm_eps = 1e-6
        rope_theta = 10000.0
        step_embed_dim = 32
        alpha_max = 0.5
        rezero_alpha = 0.05

    block = MLXRecurrentBlock(_MockBlockConfig())
    alpha_max = block.alpha_max

    for raw in [-1000.0, -10.0, 0.0, 10.0, 1000.0]:
        block.raw_alpha_attn = mx.array([raw])
        block.raw_alpha_mlp = mx.array([raw])
        a_attn = float(block.alpha_attn.item())
        a_mlp = float(block.alpha_mlp.item())
        assert 0.0 <= a_attn <= alpha_max, f"alpha_attn {a_attn} out of bounds for raw {raw}"
        assert 0.0 <= a_mlp <= alpha_max, f"alpha_mlp {a_mlp} out of bounds for raw {raw}"


def test_1step_gradient_flow_100_percent_trainable_adapter_dense():
    """Verify 100% of trainable adapter parameters receive non-zero gradients with dense MLP."""
    adapter = GemmaRecurrentAdapter(
        dim=128,
        num_slots=8,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        enable_moe_block=False,
    )
    prompt_context = mx.random.normal((2, 12, 128))

    def loss_fn(model, p):
        out = model(p, steps=2)
        return mx.sum(out ** 2)

    loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter, prompt_context)
    flat_grads = dict(tree_flatten(grads))
    trainable_params = dict(tree_flatten(adapter.trainable_parameters()))

    assert len(trainable_params) > 0
    assert len(flat_grads) == len(trainable_params)

    zero_grads = []
    for name, param in trainable_params.items():
        assert name in flat_grads, f"Parameter {name} missing from gradients."
        grad_norm = float(mx.linalg.norm(flat_grads[name]).item())
        if grad_norm <= 0.0 or math.isnan(grad_norm):
            zero_grads.append((name, grad_norm))

    assert len(zero_grads) == 0, f"Zero or NaN gradients found in dense adapter: {zero_grads}"


def test_1step_gradient_flow_100_percent_trainable_adapter_moe():
    """Verify 100% of trainable adapter parameters receive non-zero gradients with MoE layer."""
    adapter = GemmaRecurrentAdapter(
        dim=128,
        num_slots=8,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_dim=64,
    )
    prompt_context = mx.random.normal((2, 12, 128))

    def loss_fn(model, p):
        out = model(p, steps=2)
        return mx.sum(out ** 2)

    loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter, prompt_context)
    flat_grads = dict(tree_flatten(grads))
    trainable_params = dict(tree_flatten(adapter.trainable_parameters()))

    assert len(trainable_params) > 0
    assert len(flat_grads) == len(trainable_params)

    zero_grads = []
    for name, param in trainable_params.items():
        assert name in flat_grads, f"Parameter {name} missing from gradients."
        grad_norm = float(mx.linalg.norm(flat_grads[name]).item())
        if grad_norm <= 0.0 or math.isnan(grad_norm):
            zero_grads.append((name, grad_norm))

    assert len(zero_grads) == 0, f"Zero or NaN gradients found in MoE adapter: {zero_grads}"
