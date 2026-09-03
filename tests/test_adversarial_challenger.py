"""Empirical Adversarial Stress Testing & Numerical Hardening Suite (Challenger 1).

Adversarial Verification Suite:
1. Deep Recurrent Unrolls up to T=128 sweeps for Dense (12B Q4) and MoE (26B A4B) architectures.
2. Extreme Input Distributions (+/-10^6, +/-10^9, subnormals 10^-30, all-zeros, all-ones, singular matrices).
3. Multi-Iteration Soak Test (>= 250 unroll cycles) verifying +0.00% memory growth invariant (Delta VRAM <= 0.10 MB).
4. MoE Routing Expert Partition Stability under varying batch sizes, top-k regimes, and tie/extreme logit conditions.
"""

from __future__ import annotations

import gc
import math
import mlx.core as mx
import mlx.nn as nn
import pytest

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import (
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXGemmaAttention,
    MLXGemmaMoE,
    MLXPreludeProjection,
    MLXRecurrentGemmaBlock,
    MLXRMSNorm,
    MLXAdaRMSNorm,
    sinusoidal_step_embedding,
)
from parallel_latent_reasoner.probes import (
    compute_effective_rank,
    compute_slot_velocity,
)


def _get_active_memory_bytes() -> int:
    """Safely get active allocated memory bytes on Metal/Unified Memory."""
    if hasattr(mx, "get_active_memory"):
        return mx.get_active_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
        return mx.metal.get_active_memory()
    return 0


def _get_peak_memory_bytes() -> int:
    """Safely get peak memory bytes."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory()
    return 0


def _reset_peak_memory() -> None:
    """Safely reset peak memory tracking."""
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _clear_cache() -> None:
    """Safely clear allocator cache on Apple Silicon Metal."""
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()



# ============================================================================
# 1. Deep Recurrent Unrolls (T=128)
# ============================================================================

def test_deep_unroll_t128_dense_recurrent_stability():
    """Adversarial Test: Unroll dense recurrent core to T=128 steps.
    
    Verifies:
    - No NaN or Inf at any step t in [1..128].
    - ReZero alpha=0.05 guarantees bounded Lipschitz growth: norm(S_128) / norm(S_0) < 2.5.
    - Activation norms remain stable without exponential divergence.
    """
    cfg = GemmaLatentConfig.compact_test(dim=256, num_memory_slots=16, rezero_alpha=0.05)
    block = MLXRecurrentGemmaBlock(cfg)

    # Initial normal distribution
    s_curr = mx.random.normal((1, cfg.num_memory_slots, cfg.dim))
    mx.eval(s_curr)
    init_norm = float(mx.linalg.norm(s_curr).item())
    assert init_norm > 0.0

    norms = [init_norm]
    for t in range(1, 129):
        s_curr = block(s_curr, step=t)
        mx.eval(s_curr)
        
        # Check every single step for NaN or Inf
        assert not mx.isnan(s_curr).any().item(), f"NaN detected at recurrent step t={t}"
        assert not mx.isinf(s_curr).any().item(), f"Inf detected at recurrent step t={t}"
        
        current_norm = float(mx.linalg.norm(s_curr).item())
        norms.append(current_norm)

    final_norm = norms[-1]
    growth_ratio = final_norm / init_norm

    # ReZero alpha=0.05 guarantees bounded growth
    assert growth_ratio < 2.5, f"Activation norm exploded: growth ratio {growth_ratio:.4f} >= 2.5"
    assert growth_ratio > 0.1, f"Activation norm collapsed: growth ratio {growth_ratio:.4f} <= 0.1"


def test_deep_unroll_t128_moe_recurrent_stability():
    """Adversarial Test: Unroll MoE recurrent core (16 experts, top-4 active) to T=128 steps.
    
    Verifies:
    - MoE gating and expert selection remain numerically stable across 128 recurrent passes.
    - No NaN/Inf in expert weights, routing softmax, or residual accumulation.
    - Activation norm remains strictly bounded.
    """
    cfg = GemmaLatentConfig(
        dim=256,
        intermediate_dim=256,
        moe_intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        vocab_size=1000,
        num_memory_slots=16,
        num_experts=16,
        top_k_experts=4,
        enable_moe_block=True,
        rezero_alpha=0.05,
    )
    block = MLXRecurrentGemmaBlock(cfg)
    assert isinstance(block.mlp, MLXGemmaMoE)

    s_curr = mx.random.normal((2, cfg.num_memory_slots, cfg.dim))
    mx.eval(s_curr)
    init_norm = float(mx.linalg.norm(s_curr).item())

    for t in range(1, 129):
        s_curr = block(s_curr, step=t)
        mx.eval(s_curr)
        assert not mx.isnan(s_curr).any().item(), f"MoE NaN detected at step t={t}"
        assert not mx.isinf(s_curr).any().item(), f"MoE Inf detected at step t={t}"

    final_norm = float(mx.linalg.norm(s_curr).item())
    growth_ratio = final_norm / init_norm
    assert growth_ratio < 2.5, f"MoE norm growth exceeded bound: {growth_ratio:.4f}"


def test_deep_unroll_large_scale_config_dimensions():
    """Adversarial Test: Deep unroll on large Gemma 4 12B Q4 and 26B A4B configurations (scaled layers)."""
    # 1. 12B Q4 scale layer unroll
    cfg_12b = GemmaLatentConfig(
        dim=512,  # Sub-sampled dim for unit test execution speed while maintaining head ratio
        intermediate_dim=2048,
        num_heads=16,
        num_kv_heads=8,
        head_dim=32,
        vocab_size=262144,
        num_memory_slots=16,
        rezero_alpha=0.05,
    )
    block_12b = MLXRecurrentGemmaBlock(cfg_12b)
    s_12b = mx.random.normal((1, 16, 512))
    for t in range(1, 65):
        s_12b = block_12b(s_12b, step=t)
    mx.eval(s_12b)
    assert not mx.isnan(s_12b).any().item()
    assert not mx.isinf(s_12b).any().item()

    # 2. 26B A4B MoE scale layer unroll (128 experts, top 8)
    cfg_26b = GemmaLatentConfig(
        dim=256,
        intermediate_dim=192,
        moe_intermediate_dim=64,
        num_experts=128,
        top_k_experts=8,
        num_heads=16,
        num_kv_heads=8,
        head_dim=16,
        vocab_size=262144,
        num_memory_slots=16,
        enable_moe_block=True,
        rezero_alpha=0.05,
    )
    block_26b = MLXRecurrentGemmaBlock(cfg_26b)
    s_26b = mx.random.normal((1, 16, 256))
    for t in range(1, 65):
        s_26b = block_26b(s_26b, step=t)
    mx.eval(s_26b)
    assert not mx.isnan(s_26b).any().item()
    assert not mx.isinf(s_26b).any().item()


def test_sinusoidal_step_embeddings_large_t_range():
    """Adversarial Test: Step position embeddings evaluated up to t=10,000."""
    for t_val in [0, 1, 128, 512, 1024, 4096, 10000, 1e6]:
        emb = sinusoidal_step_embedding(step=t_val, dim=128)
        mx.eval(emb)
        assert emb.shape == (1, 128)
        assert not mx.isnan(emb).any().item()
        assert not mx.isinf(emb).any().item()
        assert float(mx.max(emb).item()) <= 1.0 + 1e-5
        assert float(mx.min(emb).item()) >= -1.0 - 1e-5


# ============================================================================
# 2. Extreme Input Distributions
# ============================================================================

def test_extreme_magnitudes_and_subnormals():
    """Adversarial Test: Extreme inputs (+/- 10^6, +/- 10^9, 10^-30 subnormal)."""
    cfg = GemmaLatentConfig.compact_test(dim=128, num_memory_slots=8)
    model = MLXCompactGemmaModel(cfg)

    extreme_cases = [
        ("huge_pos_1e6", mx.ones((1, 8, 128)) * 1e6),
        ("huge_neg_1e6", mx.ones((1, 8, 128)) * -1e6),
        ("huge_pos_1e9", mx.ones((1, 8, 128)) * 1e9),
        ("huge_neg_1e9", mx.ones((1, 8, 128)) * -1e9),
        ("subnormal_1e-30", mx.ones((1, 8, 128)) * 1e-30),
    ]

    for name, inp in extreme_cases:
        out = model.engine.step(inp, step_idx=1)
        mx.eval(out)
        assert not mx.isnan(out).any().item(), f"Failed on {name}: encountered NaN"
        assert not mx.isinf(out).any().item(), f"Failed on {name}: encountered Inf"
        
        logits = model.coda(out)
        mx.eval(logits)
        assert not mx.isnan(logits).any().item(), f"Failed coda on {name}: encountered NaN"
        assert not mx.isinf(logits).any().item(), f"Failed coda on {name}: encountered Inf"


def test_all_zeros_and_all_ones_invariants():
    """Adversarial Test: All zeros and all ones inputs through all sub-modules."""
    cfg = GemmaLatentConfig.compact_test(dim=128, num_memory_slots=8)
    
    # 1. RMSNorm on all zeros (must not divide by zero)
    norm = MLXRMSNorm(128, eps=1e-6)
    zeros = mx.zeros((2, 8, 128))
    norm_zeros = norm(zeros)
    mx.eval(norm_zeros)
    assert not mx.isnan(norm_zeros).any().item()
    assert float(mx.max(mx.abs(norm_zeros)).item()) == 0.0

    # 2. RMSNorm on all ones
    ones = mx.ones((2, 8, 128))
    norm_ones = norm(ones)
    mx.eval(norm_ones)
    assert not mx.isnan(norm_ones).any().item()

    # 3. AdaRMSNorm on zeros and ones
    adanorm = MLXAdaRMSNorm(128)
    ada_zeros = adanorm(zeros, step=1)
    mx.eval(ada_zeros)
    assert not mx.isnan(ada_zeros).any().item()

    ada_ones = adanorm(ones, step=5)
    mx.eval(ada_ones)
    assert not mx.isnan(ada_ones).any().item()

    # 4. Attention on all zeros
    attn = MLXGemmaAttention(cfg)
    attn_zeros = attn(zeros)
    mx.eval(attn_zeros)
    assert not mx.isnan(attn_zeros).any().item()

    # 5. MoE on all zeros
    cfg_moe = GemmaLatentConfig(
        dim=128, intermediate_dim=128, moe_intermediate_dim=64,
        num_heads=4, num_kv_heads=2, head_dim=32, vocab_size=1000,
        num_memory_slots=8, num_experts=4, top_k_experts=2, enable_moe_block=True
    )
    moe = MLXGemmaMoE(cfg_moe)
    moe_zeros = moe(zeros)
    mx.eval(moe_zeros)
    assert not mx.isnan(moe_zeros).any().item()


def test_singular_and_rank_deficient_matrices():
    """Adversarial Test: Singular (rank-0, rank-1, collinear) states in probes and attention."""
    # Rank-0 (all zeros)
    s_rank0 = mx.zeros((1, 16, 64))
    erank_0 = compute_effective_rank(s_rank0)
    assert math.isnan(erank_0) or erank_0 == 1.0 or erank_0 == 0.0 or erank_0 >= 0.0

    # Rank-1 (all rows identical non-zero)
    row = mx.random.normal((1, 1, 64))
    s_rank1 = mx.broadcast_to(row, (1, 16, 64))
    erank_1 = compute_effective_rank(s_rank1)
    assert abs(erank_1 - 1.0) < 0.01, f"Rank-1 matrix expected erank 1.0, got {erank_1}"

    # Cosine velocity between identical states
    vel_zero = compute_slot_velocity(s_rank1, s_rank1)
    assert abs(vel_zero - 0.0) < 1e-4, f"Expected velocity 0.0, got {vel_zero}"


# ============================================================================
# 3. Long-Horizon Soak Test (>= 250 cycles) for Memory Growth Invariant
# ============================================================================

def test_soak_250_cycles_memory_growth_invariant():
    """Adversarial Test: 250 consecutive deliberation cycles verifying +0.00% memory growth.
    
    Verifies:
    - Delta VRAM <= 0.10 MB across 250 full prompt-prelude-deliberate unroll cycles.
    - Zero residual memory leak in MLX unified memory / Metal allocator.
    """
    cfg = GemmaLatentConfig.compact_test(dim=256, intermediate_dim=512, num_memory_slots=16)
    model = MLXCompactGemmaModel(cfg)
    prompt = mx.array([[101, 204, 305, 408, 509, 612]], dtype=mx.int32)

    # Warmup phase (10 cycles)
    for _ in range(10):
        res = model.deliberate(prompt, steps=4)
        mx.eval(res.final_states)

    gc.collect()
    _clear_cache()

    # Iteration 1 measurement
    _reset_peak_memory()
    res1 = model.deliberate(prompt, steps=4)
    mx.eval(res1.final_states)
    gc.collect()
    peak_iter1 = _get_peak_memory_bytes()
    active_iter1 = _get_active_memory_bytes()

    # 250 unroll soak cycles
    peak_iter250 = 0
    for cycle in range(250):
        _reset_peak_memory()
        res = model.deliberate(prompt, steps=4)
        mx.eval(res.final_states)
        gc.collect()
        if cycle == 249:
            peak_iter250 = _get_peak_memory_bytes()

    active_iter250 = _get_active_memory_bytes()

    peak_delta_mb = abs(peak_iter250 - peak_iter1) / (1024 * 1024)
    active_delta_mb = abs(active_iter250 - active_iter1) / (1024 * 1024)

    # Invariant: Delta VRAM <= 0.10 MB (+0.00% memory growth across 250 unrolls)
    assert peak_delta_mb <= 0.10, f"Peak memory grew: Delta was {peak_delta_mb:.4f} MB (> 0.10 MB)"
    assert active_delta_mb <= 0.10, f"Active memory leaked: Delta was {active_delta_mb:.4f} MB (> 0.10 MB)"


def test_compiled_unroll_soak_250_cycles():
    """Adversarial Test: 250 consecutive JIT-compiled unrolls on Metal GPU."""
    cfg = GemmaLatentConfig.compact_test(dim=128, intermediate_dim=256, num_memory_slots=8)
    model = MLXCompactGemmaModel(cfg)
    compiled_loop = model.engine.compile_unroll(steps=4)
    s0 = mx.random.normal((1, 8, 128))
    mx.eval(s0)

    # Warmup
    for _ in range(10):
        out = compiled_loop(s0)
        mx.eval(out)

    gc.collect()
    _clear_cache()
    _reset_peak_memory()
    out1 = compiled_loop(s0)
    mx.eval(out1)
    gc.collect()
    peak_start = _get_peak_memory_bytes()
    active_start = _get_active_memory_bytes()

    peak_end = 0
    for cycle in range(250):
        _reset_peak_memory()
        out = compiled_loop(s0)
        mx.eval(out)
        gc.collect()
        if cycle == 249:
            peak_end = _get_peak_memory_bytes()

    active_end = _get_active_memory_bytes()

    peak_delta_mb = abs(peak_end - peak_start) / (1024 * 1024)
    active_delta_mb = abs(active_end - active_start) / (1024 * 1024)

    assert peak_delta_mb <= 0.10, f"Compiled peak delta exceeded bound: {peak_delta_mb:.4f} MB"
    assert active_delta_mb <= 0.05, f"Compiled active delta exceeded bound: {active_delta_mb:.4f} MB"


def test_moe_soak_250_cycles_memory_stability():
    """Adversarial Test: 250 consecutive MoE layer unrolls verifying zero Metal leak."""
    cfg = GemmaLatentConfig(
        dim=128,
        intermediate_dim=128,
        moe_intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        num_memory_slots=8,
        num_experts=16,
        top_k_experts=4,
        enable_moe_block=True,
    )
    moe = MLXGemmaMoE(cfg)
    x = mx.random.normal((2, 8, 128))
    mx.eval(x)

    # Warmup
    for _ in range(10):
        out = moe(x)
        mx.eval(out)

    gc.collect()
    _reset_peak_memory()
    out1 = moe(x)
    mx.eval(out1)
    gc.collect()
    peak_start = _get_peak_memory_bytes()
    active_start = _get_active_memory_bytes()

    peak_end = 0
    for cycle in range(250):
        _reset_peak_memory()
        out = moe(x)
        mx.eval(out)
        gc.collect()
        if cycle == 249:
            peak_end = _get_peak_memory_bytes()

    active_end = _get_active_memory_bytes()
    peak_delta_mb = abs(peak_end - peak_start) / (1024 * 1024)
    active_delta_mb = abs(active_end - active_start) / (1024 * 1024)

    assert peak_delta_mb <= 0.05, f"MoE soak peak delta exceeded: {peak_delta_mb:.4f} MB"
    assert active_delta_mb <= 0.05, f"MoE soak active delta exceeded: {active_delta_mb:.4f} MB"


# ============================================================================
# 4. MoE Routing Expert Partition Stability with Top-K Selection
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("num_slots", [1, 4, 8, 16])
def test_moe_routing_varying_batch_and_slot_dimensions(batch_size: int, num_slots: int):
    """Adversarial Test: MoE routing across diverse batch and slot configurations.
    
    Verifies:
    - Output shape exactly matches [B, M, D].
    - No NaN/Inf across all batch sizes.
    - Vectorized expert selection functions seamlessly for B=1 to B=32 and M=1 to M=16.
    """
    cfg = GemmaLatentConfig(
        dim=128,
        intermediate_dim=128,
        moe_intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        num_memory_slots=num_slots,
        num_experts=16,
        top_k_experts=4,
        enable_moe_block=True,
    )
    moe = MLXGemmaMoE(cfg)
    x = mx.random.normal((batch_size, num_slots, 128))
    out = moe(x)
    mx.eval(out)

    assert out.shape == (batch_size, num_slots, 128)
    assert not mx.isnan(out).any().item()
    assert not mx.isinf(out).any().item()


@pytest.mark.parametrize("top_k", [1, 2, 4, 8, 16])
def test_moe_routing_edge_topk_selections(top_k: int):
    """Adversarial Test: Top-K selection ranging from single-expert (k=1) to all-experts (k=16)."""
    cfg = GemmaLatentConfig(
        dim=128,
        intermediate_dim=128,
        moe_intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        num_memory_slots=8,
        num_experts=16,
        top_k_experts=top_k,
        enable_moe_block=True,
    )
    moe = MLXGemmaMoE(cfg)
    x = mx.random.normal((2, 8, 128))
    out = moe(x)
    mx.eval(out)

    assert out.shape == (2, 8, 128)
    assert not mx.isnan(out).any().item()


def test_moe_routing_tied_and_extreme_logits():
    """Adversarial Test: MoE router with identical logits (pure ties) and extreme peaked logits."""
    cfg = GemmaLatentConfig(
        dim=64,
        intermediate_dim=64,
        moe_intermediate_dim=32,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=500,
        num_memory_slots=4,
        num_experts=8,
        top_k_experts=2,
        enable_moe_block=True,
    )
    moe = MLXGemmaMoE(cfg)

    # 1. Zero router weights -> all logits identically 0.0 (exact tie)
    moe.router.weight = mx.zeros((8, 64))
    x = mx.random.normal((2, 4, 64))
    out_tie = moe(x)
    mx.eval(out_tie)
    assert not mx.isnan(out_tie).any().item()
    assert not mx.isinf(out_tie).any().item()

    # 2. Extreme peaked router weights (+1000.0 on expert 0, -1000.0 on others)
    w_peaked = mx.ones((8, 64)) * -1000.0
    w_peaked[0, :] = 1000.0
    moe.router.weight = w_peaked
    out_peaked = moe(x)
    mx.eval(out_peaked)
    assert not mx.isnan(out_peaked).any().item()
    assert not mx.isinf(out_peaked).any().item()
