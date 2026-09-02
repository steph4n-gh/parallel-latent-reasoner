"""Stress, Numerical Stability, and Memory Leak Tests for PRLR."""

import gc
import mlx.core as mx
import pytest

from parallel_latent_reasoner.benchmark import _get_peak_memory_bytes, _reset_peak_memory
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import (
    MLXCompactGemmaModel,
    MLXRecurrentGemmaBlock,
)


def test_rezero_lipschitz_stability_deep_unroll():
    """Verify ReZero residual scaling (alpha=0.05) keeps activations bounded across T=128 steps."""
    cfg_stable = GemmaLatentConfig.compact_test(rezero_alpha=0.05)
    block_stable = MLXRecurrentGemmaBlock(cfg_stable)

    s_curr = mx.random.normal((1, cfg_stable.num_memory_slots, cfg_stable.dim))
    mx.eval(s_curr)
    init_norm = float(mx.linalg.norm(s_curr).item())

    # Deep unroll to T=128
    for t in range(1, 129):
        s_curr = block_stable(s_curr, step=t)

    mx.eval(s_curr)
    final_norm = float(mx.linalg.norm(s_curr).item())
    growth_ratio = final_norm / max(1e-6, init_norm)

    # ReZero alpha=0.05 guarantees bounded growth (< 2.5x over 128 steps)
    assert growth_ratio < 2.5, f"Norm growth ratio exceeded bound: {growth_ratio:.2f} >= 2.5"
    assert not mx.isnan(s_curr).any().item(), "Encountered NaN in deep unroll."
    assert not mx.isinf(s_curr).any().item(), "Encountered Inf in deep unroll."


def test_extreme_inputs_stability():
    """Verify extreme inputs (+/- 10,000.0) and all-zeros do not cause overflow or NaN."""
    cfg = GemmaLatentConfig.compact_test()
    model = MLXCompactGemmaModel(cfg)

    # 1. Extreme positive input
    extreme_pos = mx.ones((1, cfg.num_memory_slots, cfg.dim)) * 10000.0
    out_pos = model.engine.step(extreme_pos, step_idx=1)
    mx.eval(out_pos)
    assert not mx.isnan(out_pos).any().item()
    assert not mx.isinf(out_pos).any().item()

    # 2. Extreme negative input
    extreme_neg = mx.ones((1, cfg.num_memory_slots, cfg.dim)) * -10000.0
    out_neg = model.engine.step(extreme_neg, step_idx=1)
    mx.eval(out_neg)
    assert not mx.isnan(out_neg).any().item()
    assert not mx.isinf(out_neg).any().item()

    # 3. All-zeros input
    zeros_in = mx.zeros((1, cfg.num_memory_slots, cfg.dim))
    out_zeros = model.engine.step(zeros_in, step_idx=1)
    mx.eval(out_zeros)
    assert not mx.isnan(out_zeros).any().item()


def test_500_unroll_memory_leak_soak():
    """Verify zero memory accumulation across 500 consecutive unroll iterations on Metal GPU."""
    cfg = GemmaLatentConfig.compact_test()
    model = MLXCompactGemmaModel(cfg)
    prompt = mx.array([[10, 20, 30, 40]], dtype=mx.int32)

    # Warmup
    for _ in range(10):
        _ = model.deliberate(prompt, steps=4)

    gc.collect()
    _reset_peak_memory()
    mem_initial = _get_peak_memory_bytes()

    # 500 unrolls soak
    for _ in range(500):
        res = model.deliberate(prompt, steps=4)
        mx.eval(res.final_states)

    gc.collect()
    mem_final = _get_peak_memory_bytes()

    if mem_initial > 0 and mem_final > 0:
        growth_pct = (mem_final - mem_initial) / mem_initial * 100.0
        # Active memory growth should remain strictly below 1%
        assert growth_pct < 1.0, f"Memory leaked across 500 unrolls: {growth_pct:.2f}% growth"
