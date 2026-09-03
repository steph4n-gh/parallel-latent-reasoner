"""Comprehensive Test Suite for PRLR Hybrid Deliberate-Then-Verify Pipeline and JIT Execution.

Covers:
1. Pure Latent Deliberation (Mode 1) with 3-Signal Dynamic Consensus E-Gate:
   - Signal 1: Velocity decay v(t)/v(1) < 0.10
   - Signal 2: Coda discrete prediction consensus y_hat^(t) == y_hat^(t-1)
   - Signal 3: SVD effective rank plateau |delta erank| < 0.005
   - Dynamic consensus halting, patience, and telemetry
2. Hybrid 'Deliberate-Then-Verify' Execution (Mode 2):
   - Phase 1: High-speed parallel Jacobi sweeps in SRAM cache
   - Phase 2: Concise grounded discrete token decoding directly conditioned on thought vector
   - Structured HybridDeliberationResult contract validation
   - Interface parity between deliberate_and_verify, deliberate_then_verify, and generate
3. Automatic and Explicit Adapter Checkpoint Loading:
   - Automatic discovery of checkpoints/prlr_latent_adapter.npz
   - Explicit path resolution and parameter binding
   - FileNotFoundError handling for missing checkpoints
4. MLX JIT Compilation Stability (@mx.compile):
   - Static graph unroll verification
   - Numerical equivalence between JIT compiled and eager execution
5. Representation Health and Memory Invariants:
   - Variable batch sizes (B=1, 2, 4) and memory slots (M=8, 16, 32)
   - ReZero Lipschitz growth ratio ||S^(T)|| / ||S^(0)|| <= 2.5
   - Effective rank health (erank >= 4.0 out of M=16)
   - Strictly zero KV-cache growth during deliberation sweeps
"""

from __future__ import annotations

import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import (
    DynamicConsensusEGate,
    DynamicDeliberationGate,
    GateDecision,
    GateTelemetry,
)
from parallel_latent_reasoner.engine import DeliberationResult, MLXParallelLatentEngine
from parallel_latent_reasoner.models import (
    MLXAdaRMSNorm,
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXPreludeProjection,
    MLXRMSNorm,
    MLXRecurrentGemmaBlock,
)
from parallel_latent_reasoner.pipeline import (
    DeliberationPipelineOutput,
    GemmaDeliberationPipeline,
    HybridDeliberationResult,
    PRLRPipeline,
)
from parallel_latent_reasoner.probes import (
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
)


# ============================================================================
# Tier 1 Tests: 3-Signal Dynamic Consensus E-Gate Mechanics
# ============================================================================

def test_egate_three_signals_all_converge():
    """Verify E-Gate triggers early halt when velocity decays, Coda agrees, and erank plateaus."""
    gate = DynamicDeliberationGate(
        tol_rel_vel=0.10,
        tol_erank_delta=0.005,
        min_steps=2,
        max_steps=10,
        patience=1,
    )

    B, M, D = 1, 16, 128
    base_state = mx.random.normal((B, M, D))
    # Step 0: initialization
    t0 = gate.update(base_state, step=0, coda_token=42)
    assert t0.halt is False
    assert t0.exit_reason == "initialization"

    # Step 1: initial movement (v1 established)
    s1 = base_state + 0.5 * mx.random.normal((B, M, D))
    t1 = gate.update(s1, step=1, coda_token=100)
    assert t1.halt is False
    assert t1.exit_reason == "active"

    # Step 2: near-zero movement, same coda token 100, saturated erank
    s2 = s1 + 0.001 * mx.random.normal((B, M, D))
    t2 = gate.update(s2, step=2, coda_token=100)
    mx.eval(s2)

    assert t2.signal_velocity is True, f"Relative velocity {t2.rel_velocity} should be < 0.10"
    assert t2.signal_coda is True, "Coda prediction should match previous step"
    assert t2.signal_erank is True, f"Delta erank {t2.delta_erank} should be < 0.005"
    assert t2.halt is True, "E-Gate must halt when all 3 signals converge"
    assert t2.exit_reason == "3_signal_consensus"


def test_egate_single_signal_divergence_prevents_premature_halt():
    """Verify that if even one signal disagrees, the E-Gate does not halt prematurely."""
    gate = DynamicDeliberationGate(
        tol_rel_vel=0.10,
        tol_erank_delta=0.005,
        min_steps=2,
        max_steps=10,
        patience=1,
    )

    B, M, D = 1, 16, 128
    s0 = mx.random.normal((B, M, D))
    gate.update(s0, step=0, coda_token=42)

    s1 = s0 + 0.5 * mx.random.normal((B, M, D))
    gate.update(s1, step=1, coda_token=100)

    # Condition: Velocity decays, erank plateaus, BUT Coda prediction flips (token 200 != 100)
    s2 = s1 + 0.001 * mx.random.normal((B, M, D))
    t2 = gate.update(s2, step=2, coda_token=200)

    assert t2.signal_velocity is True
    assert t2.signal_coda is False, "Coda tokens differ, signal_coda must be False"
    assert t2.halt is False, "E-Gate must NOT halt when Coda prediction is unstable"
    assert t2.exit_reason == "active"


def test_egate_max_steps_timeout():
    """Verify that if representations never converge, E-Gate halts strictly at max_steps."""
    max_T = 6
    gate = DynamicDeliberationGate(
        tol_rel_vel=0.01,  # Strict velocity tolerance
        tol_erank_delta=0.0001,
        min_steps=2,
        max_steps=max_T,
    )

    B, M, D = 1, 16, 128
    curr = mx.random.normal((B, M, D))
    gate.update(curr, step=0, coda_token=0)

    for t in range(1, max_T + 1):
        curr = curr + 0.5 * mx.random.normal((B, M, D))  # Continuous large motion
        telemetry = gate.update(curr, step=t, coda_token=t)
        if t < max_T:
            assert telemetry.halt is False
        else:
            assert telemetry.halt is True
            assert telemetry.exit_reason == "max_steps_timeout"


def test_egate_patience_parameter():
    """Verify patience > 1 requires consecutive consensus steps before halting."""
    gate = DynamicDeliberationGate(
        tol_rel_vel=0.10,
        tol_erank_delta=0.005,
        min_steps=2,
        max_steps=10,
        patience=2,  # Require 2 consecutive consensus steps
    )

    B, M, D = 1, 16, 128
    s0 = mx.random.normal((B, M, D))
    gate.update(s0, step=0, coda_token=42)

    s1 = s0 + 0.5 * mx.random.normal((B, M, D))
    gate.update(s1, step=1, coda_token=100)

    # First converged step: patience=2 means it should NOT halt yet
    s2 = s1 + 0.001 * mx.random.normal((B, M, D))
    t2 = gate.update(s2, step=2, coda_token=100)
    assert t2.signal_velocity and t2.signal_coda and t2.signal_erank
    assert t2.halt is False, "Patience=2 should not halt on first consensus step"

    # Second converged step: now it should halt
    s3 = s2 + 0.001 * mx.random.normal((B, M, D))
    t3 = gate.update(s3, step=3, coda_token=100)
    assert t3.halt is True
    assert t3.exit_reason == "3_signal_consensus"


# ============================================================================
# Tier 1 Tests: PRLRPipeline and Hybrid Execution Interfaces
# ============================================================================

def test_prlr_pipeline_class_and_alias_parity():
    """Verify PRLRPipeline and GemmaDeliberationPipeline are functional and identical."""
    p1 = PRLRPipeline.from_preset("compact_test")
    p2 = GemmaDeliberationPipeline.from_preset("compact_test")

    assert isinstance(p1, PRLRPipeline)
    assert isinstance(p2, PRLRPipeline)
    assert p1.config.dim == p2.config.dim


def test_hybrid_deliberate_and_verify_interface():
    """Verify deliberate_and_verify returns full HybridDeliberationResult with structured fields."""
    pipeline = PRLRPipeline.from_preset("compact_test")
    prompt = "Balance 4 items under weight limit 15kg"

    result = pipeline.deliberate_and_verify(
        prompt=prompt,
        max_steps=6,
        generate_tokens=16,
        enable_dynamic_gate=True,
        return_diagnostics=True,
    )

    assert isinstance(result, HybridDeliberationResult)
    assert result.prompt == prompt
    assert isinstance(result.decoded_text, str)
    assert result.token_ids.shape == (1, 16)
    assert 2 <= result.deliberation_steps <= 6
    assert result.final_states.shape == (1, pipeline.config.num_memory_slots, pipeline.config.dim)

    # Telemetry and verdict
    assert result.egate_verdict in ("3_signal_consensus", "max_steps_timeout", "active")
    assert result.gate_telemetry is not None
    assert len(result.gate_telemetry) >= 2

    # Latency breakdown
    assert "prefill_latency_ms" in result.latency_breakdown
    assert "deliberation_latency_ms" in result.latency_breakdown
    assert "coda_decode_latency_ms" in result.latency_breakdown
    assert "total_latency_ms" in result.latency_breakdown
    assert "throughput_tok_per_sec" in result.latency_breakdown

    # Memory stats
    assert "peak_memory_mb" in result.memory_stats
    assert "active_memory_mb" in result.memory_stats
    assert result.memory_stats["kv_cache_growth_pct"] == 0.0

    # Backwards-compatibility aliases
    assert result.metrics == result.latency_breakdown
    assert result.thought_trajectory == result.trajectory_states


def test_deliberate_then_verify_alias_parity():
    """Verify deliberate_then_verify behaves identically to deliberate_and_verify."""
    pipeline = PRLRPipeline.from_preset("compact_test")
    prompt = mx.array([[10, 20, 30, 40]], dtype=mx.int32)

    res1 = pipeline.deliberate_and_verify(prompt, max_steps=4, generate_tokens=8, temperature=0.0)
    res2 = pipeline.deliberate_then_verify(prompt, max_steps=4, generate_tokens=8, temperature=0.0)

    mx.eval(res1.token_ids, res2.token_ids)
    assert mx.array_equal(res1.token_ids, res2.token_ids).item()
    assert res1.deliberation_steps == res2.deliberation_steps


def test_hybrid_decoding_without_intermediate_tokens():
    """Verify pure latent deliberation produces final solution tokens with zero intermediate token emit."""
    config = GemmaLatentConfig.compact_test()
    pipeline = PRLRPipeline(config=config)

    prompt = mx.array([[10, 20, 30]], dtype=mx.int32)
    delib_res, telemetry = pipeline.deliberate(prompt, steps=5, return_trajectory=True)

    # Trajectory must only hold continuous latent memory tensors [B, M, D]
    assert delib_res.trajectory_states is not None
    assert len(delib_res.trajectory_states) == 6  # S^(0) through S^(5)
    for state in delib_res.trajectory_states:
        assert state.shape == (1, config.num_memory_slots, config.dim)
        mx.eval(state)
        assert not mx.isnan(state).any().item()
        assert not mx.isinf(state).any().item()


def test_decode_solution_text_formatting():
    """Verify decode_solution helper translates token ID arrays to strings."""
    pipeline = PRLRPipeline.from_preset("compact_test")

    # ASCII token IDs (e.g. ord('H'), ord('i'), ord('!'))
    toks = mx.array([[72, 105, 33]], dtype=mx.int32)
    decoded_str = pipeline.decode_solution(toks)
    assert decoded_str == "Hi!"

    # Sequence of ints
    assert pipeline.decode_solution([65, 66, 67]) == "ABC"


# ============================================================================
# Tier 2 Tests: Adapter Loading & Checkpoint Resolution
# ============================================================================

def test_automatic_adapter_loading():
    """Verify automatic loading of production checkpoint prlr_latent_adapter.npz."""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "prlr_latent_adapter.npz"
    if not ckpt_path.exists():
        pytest.skip("Production checkpoint not found in local path.")

    pipeline = PRLRPipeline.from_preset("compact_test", load_trained_adapter=True)
    assert pipeline.adapter_loaded is True
    assert pipeline.adapter_path is not None
    assert "prlr_latent_adapter" in pipeline.adapter_path

    # Verify forward pass with loaded adapter
    out = pipeline.deliberate_and_verify("Verify 2 + 2 = 4", max_steps=4, generate_tokens=8)
    assert out.adapter_loaded is True
    assert out.token_ids.shape == (1, 8)


def test_explicit_adapter_path_loading():
    """Verify passing explicit adapter checkpoint path loads and binds weights."""
    legacy_path = Path(__file__).resolve().parent.parent / "checkpoints" / "legacy_invalid_objective" / "prlr_latent_adapter.npz"
    ckpt_path = legacy_path if legacy_path.exists() else (Path(__file__).resolve().parent.parent / "checkpoints" / "prlr_latent_adapter.npz")
    if not ckpt_path.exists():
        pytest.skip("Production or legacy checkpoint not found in local path.")

    pipeline = PRLRPipeline.from_preset("compact_test", adapter_path=str(ckpt_path))
    assert pipeline.adapter_loaded is True
    assert pipeline.adapter_path == str(ckpt_path)


def test_missing_adapter_raises_file_not_found():
    """Verify passing nonexistent adapter path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PRLRPipeline.from_preset("compact_test", adapter_path="nonexistent_checkpoint_xyz.npz")


# ============================================================================
# Tier 2 Tests: MLX JIT Compilation Stability & Numerical Parity
# ============================================================================

def test_mlx_jit_compiled_step_numerical_equivalence():
    """Verify @mx.compile on recurrent step produces bit-for-bit numerical equivalence to eager mode."""
    config = GemmaLatentConfig.compact_test()
    model = MLXCompactGemmaModel(config)

    slots = mx.random.normal((1, config.num_memory_slots, config.dim))
    prompt_hiddens = mx.random.normal((1, 10, config.dim))
    prompt_kv = model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

    # 1. Eager step
    eager_out = model.engine.step(slots, step_idx=1, prompt_kv=prompt_kv, prompt_len=10)
    mx.eval(eager_out)

    # 2. JIT compiled step
    compiled_step_fn = mx.compile(model.engine.step)
    compiled_out = compiled_step_fn(slots, step_idx=1, prompt_kv=prompt_kv, prompt_len=10)
    mx.eval(compiled_out)

    assert mx.allclose(eager_out, compiled_out, atol=1e-5), "Compiled and eager step executions must match."


def test_mlx_jit_multi_iteration_graph_stability():
    """Verify compiled step executes across 30 consecutive iterations without graph re-tracing failure."""
    config = GemmaLatentConfig.compact_test()
    engine = MLXParallelLatentEngine(config)

    curr = mx.random.normal((2, config.num_memory_slots, config.dim))
    compiled_step = mx.compile(engine.step)

    for step_idx in range(1, 31):
        curr = compiled_step(curr, step_idx=step_idx)
        mx.eval(curr)
        assert curr.shape == (2, config.num_memory_slots, config.dim)
        assert not mx.isnan(curr).any().item()
        assert not mx.isinf(curr).any().item()


def test_jit_compiled_pipeline_execution():
    """Verify pipeline executes smoothly with compile_engine=True and compile_decoder=True."""
    p_jit = PRLRPipeline.from_preset("compact_test", compile_engine=True, compile_decoder=True)
    p_eager = PRLRPipeline.from_preset("compact_test", compile_engine=False, compile_decoder=False)

    # Copy weights for exact parity check
    p_eager.model.update(p_jit.model.parameters())

    prompt = mx.array([[5, 10, 15, 20]], dtype=mx.int32)
    out_jit = p_jit.deliberate_and_verify(prompt, max_steps=4, generate_tokens=8, temperature=0.0)
    out_eager = p_eager.deliberate_and_verify(prompt, max_steps=4, generate_tokens=8, temperature=0.0)

    mx.eval(out_jit.token_ids, out_eager.token_ids)
    assert mx.array_equal(out_jit.token_ids, out_eager.token_ids).item()


# ============================================================================
# Tier 2 Tests: Boundary Conditions, Scaling & Representation Invariants
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_variable_batch_sizes(batch_size: int):
    """Verify pipeline forward and deliberation passes across multiple batch sizes."""
    pipeline = PRLRPipeline.from_preset("compact_test")

    prompts = mx.zeros((batch_size, 8), dtype=mx.int32)
    output = pipeline.generate(prompts, max_new_tokens=4, deliberation_steps=3, enable_dynamic_gate=False)
    assert output.token_ids.shape == (batch_size, 4)
    assert output.final_states.shape == (batch_size, pipeline.config.num_memory_slots, pipeline.config.dim)


@pytest.mark.parametrize("num_slots", [8, 16, 32])
def test_variable_memory_slots(num_slots: int):
    """Verify architecture correctly scales with M=8, 16, and 32 working memory slots."""
    config = GemmaLatentConfig.compact_test(num_memory_slots=num_slots)
    pipeline = PRLRPipeline(config=config)

    out = pipeline.generate("Problem", max_new_tokens=4, deliberation_steps=3)
    assert out.final_states.shape == (1, num_slots, config.dim)


def test_rezero_lipschitz_norm_growth_bounded():
    """Verify ReZero residual scaling bounds hidden state norm growth ratio ||S^(T)|| / ||S^(0)|| <= 2.5."""
    config = GemmaLatentConfig.compact_test(deliberation_steps=16, rezero_alpha=0.05)
    model = MLXCompactGemmaModel(config)

    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    slots0, prompt_hiddens = model.prelude(prompt)
    prompt_len = prompt_hiddens.shape[1]
    prompt_kv = model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

    curr = slots0
    norm0 = mx.linalg.norm(curr).item()

    for t in range(1, 17):
        curr = model.engine.step(curr, step_idx=t, prompt_kv=prompt_kv, prompt_len=prompt_len)

    mx.eval(curr)
    normT = mx.linalg.norm(curr).item()
    growth_ratio = normT / (norm0 + 1e-9)

    assert growth_ratio <= 2.5, f"Lipschitz norm growth ratio ({growth_ratio:.3f}) exceeded safe bound 2.5!"


def test_representation_health_effective_rank():
    """Verify deliberated states with diverse slot embeddings maintain effective rank erank >= 4.0."""
    config = GemmaLatentConfig.compact_test(num_memory_slots=16)
    pipeline = PRLRPipeline(config=config)
    pipeline.model.prelude.slot_embeddings = mx.random.normal((1, 16, config.dim))

    out = pipeline.generate("Complex optimization query", max_new_tokens=4, deliberation_steps=6, return_diagnostics=True)
    mx.eval(out.final_states)

    final_erank = compute_effective_rank(out.final_states)
    assert final_erank >= 4.0, f"Effective rank ({final_erank:.3f}) collapsed below health floor 4.0!"


def test_extreme_magnitude_activations_stability():
    """Verify extreme activations (+1e6, -1e6, 1e-12, all zeros) do not produce NaN or Inf."""
    config = GemmaLatentConfig.compact_test()
    block = MLXRecurrentGemmaBlock(config)

    extreme_inputs = [
        mx.zeros((1, 16, config.dim)),
        mx.ones((1, 16, config.dim)) * 1e6,
        mx.ones((1, 16, config.dim)) * -1e6,
        mx.ones((1, 16, config.dim)) * 1e-12,
    ]

    for inp in extreme_inputs:
        out = block(inp, step=1)
        mx.eval(out)
        assert not mx.isnan(out).any().item(), "Extreme input caused NaN activation!"
        assert not mx.isinf(out).any().item(), "Extreme input caused Inf activation!"
