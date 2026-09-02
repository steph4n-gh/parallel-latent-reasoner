"""End-to-End integration tests for PRLR Gemma Deliberation Pipeline."""

import mlx.core as mx
import pytest

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.pipeline import GemmaDeliberationPipeline


def test_pipeline_generate_various_input_types():
    """Verify pipeline accepts string prompts, integer lists, and mx.arrays."""
    pipeline = GemmaDeliberationPipeline.from_preset("compact_test")

    # 1. String prompt
    out_str = pipeline.generate("What is 2 + 2?", max_new_tokens=8, deliberation_steps=4)
    assert out_str.token_ids.shape == (1, 8)
    assert out_str.deliberation_steps <= 4

    # 2. Integer list prompt
    out_list = pipeline.generate([10, 20, 30, 40], max_new_tokens=8, deliberation_steps=4)
    assert out_list.token_ids.shape == (1, 8)

    # 3. 2D mx.array prompt
    prompt_arr = mx.array([[5, 15, 25, 35]], dtype=mx.int32)
    out_arr = pipeline.generate(prompt_arr, max_new_tokens=8, deliberation_steps=4)
    assert out_arr.token_ids.shape == (1, 8)


def test_zero_intermediate_tokens_emitted():
    """Verify that during deliberation phase, zero intermediate tokens are emitted."""
    pipeline = GemmaDeliberationPipeline.from_preset("compact_test")
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    delib_res, _ = pipeline.deliberate(prompt, steps=8, return_trajectory=True)
    # The deliberation working memory maintains strictly fixed shape [1, M, D]
    assert delib_res.final_states.shape == (1, pipeline.config.num_memory_slots, pipeline.config.dim)
    # The intermediate trajectory contains continuous latent state vectors, not discrete tokens
    assert delib_res.trajectory_states is not None
    assert len(delib_res.trajectory_states) == 9  # S^(0) through S^(8)
    for s in delib_res.trajectory_states:
        assert s.shape == (1, pipeline.config.num_memory_slots, pipeline.config.dim)


def test_greedy_determinism():
    """Verify greedy argmax generation (temperature=0.0) is bit-for-bit deterministic."""
    pipeline = GemmaDeliberationPipeline.from_preset("compact_test")
    prompt = "If a car travels 60 mph for 2.5 hours, how far does it go?"

    out1 = pipeline.generate(prompt, max_new_tokens=12, deliberation_steps=6, temperature=0.0)
    out2 = pipeline.generate(prompt, max_new_tokens=12, deliberation_steps=6, temperature=0.0)

    mx.eval(out1.token_ids, out2.token_ids)
    diff = mx.sum(mx.abs(out1.token_ids - out2.token_ids)).item()
    assert diff == 0, "Greedy generation at temp=0.0 must be strictly deterministic."


def test_dynamic_early_exit_in_pipeline():
    """Verify dynamic E-Gate halts deliberation early when enabled."""
    pipeline = GemmaDeliberationPipeline.from_preset("compact_test")
    prompt = "What is 2 + 2?"

    out = pipeline.generate(
        prompt,
        max_new_tokens=8,
        deliberation_steps=12,
        enable_dynamic_gate=True,
        min_steps=2,
    )
    # Should execute >= min_steps and <= max_steps
    assert 2 <= out.deliberation_steps <= 12
    assert out.gate_telemetry is not None
    assert len(out.gate_telemetry) >= 2


def test_multi_preset_instantiation():
    """Verify instantiation across compact_test, gemma_2b, and gemma_9b presets."""
    p_compact = GemmaDeliberationPipeline.from_preset("compact_test")
    assert p_compact.config.dim == 256

    p_2b = GemmaDeliberationPipeline.from_preset("gemma_2b")
    assert p_2b.config.dim == 2048

    p_9b = GemmaDeliberationPipeline.from_preset("gemma_9b")
    assert p_9b.config.dim == 3584
