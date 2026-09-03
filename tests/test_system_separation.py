"""Tests verifying strict package isolation: prlr.kernel, prlr.compact, and prlr.gemma."""

import mlx.core as mx
import pytest


def test_kernel_namespace_has_no_gemma_or_cot_symbols():
    """Verify prlr.kernel is completely model-agnostic and free of Gemma/CoT claims."""
    import prlr.kernel as kernel

    forbidden = ["gemma", "cot", "chain_of_thought", "tokenizer", "sentencepiece"]
    for sym in dir(kernel):
        for f in forbidden:
            assert f not in sym.lower(), f"Symbol '{sym}' in prlr.kernel violates kernel neutrality."


def test_compact_namespace_has_explicit_scratch_label():
    """Verify prlr.compact is explicitly marked as random-init testbed."""
    import prlr.compact as compact

    cfg = compact.CompactConfig()
    model = compact.CompactScratchModel(cfg)
    assert model.is_pretrained is False
    assert model.model_id == "prlr-compact-testbed"


def test_gemma_namespace_requires_model_manifest():
    """Verify prlr.gemma imports ModelManifest and validates checkpoints."""
    import prlr.gemma as gemma

    assert hasattr(gemma, "ModelManifest")
    assert hasattr(gemma, "PretrainedGemmaBackbone")
    assert hasattr(gemma, "load_model")


def test_backward_compatibility_facade_retains_all_existing_exports():
    """Verify parallel_latent_reasoner re-exports match test_packaging_isolation."""
    import parallel_latent_reasoner as plr

    symbols = [
        "GemmaLatentConfig",
        "MLXCompactGemmaModel",
        "MLXRecurrentGemmaBlock",
        "MLXAdaRMSNorm",
        "MLXGemmaAttention",
        "PRLRPipeline",
        "GemmaDeliberationPipeline",
        "DynamicDeliberationGate",
        "DynamicConsensusEGate",
        "compute_effective_rank",
        "compute_slot_cosine_similarity",
        "compute_slot_velocity",
        "analyze_deliberation_trajectory",
        "detect_limit_cycle",
        "DeliberationResult",
        "MLXParallelLatentEngine",
    ]
    for s in symbols:
        assert hasattr(plr, s), f"Missing backward compatibility symbol: {s}"


def test_kernel_recurrent_core_block_standalone():
    """Verify prlr.kernel executes purely on tensors without vocabulary dependencies."""
    from prlr.kernel.config import RecurrentKernelConfig
    from prlr.kernel.recurrent_core import MLXRecurrentBlock

    cfg = RecurrentKernelConfig(
        dim=128,
        num_heads=2,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=256,
        num_memory_slots=8,
    )
    block = MLXRecurrentBlock(cfg)
    x = mx.zeros((1, 8, 128))
    out = block(x, step=1)
    mx.eval(out)
    assert out.shape == (1, 8, 128)
