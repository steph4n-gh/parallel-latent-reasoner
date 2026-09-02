"""Unit tests for PRLR Configuration and Model Architecture."""

import math
import mlx.core as mx
import numpy as np
import pytest

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import (
    MLXAdaRMSNorm,
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXGemmaAttention,
    MLXGemmaMLP,
    MLXPreludeProjection,
    MLXRecurrentGemmaBlock,
    MLXRMSNorm,
    sinusoidal_step_embedding,
)


def test_config_presets():
    """Test all scale presets instantiate with valid dimensions."""
    c_test = GemmaLatentConfig.compact_test()
    assert c_test.dim == 256
    assert c_test.head_dim == 64
    assert c_test.num_heads == 4
    assert c_test.num_memory_slots == 16
    assert c_test.rezero_alpha == 0.05

    c_2b = GemmaLatentConfig.gemma_2b()
    assert c_2b.dim == 2048
    assert c_2b.head_dim == 256
    assert c_2b.num_heads == 8
    assert c_2b.num_kv_heads == 4

    c_9b = GemmaLatentConfig.gemma_9b()
    assert c_9b.dim == 3584
    assert c_9b.intermediate_dim == 14336
    assert c_9b.num_heads == 16

    c_12b = GemmaLatentConfig.gemma_12b()
    assert c_12b.dim == 3840
    assert c_12b.intermediate_dim == 16384

    c_e4b = GemmaLatentConfig.gemma_e4b()
    assert c_e4b.dim == 3072


def test_config_validation_and_serialization():
    """Test invalid parameters raise ValueError and serialization roundtrips."""
    with pytest.raises(ValueError):
        GemmaLatentConfig(dim=-1)
    with pytest.raises(ValueError):
        GemmaLatentConfig(num_heads=0)
    with pytest.raises(ValueError):
        GemmaLatentConfig(rezero_alpha=-0.1)
    with pytest.raises(ValueError):
        GemmaLatentConfig(min_steps=5, max_steps=3)

    cfg = GemmaLatentConfig.compact_test()
    d = cfg.to_dict()
    cfg_roundtrip = GemmaLatentConfig.from_dict(d)
    assert cfg_roundtrip.dim == cfg.dim
    assert cfg_roundtrip.vocab_size == cfg.vocab_size


def test_sinusoidal_step_embedding():
    """Test sinusoidal step embedding generation and properties."""
    dim = 64
    emb1 = sinusoidal_step_embedding(1, dim=dim)
    emb2 = sinusoidal_step_embedding(2, dim=dim)
    assert emb1.shape == (1, dim)
    assert emb2.shape == (1, dim)

    mx.eval(emb1, emb2)
    diff = mx.sum(mx.abs(emb1 - emb2)).item()
    assert diff > 1e-3, "Different steps must yield distinct position embeddings."


def test_rmsnorm_parameterization():
    """Test Gemma RMSNorm (1.0 + weight) parameterization."""
    dims = 64
    norm = MLXRMSNorm(dims=dims, eps=1e-6)
    x = mx.random.normal((2, 10, dims))
    out = norm(x)
    mx.eval(out)
    assert out.shape == (2, 10, dims)
    # Norm along feature axis should be approximately sqrt(dims)
    sq_mean = mx.mean(mx.square(out), axis=-1)
    mx.eval(sq_mean)
    assert mx.allclose(sq_mean, mx.ones_like(sq_mean), atol=1e-3)


def test_adarmsnorm_identity_at_init():
    """Verify AdaRMSNorm is an exact mathematical identity over RMSNorm at initialization."""
    dims = 128
    ada_norm = MLXAdaRMSNorm(dims=dims, step_embed_dim=64)
    std_norm = MLXRMSNorm(dims=dims)

    x = mx.random.normal((2, 16, dims))
    for t in [1, 2, 5, 8, 12]:
        ada_out = ada_norm(x, step=t)
        std_out = std_norm(x)
        mx.eval(ada_out, std_out)
        max_diff = float(mx.max(mx.abs(ada_out - std_out)).item())
        assert max_diff < 1e-5, f"AdaRMSNorm at step {t} diverged from RMSNorm: max diff {max_diff}"


def test_rezero_identity_at_zero_alpha():
    """Verify ReZero with alpha=0.0 acts as a strict identity pass-through."""
    cfg = GemmaLatentConfig.compact_test(rezero_alpha=0.0)
    block = MLXRecurrentGemmaBlock(cfg)
    x = mx.random.normal((1, cfg.num_memory_slots, cfg.dim))
    out = block(x, step=1)
    mx.eval(x, out)
    diff = float(mx.max(mx.abs(x - out)).item())
    assert diff < 1e-6, f"ReZero at alpha=0.0 must be an identity pass-through, got diff {diff}"


def test_coda_lm_head_softcapping():
    """Verify Coda LM Head bounds logits within [-30.0, 30.0] via tanh soft-capping."""
    cfg = GemmaLatentConfig.compact_test(final_logit_softcapping=30.0)
    coda = MLXCodaLMHead(cfg)

    # Test with extreme hidden states (+1e6, -1e6)
    extreme_slots = mx.ones((2, cfg.num_memory_slots, cfg.dim)) * 1e6
    logits = coda(extreme_slots, pool=True)
    mx.eval(logits)

    max_logit = float(mx.max(logits).item())
    min_logit = float(mx.min(logits).item())
    assert max_logit <= 30.0 + 1e-5, f"Logit exceeded softcap 30.0: {max_logit}"
    assert min_logit >= -30.0 - 1e-5, f"Logit exceeded softcap -30.0: {min_logit}"


def test_weight_tying_and_parameter_invariance():
    """Verify embedding weight tying and parameter count invariance across unrolls."""
    cfg = GemmaLatentConfig.compact_test(tie_word_embeddings=True)
    model = MLXCompactGemmaModel(cfg)

    # Verify tied embedding reference
    assert model.coda.embed_tokens is model.embed_tokens
    assert model.prelude.embed_tokens is model.embed_tokens

    # Verify unroll does not mutate parameter tree or allocate new weights
    import mlx.utils
    def get_num_params(m):
        flat = mlx.utils.tree_flatten(m.trainable_parameters())
        return sum(p.size for _, p in flat)

    p0 = get_num_params(model)
    prompt = mx.array([[10, 20, 30, 40]], dtype=mx.int32)

    for T in [1, 2, 4, 8, 16, 32]:
        _ = model.deliberate(prompt, steps=T)
        p_curr = get_num_params(model)
        assert p_curr == p0, f"Parameter count changed after T={T} unrolls: {p_curr} != {p0}"
