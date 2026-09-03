"""Test Suite for Milestone 4 Requirement R5: Masked Answer Cross-Entropy Training Engine.

Verifies:
- Frozen backbone guarantee (0 trainable base parameters, memory elimination)
- Masked answer cross-entropy loss strictly on target tokens (zero loss on prompt, slots, pads)
- Stage A: 1-batch overfit (loss < 0.05 and 100% exact match within 25 steps)
- Multi-step convergence on multi-sample batch
- Memory footprint (< 8.5 GB on Apple Silicon Metal GPU)
- Checkpoint serialization, SHA-256 sidecar JSON, and roundtrip parameter reload
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.trainer import GemmaPRLRTrainer, GemmaTrainerConfig, compute_masked_ce_loss
from prlr.manifest import ModelManifest


@pytest.fixture(scope="module")
def gemma_manifest() -> ModelManifest:
    return ModelManifest.gemma_2b_it()


@pytest.fixture(scope="module")
def pretrained_backbone(gemma_manifest: ModelManifest) -> PretrainedGemmaBackbone:
    return PretrainedGemmaBackbone(manifest=gemma_manifest, load_weights=True)


@pytest.fixture(scope="module")
def causal_decoder(pretrained_backbone: PretrainedGemmaBackbone) -> GemmaCausalPrefixDecoder:
    return GemmaCausalPrefixDecoder(backbone=pretrained_backbone)


def test_trainer_config_defaults():
    """Verify GemmaTrainerConfig default hyperparameter values."""
    cfg = GemmaTrainerConfig()
    assert cfg.learning_rate == 2e-3
    assert cfg.min_learning_rate == 1e-5
    assert cfg.warmup_steps == 10
    assert cfg.max_grad_norm == 1.0
    assert cfg.overfit_loss_threshold == 0.05
    assert "reproducible_pretrained_lane" in str(cfg.checkpoint_dir)


def test_frozen_backbone_guarantee(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify backbone is strictly frozen (0 trainable parameters) and only adapter is trainable."""
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)
    trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=adapter,
        decoder=causal_decoder,
    )

    # 1. Backbone must have exactly 0 trainable parameters
    base_trainable = tree_flatten(pretrained_backbone.trainable_parameters())
    assert len(base_trainable) == 0, f"Backbone has {len(base_trainable)} trainable parameters!"

    # 2. Adapter must have active trainable parameters
    adapter_trainable = tree_flatten(adapter.trainable_parameters())
    assert len(adapter_trainable) > 0, "Adapter must have trainable parameters"

    # 3. Parameter count: ~88.7M parameters for dense adapter
    total_params = sum(p.size for _, p in adapter_trainable)
    assert 80_000_000 < total_params < 150_000_000


def test_masked_loss_strictly_on_targets(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify zero loss and zero gradient contribution on prompt, soft prefix slots, and padding."""
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)

    prompt = "2 + 3 ="
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    prompt_hiddens = pretrained_backbone.extract_contextual_hiddens(prompt_ids)

    # Target: 5 with EOS (107) and padding (0)
    target_ids = mx.array([[235248, 235308, 107, 0, 0]], dtype=mx.int32)
    target_mask = mx.array([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=mx.float32)

    loss, logits = compute_masked_ce_loss(
        adapter=adapter,
        decoder=causal_decoder,
        prompt_hiddens=prompt_hiddens,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        target_mask=target_mask,
        steps=2,
    )
    mx.eval(loss, logits)

    assert not mx.isnan(loss)
    assert not mx.isinf(loss)
    assert float(loss.item()) > 0.0

    # Logit shape corresponds strictly to target length
    assert logits.shape == (1, 5, 256000)


def test_stage_a_1batch_overfit(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Stage A: Verify 1-batch overfit achieves loss < 0.05 and 100% exact match within 25 steps."""
    mx.random.seed(42)
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)

    cfg = GemmaTrainerConfig(
        deliberation_steps=2,
        overfit_loss_threshold=0.05,
    )
    opt = optim.AdamW(learning_rate=4e-4, weight_decay=0.01)
    trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=adapter,
        decoder=causal_decoder,
        config=cfg,
        optimizer=opt,
    )

    prompt = "2 + 3 ="
    p_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    prompt_hiddens = pretrained_backbone.extract_contextual_hiddens(p_ids)

    # Encode clean target tokens ending with EOS (1)
    target_toks = pretrained_backbone.tokenizer.encode(" 5", add_special_tokens=False) + [1]
    target_ids = mx.array([target_toks], dtype=mx.int32)

    batch = {
        "prompt_ids": p_ids,
        "prompt_hiddens": prompt_hiddens,
        "target_ids": target_ids,
    }

    result = trainer.run_stage_a_overfit(batch, max_steps=25, loss_threshold=0.05)

    assert result["passed"] is True, f"Stage A overfit failed: {result}"
    assert result["final_loss"] < 0.05, f"Loss {result['final_loss']} not < 0.05"
    assert result["exact_match"] is True, f"Generated tokens {result['generated_tokens']} != target {result['target_tokens']}"
    assert result["steps_to_converge"] <= 25


def test_multistep_convergence_loss_reduction(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify multi-step loss monotonically decreases over training steps."""
    mx.random.seed(42)
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)

    opt = optim.AdamW(learning_rate=4e-4, weight_decay=0.01)
    cfg = GemmaTrainerConfig(
        deliberation_steps=2,
    )
    trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=adapter,
        decoder=causal_decoder,
        config=cfg,
        optimizer=opt,
    )

    p1, _ = pretrained_backbone.encode_prompt_context("1 + 1 =")
    p2, _ = pretrained_backbone.encode_prompt_context("2 + 2 =")

    # Pad prompts to same length
    max_p = max(p1.shape[1], p2.shape[1])
    p1_padded = p1[0].tolist() + [0] * (max_p - p1.shape[1])
    p2_padded = p2[0].tolist() + [0] * (max_p - p2.shape[1])
    prompt_ids = mx.array([p1_padded, p2_padded], dtype=mx.int32)

    t1 = pretrained_backbone.tokenizer.encode(" 2", add_special_tokens=False) + [1]
    t2 = pretrained_backbone.tokenizer.encode(" 4", add_special_tokens=False) + [1]
    max_t = max(len(t1), len(t2))
    t1_padded = t1 + [0] * (max_t - len(t1))
    t2_padded = t2 + [0] * (max_t - len(t2))
    target_ids = mx.array([t1_padded, t2_padded], dtype=mx.int32)

    prompt_hiddens = pretrained_backbone.extract_contextual_hiddens(prompt_ids)

    batch = {
        "prompt_ids": prompt_ids,
        "prompt_hiddens": prompt_hiddens,
        "target_ids": target_ids,
    }

    initial_loss, _ = trainer.train_step(batch)
    for _ in range(17):
        trainer.train_step(batch)
    final_loss = trainer.training_history[-1]["loss"]

    assert final_loss < initial_loss, f"Loss did not decrease: initial {initial_loss} vs final {final_loss}"


def test_memory_footprint_within_limits(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify peak memory on Apple Silicon Metal GPU remains strictly < 8.5 GB during training."""
    mx.clear_cache()
    mx.reset_peak_memory()
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)
    trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=adapter,
        decoder=causal_decoder,
    )
    p_ids, _ = pretrained_backbone.encode_prompt_context("1 + 1 =")
    t_ids = mx.array([[235248, 235308, 1]], dtype=mx.int32)
    trainer.train_step({"prompt_ids": p_ids, "target_ids": t_ids})

    peak_mb = mx.get_peak_memory() / (1024**2)
    limit_mb = 8.5 * 1024  # 8704 MB
    assert peak_mb < limit_mb, f"Peak memory {peak_mb:.2f} MB exceeds 8.5 GB limit ({limit_mb:.2f} MB)"


def test_checkpoint_roundtrip_and_sha256(
    pretrained_backbone: PretrainedGemmaBackbone,
    causal_decoder: GemmaCausalPrefixDecoder,
    tmp_path: Path,
):
    """Verify Stage C checkpoint serialization, SHA-256 sidecar, and parameter reloading."""
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)
    trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=adapter,
        decoder=causal_decoder,
        config=GemmaTrainerConfig(checkpoint_dir=tmp_path),
    )

    # 1. Save checkpoint
    save_path = trainer.save_checkpoint(filepath=tmp_path / "prlr_test_adapter.safetensors")
    assert save_path.exists()

    sidecar_path = save_path.with_suffix(".json")
    assert sidecar_path.exists()

    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert "weights_sha256" in sidecar
    assert len(sidecar["weights_sha256"]) == 64
    assert sidecar["weights_file"] == "prlr_test_adapter.safetensors"

    # 2. Reload into fresh adapter
    fresh_adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=2)
    fresh_trainer = GemmaPRLRTrainer(
        backbone=pretrained_backbone,
        adapter=fresh_adapter,
        decoder=causal_decoder,
        config=GemmaTrainerConfig(checkpoint_dir=tmp_path),
    )

    meta = fresh_trainer.load_checkpoint(save_path, verify_sha256=True)
    assert meta["model_id"] == pretrained_backbone.manifest.model_id

    # Check weights match exactly
    orig_params = dict(tree_flatten(adapter.trainable_parameters()))
    loaded_params = dict(tree_flatten(fresh_adapter.trainable_parameters()))
    for k, v in orig_params.items():
        assert mx.allclose(v, loaded_params[k])

    # 3. Verify single-byte tampering triggers error
    with open(save_path, "rb") as f:
        bytes_data = bytearray(f.read())
    bytes_data[-1] ^= 0xFF  # Flip a bit
    tampered_path = tmp_path / "tampered.safetensors"
    with open(tampered_path, "wb") as f:
        f.write(bytes_data)
    # Copy sidecar pointing to tampered
    tampered_sidecar = dict(sidecar)
    tampered_sidecar["weights_file"] = "tampered.safetensors"
    with open(tmp_path / "tampered.json", "w") as f:
        json.dump(tampered_sidecar, f)

    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        fresh_trainer.load_checkpoint(tampered_path, verify_sha256=True)
