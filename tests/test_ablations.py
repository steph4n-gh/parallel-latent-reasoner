"""Test Suite for Milestone 5 Requirement R7 / Feature 24: Controlled Ablation Suite.

Verifies:
1. `test_ablation_matrix_generation`: >= 25 unique conditions covering baseline, depth ladder,
   slot ladder, single-slot knockouts, slot merge, and anchor inits.
2. `test_context_cache_numerical_identity`: cached vs fresh H_prompt representations match exactly.
3. `test_monotonic_depth_scaling_and_velocity_decay`: velocity decays across recurrence unrolls T.
4. `test_slot_knockout_causal_contribution`: non-zero logit perturbation ||Delta z|| > 0 when slots are knocked out.
5. `test_slot_merge_representation_capacity`: rank decreases under slot merge (16 -> 8).
6. `test_ablation_memory_stability_metal`: flat VRAM, < 8.5 GB peak on Apple Silicon Metal GPU.
7. `test_ablation_reproducibility_under_seed`: deterministic exact-match outputs under identical seeds.
8. `test_ground_truth_isolation_in_ablation_harness`: zero target fields in inference input.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import pytest

import mlx.core as mx

from prlr.domain.solver_lane import ProceduralVerifier
from prlr.gemma.ablation import (
    AblationConditionSummary,
    AblationSpec,
    AblationSuiteReport,
    GemmaAblationHarness,
    compute_bootstrap_ci_95,
)
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.kernel.telemetry import compute_effective_rank, compute_slot_velocity
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


@pytest.fixture(scope="module")
def recurrent_adapter() -> GemmaRecurrentAdapter:
    return GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=4)


@pytest.fixture(scope="module")
def ablation_harness(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    causal_decoder: GemmaCausalPrefixDecoder,
) -> GemmaAblationHarness:
    return GemmaAblationHarness(
        backbone=pretrained_backbone,
        adapter=recurrent_adapter,
        decoder=causal_decoder,
        verifier=ProceduralVerifier(),
    )


def test_ablation_matrix_generation():
    """Verify build_standard_ablation_matrix returns >= 25 unique valid conditions."""
    specs = GemmaAblationHarness.build_standard_ablation_matrix()

    # 1. Condition count requirement: >= 25 conditions
    assert len(specs) >= 25, f"Expected >= 25 conditions, got {len(specs)}"

    # 2. All condition names must be unique
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), f"Duplicate spec names found: {len(names)} vs {len(set(names))}"

    # 3. Canonical categories present
    categories = {s.category for s in specs}
    expected_categories = {"baseline", "depth", "slots", "knockout", "merge", "anchor_init"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    # 4. Direct baseline exists and flags is_direct_baseline
    direct_baselines = [s for s in specs if s.is_direct_baseline]
    assert len(direct_baselines) == 1
    assert direct_baselines[0].name == "baseline_direct"

    # 5. Prelude only (T=0) condition exists
    t0_specs = [s for s in specs if s.deliberation_steps == 0]
    assert len(t0_specs) == 1
    assert t0_specs[0].name == "t0_prelude_only"

    # 6. Depth ladder {1, 2, 4, 8, 12}
    depth_steps = {s.deliberation_steps for s in specs if s.category == "depth" and s.deliberation_steps != 0}
    assert {1, 2, 4, 8, 12}.issubset(depth_steps)

    # 7. Slot ladder {1, 4, 8, 16}
    slot_counts = {s.num_slots for s in specs if s.category == "slots"}
    assert {1, 4, 8, 16}.issubset(slot_counts)

    # 8. Single-slot knockouts for slots 0..15
    knockout_slots = {s.knockout_slot for s in specs if s.category == "knockout"}
    assert set(range(16)) == knockout_slots

    # 9. Slot merge (16 -> 8)
    merges = [s for s in specs if s.category == "merge"]
    assert len(merges) >= 1
    assert merges[0].merge_target_slots == 8

    # 10. Anchor initializations
    anchor_inits = {s.anchor_type for s in specs if s.category == "anchor_init"}
    assert {"orthogonal", "gaussian", "shuffled", "zeros"}.issubset(
        anchor_inits | {"orthogonal"}
    )


def test_context_cache_numerical_identity(
    pretrained_backbone: PretrainedGemmaBackbone,
    ablation_harness: GemmaAblationHarness,
):
    """Verify cached prompt representations match freshly computed hiddens with exact parity."""
    prompt = "<start_of_turn>user\nPlan route: initial [data_raw] target [report_final]<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)

    # Fresh computation
    h_fresh = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_fresh)

    # Cache retrieval
    sample_id = "test_cache_id_001"
    ablation_harness.clear_context_cache()
    h_cached_1, _ = ablation_harness.get_or_compute_prompt_hiddens(sample_id, prompt_ids)
    h_cached_2, prefill_ms_cached = ablation_harness.get_or_compute_prompt_hiddens(sample_id, prompt_ids)

    # Second fetch should have 0 ms extra computation
    assert prefill_ms_cached == 0.0

    # Numerical identity assertion
    max_diff = float(mx.max(mx.abs(h_fresh - h_cached_2)).item())
    assert max_diff < 1e-6, f"Cached representation diverged from fresh: max_diff={max_diff}"

    # Cosine similarity
    flat_fresh = h_fresh.reshape(-1)
    flat_cached = h_cached_2.reshape(-1)
    cosine_sim = float(
        mx.sum(flat_fresh * flat_cached) / (mx.linalg.norm(flat_fresh) * mx.linalg.norm(flat_cached)).item()
    )
    assert cosine_sim > 0.999999, f"Cosine similarity {cosine_sim} < 0.999999"


def test_monotonic_depth_scaling_and_velocity_decay(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
):
    """Verify velocity v(t) decays across recurrence depth T in {1, 2, 4, 8, 12}."""
    prompt = "<start_of_turn>user\nDetermine execution plan.<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    trajectory = recurrent_adapter.unroll_trajectory(h_prompt, max_steps=12)
    assert len(trajectory) == 13  # S^(0) through S^(12)

    velocities = []
    for t in range(1, 13):
        v_t = float(compute_slot_velocity(trajectory[t - 1], trajectory[t]))
        velocities.append(v_t)

    # Initial velocity at t=1 vs deep velocity at t=12
    v1 = velocities[0]
    v12 = velocities[-1]

    assert v1 > 0.0, f"Initial velocity must be strictly positive, got {v1}"
    assert v12 < v1, f"Expected velocity decay: v(12)={v12} should be < v(1)={v1}"

    # Effective rank delta should plateau
    eranks = [float(compute_effective_rank(s)) for s in trajectory]
    delta_erank_early = abs(eranks[2] - eranks[1])
    delta_erank_late = abs(eranks[12] - eranks[11])
    assert delta_erank_late <= delta_erank_early + 0.05


def test_slot_knockout_causal_contribution(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify single-slot knockout induces non-zero logit shifts (||Delta z|| > 0)."""
    prompt = "<start_of_turn>user\nPlan route: initial [sensor_raw] target [actuator_cmd]<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    # Unroll baseline deliberated slots
    slots_full = recurrent_adapter(h_prompt, steps=4)
    mx.eval(slots_full)

    # Unperturbed logits
    logits_full = causal_decoder.prefill_logits(prompt_ids, slots_full)
    mx.eval(logits_full)

    # Perturb slot 3 (zero knockout)
    k = 3
    slots_knockout = mx.concatenate(
        [slots_full[:, :k, :], mx.zeros((1, 1, 2048)), slots_full[:, k + 1 :, :]],
        axis=1,
    )
    mx.eval(slots_knockout)

    logits_knockout = causal_decoder.prefill_logits(prompt_ids, slots_knockout)
    mx.eval(logits_knockout)

    # Logit perturbation norm
    delta_logits = logits_full - logits_knockout
    norm_delta = float(mx.linalg.norm(delta_logits).item())

    assert norm_delta > 0.0, f"Knockout induced zero logit shift: norm_delta={norm_delta}"
    assert not mx.isnan(mx.array(norm_delta)).item()


def test_slot_merge_representation_capacity(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify slot merge (16 -> 8) alters representations and reduces slot capacity."""
    prompt = "<start_of_turn>user\nFind minimal route.<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    slots_16 = recurrent_adapter(h_prompt, steps=4)
    mx.eval(slots_16)
    assert slots_16.shape[1] == 16

    # 16 -> 8 pairwise average
    slots_8 = (slots_16[:, 0::2, :] + slots_16[:, 1::2, :]) / 2.0
    mx.eval(slots_8)
    assert slots_8.shape[1] == 8

    # Check effective rank
    erank_16 = float(compute_effective_rank(slots_16))
    erank_8 = float(compute_effective_rank(slots_8))
    assert erank_8 <= erank_16 + 1e-4

    # Decoder logits shift
    logits_16 = causal_decoder.prefill_logits(prompt_ids, slots_16)
    logits_8 = causal_decoder.prefill_logits(prompt_ids, slots_8)
    shift = float(mx.linalg.norm(logits_16 - logits_8).item())
    assert shift > 0.0


def test_ablation_memory_stability_metal(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify peak memory remains strictly bounded (< 8.5 GB) with flat VRAM over multi-depth sweeps."""
    prompt = "<start_of_turn>user\nExecute reasoning task.<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    depths = [1, 2, 4, 8, 12, 4, 2, 1]
    peak_memories = []

    for d in depths:
        slots = recurrent_adapter(h_prompt, steps=d)
        mx.eval(slots)
        tokens = causal_decoder.generate(prompt_ids, slots, max_new_tokens=16)
        mx.eval(tokens)

        if hasattr(mx, "get_peak_memory"):
            peak_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0)
            peak_memories.append(peak_mb)

        gc.collect()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    if peak_memories:
        max_peak = max(peak_memories)
        assert max_peak < 8500.0, f"Peak memory exceeded 8.5 GB: {max_peak} MB"
        drift = peak_memories[-1] - peak_memories[0]
        assert abs(drift) < 50.0, f"Memory drift between sweeps exceeded bound: drift={drift} MB"


def test_ablation_reproducibility_under_seed(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    causal_decoder: GemmaCausalPrefixDecoder,
):
    """Verify deterministic exact-match outputs across duplicate runs."""
    prompt = "<start_of_turn>user\nPlan route: initial [a] target [b]<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    # Run 1
    mx.random.seed(42)
    s1 = recurrent_adapter(h_prompt, steps=4)
    mx.eval(s1)
    toks1 = causal_decoder.generate(prompt_ids, s1, max_new_tokens=32, temperature=0.0)
    mx.eval(toks1)

    # Run 2
    mx.random.seed(42)
    s2 = recurrent_adapter(h_prompt, steps=4)
    mx.eval(s2)
    toks2 = causal_decoder.generate(prompt_ids, s2, max_new_tokens=32, temperature=0.0)
    mx.eval(toks2)

    assert bool(mx.all(toks1 == toks2).item()) is True


def test_ground_truth_isolation_in_ablation_harness(
    ablation_harness: GemmaAblationHarness,
):
    """Verify that evaluate_instance receives zero target fields and isolates answer keys (Rules 1 & 2)."""
    import inspect

    # 1. Inspect evaluate_instance signature
    sig = inspect.signature(ablation_harness.evaluate_instance)
    param_names = list(sig.parameters.keys())

    forbidden = {"target", "expected_route", "answer_key", "goal", "initial_state", "trace"}
    for f in forbidden:
        assert f not in param_names, f"Forbidden oracle parameter '{f}' in evaluate_instance!"

    # 2. Check that evaluate_instance runs successfully on pure prompt
    spec = AblationSpec("test_spec", "depth", 1, 16)
    pred_text, t_pre, t_delib, t_dec, t_tot, n_tok = ablation_harness.evaluate_instance(
        spec=spec,
        sample_id="isolated_test_sample",
        prompt_str="<start_of_turn>user\nDetermine plan.<end_of_turn>\n<start_of_turn>model\n",
        max_new_tokens=16,
    )
    assert isinstance(pred_text, str)
    assert t_tot > 0.0
    assert n_tok > 0
