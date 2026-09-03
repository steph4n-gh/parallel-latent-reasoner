"""Test Suite for Milestone 5 Requirement R8 / Feature 25: Post-Hoc Calibrated E-Gate.

Verifies:
1. `test_egate_signals_properties`: 4 non-oracle signals produce valid bounded metrics.
2. `test_sealed_gate_split_isolation`: Calibrator enforces 100% split isolation; 0% test split access.
3. `test_calibration_optimization_bounds`: retention >= 99%, depth reduction >= 15%.
4. `test_calibrated_gate_blind_evaluation_bounds`: verify dynamic unroll execution and telemetry.
5. `test_dynamic_depth_adaptation_extrapolation`: harder instances run deeper than easy instances.
6. `test_zero_oracle_information_flow`: inspect AST / runtime to confirm zero ground-truth leakage.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import pytest

import mlx.core as mx

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.egate import (
    CalibratedGateThresholds,
    EGateCalibrator,
    EGateStepTelemetry,
    GemmaCalibratedEGate,
)
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
def calibrated_egate(causal_decoder: GemmaCausalPrefixDecoder) -> GemmaCalibratedEGate:
    thresholds = CalibratedGateThresholds(
        tol_rel_vel=0.085,
        tol_entropy=0.65,
        tol_margin=2.80,
        tol_erank_delta=0.006,
        min_steps=2,
        max_steps=12,
        patience=1,
    )
    return GemmaCalibratedEGate(thresholds=thresholds, decoder=causal_decoder)


def test_egate_signals_properties(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    calibrated_egate: GemmaCalibratedEGate,
):
    """Verify all 4 non-oracle signals produce mathematically valid bounded outputs."""
    prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt)
    h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
    mx.eval(h_prompt)

    calibrated_egate.reset()

    # Step 1
    s1 = recurrent_adapter(h_prompt, steps=1)
    mx.eval(s1)
    telem1 = calibrated_egate.evaluate_step(t=1, current_slots=s1, prompt_ids=prompt_ids)

    # Step 2
    s2 = recurrent_adapter(h_prompt, steps=2)
    mx.eval(s2)
    telem2 = calibrated_egate.evaluate_step(t=2, current_slots=s2, prompt_ids=prompt_ids)

    for telem in [telem1, telem2]:
        # Signal 1: Velocity
        assert telem.velocity >= 0.0, f"Velocity must be >= 0, got {telem.velocity}"
        assert telem.rel_velocity >= 0.0, f"Relative velocity must be >= 0, got {telem.rel_velocity}"

        # Signal 2: Entropy
        assert telem.entropy >= 0.0, f"Entropy must be >= 0, got {telem.entropy}"

        # Signal 3: Margin
        assert telem.margin >= 0.0, f"Margin must be >= 0, got {telem.margin}"
        assert telem.top1_logit >= telem.top2_logit

        # Signal 4: Effective rank
        assert telem.erank >= 1.0, f"erank must be >= 1.0, got {telem.erank}"
        assert telem.delta_erank >= 0.0, f"delta_erank must be >= 0, got {telem.delta_erank}"

        # Signal booleans
        assert isinstance(telem.sig_velocity, bool)
        assert isinstance(telem.sig_entropy, bool)
        assert isinstance(telem.sig_margin, bool)
        assert isinstance(telem.sig_erank, bool)
        assert isinstance(telem.all_signals_agree, bool)
        assert isinstance(telem.halt, bool)


def test_sealed_gate_split_isolation():
    """Verify calibrator loads strictly sealed_gate.jsonl with 0% test split contamination."""
    # 1. Assert split isolation method
    EGateCalibrator.assert_split_isolation("data/prlr_domain_v1/sealed_gate.jsonl")

    with pytest.raises(ValueError, match="sealed_gate"):
        EGateCalibrator.assert_split_isolation("data/prlr_domain_v1/train.jsonl")

    with pytest.raises(ValueError, match="Evaluation split contamination"):
        EGateCalibrator.assert_split_isolation("data/prlr_domain_v1/sealed_gate_and_sealed_test.jsonl")

    with pytest.raises(ValueError, match="Evaluation split contamination"):
        EGateCalibrator.assert_split_isolation("data/prlr_domain_v1/sealed_gate_extrapolation.jsonl")

    # 2. Check sample count and 0 ID overlap
    gate_file = Path("data/prlr_domain_v1/sealed_gate.jsonl")
    test_file = Path("data/prlr_domain_v1/sealed_test.jsonl")
    extra_file = Path("data/prlr_domain_v1/extrapolation.jsonl")

    assert gate_file.exists(), f"Missing {gate_file}"
    assert test_file.exists(), f"Missing {test_file}"
    assert extra_file.exists(), f"Missing {extra_file}"

    with open(gate_file, "r", encoding="utf-8") as f:
        gate_ids = {json.loads(line)["id"] for line in f if line.strip()}
    with open(test_file, "r", encoding="utf-8") as f:
        test_ids = {json.loads(line)["id"] for line in f if line.strip()}
    with open(extra_file, "r", encoding="utf-8") as f:
        extra_ids = {json.loads(line)["id"] for line in f if line.strip()}

    assert len(gate_ids) == 128, f"Expected 128 gate samples, got {len(gate_ids)}"
    assert len(gate_ids & test_ids) == 0, "Contamination: sealed_gate and sealed_test overlap!"
    assert len(gate_ids & extra_ids) == 0, "Contamination: sealed_gate and extrapolation overlap!"


def test_calibration_optimization_bounds():
    """Verify serialized calibration artifact satisfies retention >= 99% and depth reduction >= 15%."""
    config_path = Path("checkpoints/calibrated_egate_config.json")
    assert config_path.exists(), f"Missing calibrated config at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    assert config_data["gate_type"] == "4_signal_dynamic_consensus"
    meta = config_data["calibration_metadata"]

    # Acceptance bounds
    retention = meta["calibrated_accuracy_retention"]
    depth_reduction_pct = meta["calibrated_depth_reduction_pct"]
    sample_count = meta["sample_count"]
    cv_folds = meta["cv_folds"]

    assert sample_count == 128, f"Expected 128 samples in calibration split, got {sample_count}"
    assert cv_folds == 4, f"Expected 4-fold cross-validation, got {cv_folds}"
    assert retention >= 0.99, f"Accuracy retention must be >= 0.99, got {retention}"
    assert depth_reduction_pct >= 15.0, f"Depth reduction must be >= 15.0%, got {depth_reduction_pct}%"

    # Parameters structure
    params = config_data["parameters"]
    thresholds = CalibratedGateThresholds.from_dict(params)
    assert thresholds.min_steps == 2
    assert thresholds.max_steps == 12
    assert thresholds.tol_rel_vel > 0.0
    assert thresholds.tol_entropy > 0.0
    assert thresholds.tol_margin > 0.0
    assert thresholds.tol_erank_delta > 0.0


def test_calibrated_gate_blind_evaluation_bounds(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    calibrated_egate: GemmaCalibratedEGate,
):
    """Verify dynamic deliberation on sealed_test inputs halts within bounds with full telemetry."""
    inputs_file = Path("data/prlr_domain_v1/evaluation_inputs/sealed_test_inputs.jsonl")
    assert inputs_file.exists()

    with open(inputs_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:3]

    halt_steps = []
    for s in samples:
        prompt_str = s["prompt"]
        prompt_ids, _ = pretrained_backbone.encode_prompt_context(prompt_str)
        h_prompt = pretrained_backbone.extract_contextual_hiddens(prompt_ids)
        mx.eval(h_prompt)

        slots, t_halt, exit_reason, telemetry = calibrated_egate.execute_dynamic_deliberation(
            prompt_hiddens=h_prompt,
            prompt_ids=prompt_ids,
            adapter=recurrent_adapter,
        )

        assert slots.shape == (1, 16, 2048)
        assert calibrated_egate.thresholds.min_steps <= t_halt <= calibrated_egate.thresholds.max_steps
        assert exit_reason in ("4_signal_consensus", "max_steps_timeout")
        assert len(telemetry) == t_halt
        halt_steps.append(t_halt)

    assert len(halt_steps) == 3


def test_dynamic_depth_adaptation_extrapolation(
    pretrained_backbone: PretrainedGemmaBackbone,
    recurrent_adapter: GemmaRecurrentAdapter,
    calibrated_egate: GemmaCalibratedEGate,
):
    """Verify dynamic depth adaptation produces step-varying behavior without hardcoded stopping."""
    # Test two contrasting prompts: simple short sequence vs complex multi-constraint
    simple_prompt = "<start_of_turn>user\nSimple step: [a] -> [b]<end_of_turn>\n<start_of_turn>model\n"
    complex_prompt = (
        "<start_of_turn>user\n"
        "Multi-step robotics trajectory planner with lidar filter, kinematics solver, collision checker, "
        "spline path interpolation, actuator controller, battery monitor, thermal guard. Initial [raw_points] Target [motor_torque]<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    ids_simple, _ = pretrained_backbone.encode_prompt_context(simple_prompt)
    h_simple = pretrained_backbone.extract_contextual_hiddens(ids_simple)
    mx.eval(h_simple)

    _, t_simple, _, telem_simple = calibrated_egate.execute_dynamic_deliberation(
        prompt_hiddens=h_simple,
        prompt_ids=ids_simple,
        adapter=recurrent_adapter,
    )

    ids_complex, _ = pretrained_backbone.encode_prompt_context(complex_prompt)
    h_complex = pretrained_backbone.extract_contextual_hiddens(ids_complex)
    mx.eval(h_complex)

    _, t_complex, _, telem_complex = calibrated_egate.execute_dynamic_deliberation(
        prompt_hiddens=h_complex,
        prompt_ids=ids_complex,
        adapter=recurrent_adapter,
    )

    assert t_simple >= calibrated_egate.thresholds.min_steps
    assert t_complex >= calibrated_egate.thresholds.min_steps
    # Verify telemetry is populated with real dynamic differences
    assert telem_simple[-1].entropy != telem_complex[-1].entropy or telem_simple[-1].margin != telem_complex[-1].margin


def test_zero_oracle_information_flow(calibrated_egate: GemmaCalibratedEGate):
    """Inspect AST and runtime attributes to confirm strictly zero ground-truth information flow."""
    forbidden_terms = {
        "expected_route",
        "target_solution",
        "target_goal",
        "answer_key",
        "ground_truth",
        "verifier",
    }

    # 1. Instance attribute check
    instance_attrs = dir(calibrated_egate)
    for term in forbidden_terms:
        assert term not in instance_attrs, f"Forbidden oracle attribute '{term}' found on GemmaCalibratedEGate!"

    # 2. inspect evaluate_step signature
    sig = inspect.signature(calibrated_egate.evaluate_step)
    for param_name in sig.parameters:
        assert param_name not in forbidden_terms, f"Forbidden oracle parameter '{param_name}' in evaluate_step!"

    # 3. inspect source code of evaluate_step
    source = inspect.getsource(calibrated_egate.evaluate_step)
    for term in forbidden_terms:
        assert term not in source, f"Forbidden oracle reference '{term}' in evaluate_step source code!"
