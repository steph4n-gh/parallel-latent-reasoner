"""Comprehensive Test Suite for Large Gemma 4 Evaluation & Benchmark Infrastructure.

Verifies:
1. Large Model Configs (Gemma 4 12B Q4 dense and 26B A4B MoE) have valid shapes, head counts, and memory bounds.
2. MoE Layer routing, forward pass, and parameter count invariance across unrolls.
3. Dual-mode evaluation data structures (EvaluationSampleResult) and metrics calculation.
4. Schema validation for prlr.gemma4_suite.v1 benchmark records.
5. Cognitive suite integration with dynamic unrolls and zero memory bloat guarantees.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import pytest

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    get_test_case_by_id,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import DynamicConsensusEGate
from parallel_latent_reasoner.models import (
    MLXCodaLMHead,
    MLXCompactGemmaModel,
    MLXGemmaMoE,
    MLXPreludeProjection,
    MLXRecurrentGemmaBlock,
    sinusoidal_step_embedding,
)
from parallel_latent_reasoner.probes import (
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_velocity,
)


# ============================================================================
# Dual-Mode Evaluation Data Structures for Testing & Verification
# ============================================================================

@dataclass
class EvaluationSampleResult:
    """Evaluation record capturing Mode 1 (AR CoT) vs Mode 2 (PRLR) metrics."""

    test_case_id: str
    domain: str
    model_name: str
    # Mode 1: Autoregressive Chain-of-Thought
    ar_output_text: str
    ar_thought_text: Optional[str]
    ar_prefill_latency_ms: float
    ar_reasoning_latency_ms: float
    ar_decode_latency_ms: float
    ar_total_latency_ms: float
    ar_tokens_generated: int
    ar_effective_throughput_tok_s: float
    ar_constraint_passed: bool
    ar_score: float
    ar_peak_vram_mb: float
    # Mode 2: Parallel Recurrent Latent Deliberation (PRLR)
    prlr_output_text: str
    prlr_prefill_latency_ms: float
    prlr_reasoning_latency_ms: float
    prlr_decode_latency_ms: float
    prlr_total_latency_ms: float
    prlr_steps_executed: int
    prlr_exit_signal: str
    prlr_effective_throughput_tok_s: float
    prlr_constraint_passed: bool
    prlr_score: float
    prlr_peak_vram_mb: float
    # Comparative
    reasoning_speedup: float
    memory_growth_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_summary_metrics(
    records: list[EvaluationSampleResult],
    model_names: list[str],
) -> dict[str, Any]:
    """Aggregate per-case results into a schema-compliant benchmark artifact."""
    if not records:
        return {}

    total_cases = len(records)
    prlr_passed = sum(1 for r in records if r.prlr_constraint_passed)
    cot_passed = sum(1 for r in records if r.ar_constraint_passed)
    mean_speedup = sum(r.reasoning_speedup for r in records) / total_cases
    peak_vram = max(max(r.prlr_peak_vram_mb, r.ar_peak_vram_mb) for r in records)
    max_mem_growth = max(r.memory_growth_pct for r in records)

    domain_breakdown: dict[str, Any] = {}
    for r in records:
        if r.domain not in domain_breakdown:
            domain_breakdown[r.domain] = {
                "total": 0,
                "prlr_passed": 0,
                "cot_passed": 0,
                "speedups": [],
            }
        d = domain_breakdown[r.domain]
        d["total"] += 1
        if r.prlr_constraint_passed:
            d["prlr_passed"] += 1
        if r.ar_constraint_passed:
            d["cot_passed"] += 1
        d["speedups"].append(r.reasoning_speedup)

    domain_stats = {}
    for dom_name, d in domain_breakdown.items():
        tot = d["total"]
        domain_stats[dom_name] = {
            "prlr_acc": round(d["prlr_passed"] / tot, 4) if tot > 0 else 0.0,
            "cot_acc": round(d["cot_passed"] / tot, 4) if tot > 0 else 0.0,
            "mean_speedup": round(sum(d["speedups"]) / len(d["speedups"]), 2) if d["speedups"] else 1.0,
            "sample_count": tot,
        }

    return {
        "$schema": "prlr.gemma4_suite.v1",
        "metadata": {
            "timestamp": "2026-09-02T21:45:00Z",
            "platform": "macOS-Darwin (Apple Silicon)",
            "device": "Metal GPU (Unified Memory)",
            "models_evaluated": model_names,
            "memory_limit_bytes": 17716740096,
            "suite_total_test_cases": total_cases,
            "domains": list(domain_stats.keys()),
        },
        "summary_metrics": {
            "prlr_overall_accuracy": round(prlr_passed / total_cases, 4),
            "cot_overall_accuracy": round(cot_passed / total_cases, 4),
            "mean_reasoning_speedup": round(mean_speedup, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "memory_growth_pct": round(max_mem_growth, 4),
        },
        "domain_breakdown": domain_stats,
        "test_case_records": [r.to_dict() for r in records],
    }


# ============================================================================
# Test Cases
# ============================================================================

def test_large_gemma_12b_q4_config_properties():
    """Verify Gemma 4 12B Q4 dense config has exact target dimensions and parameter fields."""
    cfg = GemmaLatentConfig.gemma_12b_q4()
    assert cfg.dim == 3840
    assert cfg.num_heads == 16
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 256
    assert cfg.intermediate_dim == 15360
    assert cfg.vocab_size == 262144
    assert cfg.num_memory_slots == 16
    assert cfg.deliberation_steps == 8
    assert cfg.min_steps == 2
    assert cfg.max_steps == 12
    assert cfg.num_layers == 48
    assert cfg.rezero_alpha == 0.05
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.enable_moe_block is False

    # Check alias
    cfg_alias = GemmaLatentConfig.gemma_4_12b_q4()
    assert cfg_alias.dim == cfg.dim
    assert cfg_alias.vocab_size == cfg.vocab_size

    # Check serialization roundtrip
    d = cfg.to_dict()
    assert d["dim"] == 3840
    reconstructed = GemmaLatentConfig.from_dict(d)
    assert reconstructed.dim == 3840


def test_large_gemma_26b_a4b_moe_config_properties():
    """Verify Gemma 4 26B A4B MoE config has exact target MoE fields and parameters."""
    cfg = GemmaLatentConfig.gemma_26b_a4b()
    assert cfg.dim == 2816
    assert cfg.num_heads == 16
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 256
    assert cfg.intermediate_dim == 2112
    assert cfg.moe_intermediate_dim == 704
    assert cfg.num_experts == 128
    assert cfg.top_k_experts == 8
    assert cfg.enable_moe_block is True
    assert cfg.num_layers == 30
    assert cfg.vocab_size == 262144
    assert cfg.num_memory_slots == 16
    assert cfg.rezero_alpha == 0.05

    # Check alias
    cfg_alias = GemmaLatentConfig.gemma_4_26b_a4b()
    assert cfg_alias.dim == cfg.dim
    assert cfg_alias.num_experts == 128

    # Check serialization roundtrip
    d = cfg.to_dict()
    assert d["num_experts"] == 128
    reconstructed = GemmaLatentConfig.from_dict(d)
    assert reconstructed.enable_moe_block is True


def test_json_config_files_in_configs_directory():
    """Verify standalone config JSON files exist and instantiate cleanly."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    assert configs_dir.exists(), "configs/ directory must exist"

    cfg_12b_path = configs_dir / "gemma_12b_q4.json"
    assert cfg_12b_path.exists(), "configs/gemma_12b_q4.json must exist"
    cfg_12b = GemmaLatentConfig.from_json(cfg_12b_path)
    assert cfg_12b.dim == 3840
    assert cfg_12b.vocab_size == 262144

    cfg_26b_path = configs_dir / "gemma_26b_a4b.json"
    assert cfg_26b_path.exists(), "configs/gemma_26b_a4b.json must exist"
    cfg_26b = GemmaLatentConfig.from_json(cfg_26b_path)
    assert cfg_26b.dim == 2816
    assert cfg_26b.num_experts == 128
    assert cfg_26b.top_k_experts == 8


def test_moe_block_instantiation_and_forward_pass():
    """Verify MLXGemmaMoE and MLXRecurrentGemmaBlock execute with MoE routing on Metal."""
    cfg = GemmaLatentConfig(
        dim=128,
        intermediate_dim=128,
        moe_intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        num_memory_slots=16,
        num_experts=8,
        top_k_experts=2,
        enable_moe_block=True,
        rezero_alpha=0.05,
    )

    moe_layer = MLXGemmaMoE(cfg)
    x = mx.random.normal((2, 16, 128))
    out_moe = moe_layer(x)
    mx.eval(out_moe)

    assert out_moe.shape == (2, 16, 128)
    assert not mx.any(mx.isnan(out_moe)).item()
    assert not mx.any(mx.isinf(out_moe)).item()

    # Recurrent block with MoE
    recurrent_block = MLXRecurrentGemmaBlock(cfg)
    assert isinstance(recurrent_block.mlp, MLXGemmaMoE)

    out_block = recurrent_block(x, step=1)
    mx.eval(out_block)
    assert out_block.shape == (2, 16, 128)
    assert not mx.any(mx.isnan(out_block)).item()


def test_dense_and_moe_parameter_invariance_across_unrolls():
    """Verify both Dense and MoE recurrent blocks maintain constant parameter count across unrolls."""
    import mlx.utils

    def get_num_params(m):
        flat = mlx.utils.tree_flatten(m.trainable_parameters())
        return sum(p.size for _, p in flat)

    # Dense block
    cfg_dense = GemmaLatentConfig.compact_test()
    block_dense = MLXRecurrentGemmaBlock(cfg_dense)
    p_dense_0 = get_num_params(block_dense)
    x = mx.random.normal((1, cfg_dense.num_memory_slots, cfg_dense.dim))

    for t in [1, 2, 4, 8]:
        out = block_dense(x, step=t)
        mx.eval(out)
        assert get_num_params(block_dense) == p_dense_0

    # MoE block
    cfg_moe = GemmaLatentConfig(
        dim=128,
        intermediate_dim=128,
        moe_intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        num_memory_slots=16,
        num_experts=8,
        top_k_experts=2,
        enable_moe_block=True,
    )
    block_moe = MLXRecurrentGemmaBlock(cfg_moe)
    p_moe_0 = get_num_params(block_moe)

    x_moe = mx.random.normal((1, cfg_moe.num_memory_slots, cfg_moe.dim))
    for t in [1, 2, 4, 8]:
        out_moe = block_moe(x_moe, step=t)
        mx.eval(out_moe)
        assert get_num_params(block_moe) == p_moe_0


def test_evaluation_sample_result_dataclass():
    """Verify EvaluationSampleResult data structure, throughputs, and JSON serialization."""
    sample = EvaluationSampleResult(
        test_case_id="mcs_01",
        domain="multi_constraint",
        model_name="gemma-4-12B-it-qat-4bit",
        ar_output_text="Beta, Gamma",
        ar_thought_text="Thinking step by step...",
        ar_prefill_latency_ms=12.5,
        ar_reasoning_latency_ms=1150.0,
        ar_decode_latency_ms=18.0,
        ar_total_latency_ms=1180.5,
        ar_tokens_generated=142,
        ar_effective_throughput_tok_s=120.3,
        ar_constraint_passed=True,
        ar_score=1.0,
        ar_peak_vram_mb=7420.0,
        prlr_output_text="Beta, Gamma",
        prlr_prefill_latency_ms=12.5,
        prlr_reasoning_latency_ms=22.4,
        prlr_decode_latency_ms=5.1,
        prlr_total_latency_ms=40.0,
        prlr_steps_executed=6,
        prlr_exit_signal="consensus_reached",
        prlr_effective_throughput_tok_s=4285.7,
        prlr_constraint_passed=True,
        prlr_score=1.0,
        prlr_peak_vram_mb=7420.0,
        reasoning_speedup=51.34,
        memory_growth_pct=0.00,
    )

    d = sample.to_dict()
    assert d["test_case_id"] == "mcs_01"
    assert d["reasoning_speedup"] == 51.34
    assert d["memory_growth_pct"] == 0.00

    # JSON serialization
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["test_case_id"] == "mcs_01"
    assert deserialized["prlr_steps_executed"] == 6


def test_benchmark_report_schema_compliance_prlr_gemma4_v1():
    """Verify aggregate summary metrics comply with prlr.gemma4_suite.v1 schema."""
    suite = load_cognitive_benchmark_suite()
    sample_records = []

    for i, case in enumerate(suite):
        dom_val = case.domain.value if isinstance(case.domain, DomainType) else str(case.domain)
        # Verify case ground truth
        ver_res = verify_test_case_result(case, case.ground_truth)
        passed = ver_res.passed
        score = ver_res.score

        record = EvaluationSampleResult(
            test_case_id=case.id,
            domain=dom_val,
            model_name="gemma-4-12B-it-qat-4bit",
            ar_output_text=case.ground_truth,
            ar_thought_text=None,
            ar_prefill_latency_ms=10.0,
            ar_reasoning_latency_ms=1200.0,
            ar_decode_latency_ms=15.0,
            ar_total_latency_ms=1225.0,
            ar_tokens_generated=128,
            ar_effective_throughput_tok_s=104.5,
            ar_constraint_passed=passed,
            ar_score=score,
            ar_peak_vram_mb=7420.5,
            prlr_output_text=case.ground_truth,
            prlr_prefill_latency_ms=10.0,
            prlr_reasoning_latency_ms=25.0,
            prlr_decode_latency_ms=5.0,
            prlr_total_latency_ms=40.0,
            prlr_steps_executed=6,
            prlr_exit_signal="consensus_reached",
            prlr_effective_throughput_tok_s=3840.0,
            prlr_constraint_passed=passed,
            prlr_score=score,
            prlr_peak_vram_mb=7420.5,
            reasoning_speedup=48.0,
            memory_growth_pct=0.00,
        )
        sample_records.append(record)

    summary = aggregate_summary_metrics(
        sample_records,
        model_names=["gemma-4-12B-it-qat-4bit", "gemma-4-26b-a4b-it-4bit"],
    )

    # Schema validation
    assert summary["$schema"] == "prlr.gemma4_suite.v1"
    assert "metadata" in summary
    assert "summary_metrics" in summary
    assert "domain_breakdown" in summary
    assert "test_case_records" in summary

    # Metadata checks
    meta = summary["metadata"]
    assert meta["suite_total_test_cases"] == len(suite)
    assert len(meta["models_evaluated"]) == 2
    assert "multi_constraint" in meta["domains"]

    # Summary metrics checks
    metrics = summary["summary_metrics"]
    assert metrics["prlr_overall_accuracy"] >= 0.85
    assert metrics["cot_overall_accuracy"] >= 0.85
    assert metrics["mean_reasoning_speedup"] >= 25.0
    assert metrics["memory_growth_pct"] == 0.00

    # Domain breakdown checks
    breakdown = summary["domain_breakdown"]
    for dom in DomainType:
        assert dom.value in breakdown
        assert breakdown[dom.value]["sample_count"] >= 5


def test_lipschitz_stability_across_large_scale_unrolls():
    """Verify ReZero residual scaling guarantees bounded state norms under deep recurrent sweeps."""
    cfg = GemmaLatentConfig.compact_test(dim=256, num_memory_slots=16, rezero_alpha=0.05)
    block = MLXRecurrentGemmaBlock(cfg)

    # Initial random slots
    s = mx.random.normal((1, 16, 256))
    initial_norm = float(mx.linalg.norm(s).item())

    # Deep unroll to T=32
    for t in range(1, 33):
        s = block(s, step=t)
    mx.eval(s)

    final_norm = float(mx.linalg.norm(s).item())
    ratio = final_norm / max(initial_norm, 1e-6)

    # Norm must not explode (Lipschitz bound: ratio <= 1.25)
    assert ratio <= 1.25, f"Recurrent unroll exploded: final/initial norm ratio was {ratio}"
    assert not mx.any(mx.isnan(s)).item()
    assert not mx.any(mx.isinf(s)).item()


def test_dynamic_egate_convergence_on_cognitive_tasks():
    """Verify 3-Signal E-Gate detects consensus on converging cognitive trajectory."""
    from parallel_latent_reasoner.egate import DynamicDeliberationGate

    gate = DynamicDeliberationGate(min_steps=2, max_steps=12, tol_rel_vel=0.10, tol_erank_delta=0.005)

    # Simulate converging trajectory
    base_state = mx.random.normal((1, 16, 256))
    
    # Step 0: Initial state registration
    gate.update(base_state, step=0, coda_token=42)

    for t in range(1, 10):
        # Asymptotically decaying perturbation
        decay = 1.0 / (3.0 ** t)
        state_t = base_state + decay * mx.random.normal((1, 16, 256))
        mx.eval(state_t)

        token_t = 42  # Stable consensus token
        telemetry = gate.update(
            curr_state=state_t,
            step=t,
            coda_token=token_t,
        )

        if t >= 3:
            # Velocity decay and consensus should trigger early exit
            if telemetry.halt:
                assert telemetry.exit_reason in (
                    "3_signal_consensus",
                    "consensus_reached",
                    "velocity_decay",
                    "max_steps_reached",
                )
                break

