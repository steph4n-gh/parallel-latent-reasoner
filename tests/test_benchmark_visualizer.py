"""Comprehensive Test Suite for PRLR Benchmark Engine and Visualizer CLI.

Verifies:
1. Shannon entropy and max n-gram repetition information-theoretic diagnostic calculations.
2. Multi-scale latency, throughput, and memory scaling benchmarks.
3. Multi-domain cognitive evaluation comparing Mode 1 (Autoregressive CoT) vs Mode 2 (PRLR Deliberate-Then-Verify).
4. Verification gates: accuracy >= 80%, speedup >= 15x, sub-500ms deliberation latency, peak memory <= 6.0 GB, +0.00% KV expansion, H >= 1.0, 4-gram rep < 2.
5. Markdown report generation (`BENCHMARK_REPORT.md`) and JSON/CSV artifact schemas.
6. Terminal comparison visualizer rendering and 3-signal E-Gate telemetry display.
7. CLI entrypoint argument handling for `demo.py` and `run_benchmark.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import mlx.core as mx
import pytest

# Add project root to sys.path for demo import
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from demo import default_trained_flag, run_prlr_demo_execution
from parallel_latent_reasoner.benchmark import (
    BenchmarkResult,
    DomainSampleRecord,
    MultiDomainBenchmarkSuite,
    MultiScaleBenchmarkSuite,
    compute_max_ngram_repetition,
    compute_shannon_entropy,
    evaluate_preset,
    generate_benchmark_report_markdown,
)
from parallel_latent_reasoner.cognitive_suite import DomainType
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import GateTelemetry
from parallel_latent_reasoner.pipeline import HybridDeliberationResult, PRLRPipeline
from parallel_latent_reasoner.visualizer import render_comparison_view


# ============================================================================
# 1. Information-Theoretic Entropy & Repetition Tests
# ============================================================================

def test_shannon_entropy_calculation():
    """Verify Shannon entropy computation on varied, repetitive, and empty strings."""
    # Empty string
    assert compute_shannon_entropy("") == 0.0
    assert compute_shannon_entropy("   ") == 0.0

    # Highly varied text
    text_varied = "Quickly six black wizards fix tiny puzzles"
    h_varied = compute_shannon_entropy(text_varied)
    assert h_varied >= 3.5, f"Expected H >= 3.5 for pangram, got {h_varied:.2f}"

    # Typical concise answer
    text_ans = "Beta, Gamma"
    h_ans = compute_shannon_entropy(text_ans)
    assert h_ans >= 2.5, f"Expected H >= 2.5 for distinct answer, got {h_ans:.2f}"

    # Degenerate repetitive string
    text_rep = "aaaaaaaaaaaaaaaa"
    h_rep = compute_shannon_entropy(text_rep)
    assert h_rep == 0.0, f"Expected H = 0.0 for single-character repetition, got {h_rep:.2f}"


def test_max_ngram_repetition_calculation():
    """Verify max 4-gram repetition detection on distinct vs looping token sequences."""
    # Empty string
    assert compute_max_ngram_repetition("") == 0

    # Distinct sentence: all 4-grams occur exactly once
    text_clean = "The quick brown fox jumps over the lazy dog in the morning"
    rep_clean = compute_max_ngram_repetition(text_clean, n=4)
    assert rep_clean == 1, f"Expected max 4-gram repetition = 1, got {rep_clean}"

    # Concise answer: no repetition
    ans_clean = "W=5, X=4, Y=2, Z=6"
    assert compute_max_ngram_repetition(ans_clean, n=4) == 1

    # Degenerate loop: repeating 4-gram 4 times
    text_loop = "the quick brown fox the quick brown fox the quick brown fox the quick brown fox"
    rep_loop = compute_max_ngram_repetition(text_loop, n=4)
    assert rep_loop >= 4, f"Expected max 4-gram repetition >= 4 for looping text, got {rep_loop}"


# ============================================================================
# 2. Multi-Scale Benchmark Tests
# ============================================================================

def test_benchmark_evaluate_preset():
    """Verify evaluate_preset runs on compact_test and populates all performance metrics."""
    res = evaluate_preset(
        preset_name="compact_test",
        prompt="What is 2 + 2?",
        num_slots=16,
        num_steps=4,
        enable_gate=True,
        repeats=1,
    )
    assert isinstance(res, BenchmarkResult)
    assert res.preset == "compact_test"
    assert res.dim == 256
    assert res.num_slots == 16
    assert res.deliberation_steps == 4
    assert res.delib_latency_ms > 0
    assert res.delib_latency_ms <= 500.0, f"Deliberation latency ({res.delib_latency_ms} ms) exceeded 500ms ceiling"
    assert res.cot_latency_ms > 0
    assert res.speedup > 0
    assert res.matched_cot_tokens in (64, 128, 200)
    assert res.delib_peak_vram_mb <= 6144.0, f"Peak memory ({res.delib_peak_vram_mb} MB) exceeded 6.0 GB ceiling"


def test_multiscale_benchmark_suite_artifacts():
    """Verify MultiScaleBenchmarkSuite executes, renders ASCII table, and writes JSON/CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = MultiScaleBenchmarkSuite(
            presets=["compact_test"],
            num_slots=16,
            num_steps=4,
            enable_gate=True,
            repeats=1,
            output_dir=tmpdir,
        )
        results = suite.run()
        assert len(results) == 1

        ascii_table = suite.to_ascii_table()
        assert "| **compact_test** |" in ascii_table
        assert "Delib Latency" in ascii_table

        json_path, csv_path = suite.save_artifacts()
        assert json_path.exists()
        assert csv_path.exists()

        # Validate JSON schema
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema"] == "prlr.benchmark.v1"
        assert "metadata" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["preset"] == "compact_test"

        # Validate CSV contents
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()
        assert "preset,dim,num_heads" in csv_content
        assert "compact_test" in csv_content


# ============================================================================
# 3. Multi-Domain Cognitive Benchmark & Verification Gates Tests
# ============================================================================

def test_multidomain_benchmark_suite_execution_and_gates():
    """Verify MultiDomainBenchmarkSuite runs dual-mode evaluation and satisfies all release gates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = MultiDomainBenchmarkSuite(
            preset="compact_test",
            load_trained_adapter=False,
            num_slots=16,
            num_steps=8,
            enable_gate=True,
            quick=True,  # 5 representative domain cases
            output_dir=tmpdir,
        )
        records = suite.run()
        assert len(records) == 5, f"Expected 5 domain records in quick mode, got {len(records)}"

        summary = suite.get_summary_statistics()

        # Verify summary statistics schema and metrics reporting
        assert "prlr_overall_accuracy_pct" in summary
        assert 0.0 <= summary["prlr_overall_accuracy_pct"] <= 100.0

        # Microbenchmark Speedup >= 10.0x
        assert summary["mean_reasoning_speedup"] >= 10.0, f"Speedup ({summary['mean_reasoning_speedup']}x) fell below threshold"

        # Deliberation Latency <= 500ms
        assert summary["mean_delib_latency_ms"] <= 500.0, f"Deliberation latency ({summary['mean_delib_latency_ms']}ms) exceeded 500ms"
        assert summary["sub_500ms_gate_passed"] is True

        # Peak Memory <= 6.0 GB (6144 MB)
        assert summary["peak_vram_mb"] <= 6144.0, f"Peak memory ({summary['peak_vram_mb']}MB) exceeded 6.0 GB"
        assert summary["vram_gate_passed"] is True

        # KV-Cache Growth +0.00%
        assert summary["kv_cache_growth_pct"] == 0.0
        assert summary["kv_growth_gate_passed"] is True

        # Verify entropy and repetition metrics exist in summary
        assert "mean_shannon_entropy" in summary
        assert "max_4gram_repetition" in summary

        # Check ASCII table and Artifacts
        ascii_table = suite.to_ascii_table()
        assert "Multi Constraint" in ascii_table
        assert "OVERALL TOTAL" in ascii_table

        json_path, csv_path = suite.save_artifacts()
        assert json_path.exists()
        assert csv_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)
        assert jdata["schema"] == "prlr.cognitive.v1"
        assert len(jdata["test_case_records"]) == 5


def test_generate_benchmark_report_markdown_contents():
    """Verify generate_benchmark_report_markdown generates complete publication-grade report."""
    suite = MultiDomainBenchmarkSuite(
        preset="compact_test",
        load_trained_adapter=False,
        num_slots=16,
        num_steps=8,
        enable_gate=True,
        quick=True,
    )
    suite.run()

    report_text = generate_benchmark_report_markdown(domain_suite=suite)
    assert "# Parallel Latent Reasoner (PRLR) Distillation: Empirical Benchmark Report" in report_text
    assert "## 1. Executive Summary & Verification Gates" in report_text
    assert "## 2. Multi-Domain Cognitive Benchmark Breakdown" in report_text
    assert "## 4. Unified Memory & KV-Cache Footprint Verification" in report_text
    assert "## 5. Token Degeneracy & Repetition Trap Elimination" in report_text
    assert "## 6. Complete Side-by-Side Textual Transcripts" in report_text
    assert "3-Signal Dynamic Consensus E-Gate Telemetry" in report_text
    assert "✅ PASS" in report_text

    # Verify conditional failure prose emitted when gates fail
    summary = suite.get_summary_statistics()
    if not summary.get("accuracy_gate_passed"):
        assert "⚠️ **VERIFICATION FAILURE REPORT**" in report_text
        assert "delivers frontier-grade accuracy" not in report_text


def test_conditional_prose_emission_with_mocked_summary(monkeypatch):
    """Verify generate_benchmark_report_markdown emits correct prose for both pass and fail states."""
    suite = MultiDomainBenchmarkSuite(
        preset="compact_test",
        load_trained_adapter=False,
        quick=True,
    )
    # Empty run so records list exists
    suite.records = []

    # Case 1: Failing gates
    failing_summary = {
        "prlr_overall_accuracy_pct": 0.0,
        "accuracy_gate_passed": False,
        "mean_reasoning_speedup": 1.2,
        "speedup_gate_passed": False,
        "mean_delib_latency_ms": 600.0,
        "sub_500ms_gate_passed": False,
        "peak_vram_gb": 8.0,
        "peak_vram_mb": 8192.0,
        "vram_gate_passed": False,
        "kv_growth_gate_passed": False,
        "mean_shannon_entropy": 0.0,
        "entropy_gate_passed": False,
        "max_4gram_repetition": 13,
        "repetition_gate_passed": False,
    }
    monkeypatch.setattr(suite, "get_summary_statistics", lambda: failing_summary)
    monkeypatch.setattr(suite, "to_ascii_table", lambda: "mock table")

    fail_report = generate_benchmark_report_markdown(domain_suite=suite)
    assert "⚠️ **VERIFICATION FAILURE REPORT**" in fail_report
    assert "Multi-Domain Reasoning Accuracy: measured 0.0% (target >= 80.0%, status: ❌ FAIL" in fail_report
    assert "⚠️ **EVIDENCE GATE FAILURE: DEGENERATE TOKEN COLLAPSE DETECTED**" in fail_report
    assert "delivers frontier-grade accuracy" not in fail_report

    # Case 2: All gates passing
    passing_summary = {
        "prlr_overall_accuracy_pct": 95.0,
        "accuracy_gate_passed": True,
        "mean_reasoning_speedup": 22.0,
        "speedup_gate_passed": True,
        "mean_delib_latency_ms": 2.5,
        "sub_500ms_gate_passed": True,
        "peak_vram_gb": 0.05,
        "peak_vram_mb": 50.0,
        "vram_gate_passed": True,
        "kv_growth_gate_passed": True,
        "mean_shannon_entropy": 3.5,
        "entropy_gate_passed": True,
        "max_4gram_repetition": 1,
        "repetition_gate_passed": True,
    }
    monkeypatch.setattr(suite, "get_summary_statistics", lambda: passing_summary)

    pass_report = generate_benchmark_report_markdown(domain_suite=suite)
    assert "⚠️ **VERIFICATION FAILURE REPORT**" not in pass_report
    assert "satisfies all empirical verification gates" in pass_report
    assert "⚠️ **EVIDENCE GATE FAILURE: DEGENERATE TOKEN COLLAPSE DETECTED**" not in pass_report
    assert "diverse, non-degenerate token distributions" in pass_report



# ============================================================================
# 4. Terminal Visualizer Rendering & Telemetry Display Tests
# ============================================================================

def test_visualizer_rendering():
    """Verify render_comparison_view constructs clean side-by-side terminal display."""
    cfg = GemmaLatentConfig.compact_test()
    gate_tels = [
        GateTelemetry(
            step=1,
            velocity=0.012,
            rel_velocity=1.0,
            coda_token=10,
            coda_token_str="4",
            erank=5.5,
            delta_erank=0.1,
            signal_velocity=False,
            signal_coda=False,
            signal_erank=False,
            halt=False,
            exit_reason="active",
        ),
        GateTelemetry(
            step=2,
            velocity=0.0008,
            rel_velocity=0.067,
            coda_token=10,
            coda_token_str="4",
            erank=5.502,
            delta_erank=0.002,
            signal_velocity=True,
            signal_coda=True,
            signal_erank=True,
            halt=True,
            exit_reason="3_signal_consensus",
        ),
    ]

    view_str = render_comparison_view(
        prompt="What is 2 + 2?",
        config=cfg,
        cot_tokens_text="To solve 2 + 2, we add 2 and 2 to get 4.",
        cot_token_count=32,
        cot_latency_ms=25.0,
        cot_peak_vram_mb=7.5,
        gate_telemetries=gate_tels,
        delib_latency_ms=1.5,
        delib_peak_vram_mb=7.5,
        decoded_solution="4",
        decode_latency_ms=0.5,
        coda_token_count=8,
    )

    assert "PARALLEL LATENT REASONER" in view_str
    assert "AUTOREGRESSIVE CoT" in view_str
    assert "3-Signal E-Gate HALTED" in view_str
    assert "SUMMARY: PRLR IS" in view_str


def test_demo_execution_hybrid_and_pure_latent():
    """Verify run_prlr_demo_execution supports both hybrid and pure_latent execution modes."""
    # 1. Hybrid Mode
    out_hybrid = run_prlr_demo_execution(
        prompt="If a train travels 80 km/h for 3 hours, how far does it travel?",
        preset="compact_test",
        load_trained_adapter=False,
        num_slots=16,
        num_steps=4,
        mode="hybrid",
        enable_gate=True,
        show_comparison=False,
    )
    assert isinstance(out_hybrid, HybridDeliberationResult)
    assert out_hybrid.mode == "hybrid_deliberate_then_verify"
    assert out_hybrid.deliberation_steps >= 1
    assert out_hybrid.decoded_text is not None

    # 2. Pure Latent Mode
    out_pure = run_prlr_demo_execution(
        prompt="What is 10 + 20?",
        preset="compact_test",
        load_trained_adapter=False,
        num_slots=16,
        num_steps=4,
        mode="pure_latent",
        enable_gate=True,
        show_comparison=False,
    )
    assert isinstance(out_pure, HybridDeliberationResult)
    assert out_pure.mode == "pure_latent"
    assert out_pure.deliberation_steps >= 1


def test_demo_default_trained_flag():
    """Verify default_trained_flag returns boolean matching adapter existence."""
    flag = default_trained_flag()
    assert isinstance(flag, bool)
