"""Unit tests for PRLR Benchmark Suite and Terminal Visualizer."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from parallel_latent_reasoner.benchmark import (
    BenchmarkResult,
    MultiScaleBenchmarkSuite,
    evaluate_preset,
)
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import GateTelemetry
from parallel_latent_reasoner.visualizer import render_comparison_view


def test_benchmark_evaluate_preset():
    """Verify evaluate_preset runs on compact_test and populates all metrics."""
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
    assert res.delib_latency_ms > 0
    assert res.cot_latency_ms > 0
    assert res.speedup > 0
    assert res.matched_cot_tokens == 64  # 4 * 16


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
