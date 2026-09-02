"""Multi-Scale Automated Benchmark Suite for Parallel Latent Reasoner (PRLR).

Compares parallel continuous latent deliberation against sequential autoregressive
chain-of-thought (CoT) generation at matched compute (K_cot = T * M linear evaluations)
across Gemma resident scales (Compact Test, 2B, 9B, 12B).

Outputs results in JSON (schema prlr.benchmark.v1), CSV, and ASCII tables.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.pipeline import GemmaDeliberationPipeline
from parallel_latent_reasoner.probes import (
    analyze_deliberation_trajectory,
    compute_effective_rank,
)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _get_peak_memory_bytes() -> int:
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory()
    return 0


@dataclass
class BenchmarkResult:
    """Benchmark evaluation result for a single configuration preset."""

    preset: str
    dim: int
    num_heads: int
    num_slots: int
    deliberation_steps: int
    steps_executed: int
    matched_cot_tokens: int
    delib_latency_ms: float
    cot_latency_ms: float
    speedup: float
    delib_throughput_eff: float
    cot_throughput: float
    delib_peak_vram_mb: float
    cot_peak_vram_mb: float
    compute_saved_pct: float
    exit_reason: str
    initial_erank: float
    final_erank: float
    final_velocity: float


def run_ar_cot_benchmark(
    model: MLXCompactGemmaModel,
    prompt_tokens: mx.array,
    k_cot_tokens: int,
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[float, float]:
    """Execute sequential Autoregressive CoT baseline and measure latency & peak memory."""
    # Warmup
    for _ in range(warmup):
        slots, prompt_hiddens = model.prelude(prompt_tokens)
        prompt_len = prompt_hiddens.shape[1]
        prompt_kv = model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)
        curr = slots[:, :1, :]
        for step in range(1, min(4, k_cot_tokens) + 1):
            curr = model.engine.step(curr, step_idx=step, prompt_kv=prompt_kv, prompt_len=prompt_len + step - 1)
        mx.eval(curr)

    latencies: list[float] = []
    peak_mems: list[int] = []

    for _ in range(repeats):
        _reset_peak_memory()
        t0 = time.perf_counter()

        slots, prompt_hiddens = model.prelude(prompt_tokens)
        prompt_len = prompt_hiddens.shape[1]
        prompt_kv = model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

        curr = slots[:, :1, :]
        for step in range(1, k_cot_tokens + 1):
            curr = model.engine.step(
                curr,
                step_idx=step,
                prompt_kv=prompt_kv,
                prompt_len=prompt_len + step - 1,
            )
            # Simulated token embedding readout and addition
            logits = model.coda.project_logits(model.coda.final_norm(curr[:, 0, :]))
            next_tok = mx.argmax(logits, axis=-1, keepdims=True)
            tok_embed = model.prelude.embed_prompt(next_tok)
            curr = curr + 0.1 * tok_embed

        mx.eval(curr)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000.0)
        peak_mems.append(_get_peak_memory_bytes())

    mean_latency_ms = float(sum(latencies) / len(latencies))
    mean_peak_mb = float(max(peak_mems) / (1024 * 1024)) if max(peak_mems) > 0 else (
        (model.config.dim * model.config.intermediate_dim * 4 * 2) / (1024 * 1024)
    )
    return mean_latency_ms, mean_peak_mb


def run_prlr_benchmark(
    pipeline: GemmaDeliberationPipeline,
    prompt_tokens: mx.array,
    steps_t: int,
    num_slots_m: int,
    enable_gate: bool = True,
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[float, float, int, str, float, float, float]:
    """Execute Parallel Latent Deliberation and measure latency, memory, and probes."""
    # Warmup
    for _ in range(warmup):
        delib_res, _ = pipeline.deliberate(
            prompt_tokens,
            steps=steps_t,
            enable_dynamic_gate=enable_gate,
        )
        mx.eval(delib_res.final_states)

    latencies: list[float] = []
    peak_mems: list[int] = []
    final_steps_executed = steps_t
    exit_reason = "fixed_unroll"
    init_erank = 1.0
    fin_erank = 1.0
    fin_vel = 0.0

    for _ in range(repeats):
        _reset_peak_memory()
        t0 = time.perf_counter()

        delib_res, gate_telemetry = pipeline.deliberate(
            prompt_tokens,
            steps=steps_t,
            enable_dynamic_gate=enable_gate,
            return_trajectory=True,
            compute_probes=True,
        )
        mx.eval(delib_res.final_states)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000.0)
        peak_mems.append(_get_peak_memory_bytes())

        final_steps_executed = delib_res.steps_executed
        if gate_telemetry:
            last_tel = gate_telemetry[-1]
            exit_reason = last_tel.exit_reason

        if delib_res.trajectory_states and len(delib_res.trajectory_states) >= 2:
            analysis = analyze_deliberation_trajectory(delib_res.trajectory_states, compute_erank=True)
            if analysis.effective_ranks:
                init_erank = analysis.effective_ranks[0]
                fin_erank = analysis.effective_ranks[-1]
            if analysis.step_velocities:
                fin_vel = analysis.step_velocities[-1]

    mean_latency_ms = float(sum(latencies) / len(latencies))
    mean_peak_mb = float(max(peak_mems) / (1024 * 1024)) if max(peak_mems) > 0 else (
        (pipeline.config.dim * pipeline.config.intermediate_dim * 4 * 2) / (1024 * 1024)
    )

    return (
        mean_latency_ms,
        mean_peak_mb,
        final_steps_executed,
        exit_reason,
        init_erank,
        fin_erank,
        fin_vel,
    )


def evaluate_preset(
    preset_name: str,
    prompt: str = "If a car travels 60 mph for 2.5 hours, how far does it go?",
    num_slots: int = 16,
    num_steps: int = 8,
    enable_gate: bool = True,
    repeats: int = 3,
) -> BenchmarkResult:
    """Run comparative benchmark for a single scale preset."""
    pipeline = GemmaDeliberationPipeline.from_preset(
        preset=preset_name,
        num_memory_slots=num_slots,
        deliberation_steps=num_steps,
    )
    config = pipeline.config

    prompt_tokens = mx.array([[ord(c) % config.vocab_size for c in prompt]], dtype=mx.int32)
    mx.eval(prompt_tokens)

    # 1. PRLR Deliberation
    delib_lat, delib_vram, steps_exec, exit_reason, init_er, fin_er, fin_v = run_prlr_benchmark(
        pipeline=pipeline,
        prompt_tokens=prompt_tokens,
        steps_t=num_steps,
        num_slots_m=num_slots,
        enable_gate=enable_gate,
        repeats=repeats,
    )

    # Matched compute budget: K_cot = T * M
    k_cot = num_steps * num_slots

    # 2. Autoregressive CoT
    cot_lat, cot_vram = run_ar_cot_benchmark(
        model=pipeline.model,
        prompt_tokens=prompt_tokens,
        k_cot_tokens=k_cot,
        repeats=repeats,
    )

    speedup = cot_lat / max(0.001, delib_lat)
    delib_eff_tps = (k_cot / (delib_lat / 1000.0)) if delib_lat > 0 else 0.0
    cot_tps = (k_cot / (cot_lat / 1000.0)) if cot_lat > 0 else 0.0
    compute_saved = max(0.0, (num_steps - steps_exec) / max(1, num_steps) * 100.0)

    return BenchmarkResult(
        preset=preset_name,
        dim=config.dim,
        num_heads=config.num_heads,
        num_slots=num_slots,
        deliberation_steps=num_steps,
        steps_executed=steps_exec,
        matched_cot_tokens=k_cot,
        delib_latency_ms=round(delib_lat, 2),
        cot_latency_ms=round(cot_lat, 2),
        speedup=round(speedup, 2),
        delib_throughput_eff=round(delib_eff_tps, 1),
        cot_throughput=round(cot_tps, 1),
        delib_peak_vram_mb=round(delib_vram, 2),
        cot_peak_vram_mb=round(cot_vram, 2),
        compute_saved_pct=round(compute_saved, 1),
        exit_reason=exit_reason,
        initial_erank=round(init_er, 2),
        final_erank=round(fin_er, 2),
        final_velocity=round(fin_v, 6),
    )


class MultiScaleBenchmarkSuite:
    """Harness managing multi-scale comparative evaluations and exporting artifacts."""

    def __init__(
        self,
        presets: Sequence[str] = ("compact_test", "gemma_2b", "gemma_9b", "gemma_12b"),
        num_slots: int = 16,
        num_steps: int = 8,
        enable_gate: bool = True,
        repeats: int = 3,
        output_dir: str | Path = "results",
    ):
        self.presets = list(presets)
        self.num_slots = num_slots
        self.num_steps = num_steps
        self.enable_gate = enable_gate
        self.repeats = repeats
        self.output_dir = Path(output_dir)
        self.results: list[BenchmarkResult] = []

    def run(self) -> list[BenchmarkResult]:
        """Execute benchmarks across all configured scale presets."""
        self.results.clear()
        for preset in self.presets:
            print(f"[*] Benchmarking resident scale preset: '{preset}' (M={self.num_slots}, T={self.num_steps})...")
            try:
                res = evaluate_preset(
                    preset_name=preset,
                    num_slots=self.num_slots,
                    num_steps=self.num_steps,
                    enable_gate=self.enable_gate,
                    repeats=self.repeats,
                )
                self.results.append(res)
                print(f"    -> Delib: {res.delib_latency_ms:.2f} ms | CoT: {res.cot_latency_ms:.2f} ms | Speedup: {res.speedup:.1f}x")
            except Exception as e:
                print(f"    [!] Error running preset '{preset}': {e}")
        return self.results

    def to_ascii_table(self) -> str:
        """Render results as an ASCII / Markdown table."""
        header = "| Preset | Dim | Delib Latency | CoT Latency | Speedup | Eff Throughput | Peak VRAM | Exit Step | Compute Saved |"
        sep = "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        rows = [header, sep]
        for r in self.results:
            rows.append(
                f"| **{r.preset}** | {r.dim} | **{r.delib_latency_ms:.2f} ms** | {r.cot_latency_ms:.2f} ms | **{r.speedup:.1f}x** | {r.delib_throughput_eff:,.1f} tok/s | {r.delib_peak_vram_mb:.2f} MB | t={r.steps_executed} | {r.compute_saved_pct:.1f}% |"
            )
        return "\n".join(rows)

    def save_artifacts(
        self,
        json_filename: str = "benchmark_summary.json",
        csv_filename: str = "benchmark_summary.csv",
    ) -> tuple[Path, Path]:
        """Save benchmark results in JSON (schema prlr.benchmark.v1) and CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / json_filename
        csv_path = self.output_dir / csv_filename

        # JSON artifact
        json_data = {
            "schema": "prlr.benchmark.v1",
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "device": "Apple Silicon Metal GPU (Unified Memory)",
                "mlx_version": getattr(mx, "__version__", "unknown"),
                "num_slots_m": self.num_slots,
                "deliberation_steps_t": self.num_steps,
                "enable_gate": self.enable_gate,
                "repeats": self.repeats,
            },
            "results": [asdict(r) for r in self.results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # CSV artifact
        if self.results:
            fieldnames = list(asdict(self.results[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))

        return json_path, csv_path


__all__ = [
    "BenchmarkResult",
    "MultiScaleBenchmarkSuite",
    "evaluate_preset",
    "run_ar_cot_benchmark",
    "run_prlr_benchmark",
]
