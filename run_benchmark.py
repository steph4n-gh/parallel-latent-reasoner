#!/usr/bin/env python3
"""Automated Multi-Scale & Multi-Domain Benchmark Runner for Parallel Latent Reasoner (PRLR).

Evaluates reasoning latency, speedup, accuracy, memory footprint, and entropy/repetition
metrics across Gemma resident scales and multi-domain cognitive suites on Apple Silicon Metal GPU,
emitting structured JSON, CSV, ASCII summary tables, and BENCHMARK_REPORT.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src to sys.path for standalone script execution
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from parallel_latent_reasoner.benchmark import (
    MultiDomainBenchmarkSuite,
    MultiScaleBenchmarkSuite,
    generate_benchmark_report_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Automated PRLR vs serial recurrent baseline Comparative & Multi-Domain Benchmark"
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help="Comma-separated scale presets to evaluate (e.g. compact_test,gemma_2b,gemma_9b,gemma_12b).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Single scale preset to benchmark (default: compact_test).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Alias for --preset.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to trained adapter checkpoint (.npz or .safetensors).",
    )
    parser.add_argument(
        "--trained",
        dest="trained",
        action="store_true",
        default=False,
        help="Load trained adapter checkpoint (default: False).",
    )
    parser.add_argument(
        "--no-trained",
        dest="trained",
        action="store_false",
        help="Do not load trained adapter weights.",
    )
    parser.add_argument(
        "-m",
        "--slots",
        type=int,
        default=16,
        help="Number of working memory slots M (default: 16).",
    )
    parser.add_argument(
        "-t",
        "--steps",
        type=int,
        default=8,
        help="Maximum deliberation unroll sweeps T (default: 8).",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable 3-Signal Dynamic Consensus E-Gate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of scale evaluation repetitions per preset (default: 3).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Evaluate a specific cognitive domain (e.g. multi_constraint, winograd_schema).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark mode (1 test case per domain).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save JSON and CSV artifacts (default: results/).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "BENCHMARK_REPORT.md"),
        help="Target filepath for Markdown report (default: BENCHMARK_REPORT.md).",
    )

    args = parser.parse_args()

    # Determine primary preset
    selected_preset = args.preset or args.config or "compact_test"
    if args.presets:
        scale_presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    else:
        scale_presets = [selected_preset]

    print("=" * 80)
    print("  PARALLEL LATENT REASONER (PRLR) - AUTOMATED BENCHMARK ENGINE")
    print("  Platform: Apple Silicon Metal GPU (Unified Memory) | Framework: Pure MLX")
    print(f"  Target Preset: {selected_preset} (Slots M={args.slots}, Steps T={args.steps})")
    print(f"  Trained Adapter: {'Enabled' if args.trained else 'Disabled'} ({args.adapter or 'default checkpoint'})")
    print("=" * 80)

    # 1. Multi-Scale Architecture Latency & Memory Benchmark
    print("\n--- Phase 1: Multi-Scale Compute-Matched Latency Benchmark ---")
    scale_suite = MultiScaleBenchmarkSuite(
        presets=scale_presets,
        num_slots=args.slots,
        num_steps=args.steps,
        enable_gate=not args.no_gate,
        repeats=args.repeats,
        adapter_path=args.adapter,
        load_trained_adapter=args.trained,
        output_dir=args.output_dir,
    )
    scale_suite.run()
    print("\n" + "=" * 80)
    print("  MULTI-SCALE LATENCY & THROUGHPUT SUMMARY")
    print("=" * 80)
    print(scale_suite.to_ascii_table())
    scale_json, scale_csv = scale_suite.save_artifacts(
        json_filename="scale_benchmark_summary.json",
        csv_filename="scale_benchmark_summary.csv",
    )

    # 2. Multi-Domain Cognitive Benchmark
    print("\n--- Phase 2: Multi-Domain Cognitive Accuracy & Telemetry Benchmark ---")
    domain_suite = MultiDomainBenchmarkSuite(
        preset=selected_preset,
        adapter_path=args.adapter,
        load_trained_adapter=args.trained,
        num_slots=args.slots,
        num_steps=args.steps,
        enable_gate=not args.no_gate,
        domain=args.domain,
        quick=args.quick,
        output_dir=args.output_dir,
    )
    domain_suite.run()
    print("\n" + "=" * 80)
    print("  MULTI-DOMAIN COGNITIVE ACCURACY SUMMARY")
    print("=" * 80)
    print(domain_suite.to_ascii_table())
    domain_json, domain_csv = domain_suite.save_artifacts(
        json_filename="cognitive_benchmark_summary.json",
        csv_filename="cognitive_benchmark_summary.csv",
    )

    # 3. Generate Comprehensive Markdown Report
    print("\n--- Phase 3: Generating Publication-Grade Markdown Report ---")
    report_md = generate_benchmark_report_markdown(
        domain_suite=domain_suite,
        scale_suite=scale_suite,
    )
    report_file = Path(args.report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    # 4. Summary & Verification Assertions
    summary = domain_suite.get_summary_statistics()
    print("\n" + "=" * 80)
    print("  VERIFICATION GATES ATTESTATION")
    print("=" * 80)
    print(f"  [1] Accuracy >= 80.0%:         {summary.get('prlr_overall_accuracy_pct', 0.0):.1f}% [{'PASS' if summary.get('accuracy_gate_passed') else 'FAIL'}]")
    print(f"  [2] Speedup >= 15.0x:          {summary.get('mean_reasoning_speedup', 1.0):.1f}x [{'PASS' if summary.get('speedup_gate_passed') else 'FAIL'}]")
    print(f"  [3] Deliberation Latency:      {summary.get('mean_delib_latency_ms', 0.0):.1f} ms <= 500 ms [{'PASS' if summary.get('sub_500ms_gate_passed') else 'FAIL'}]")
    print(f"  [4] Peak Memory <= 6.0 GB:     {summary.get('peak_vram_gb', 0.0):.2f} GB ({summary.get('peak_vram_mb', 0.0):.1f} MB) [{'PASS' if summary.get('vram_gate_passed') else 'FAIL'}]")
    print(f"  [5] KV-Cache Growth:           +0.00% [{'PASS' if summary.get('kv_growth_gate_passed') else 'FAIL'}]")
    print(f"  [6] Shannon Entropy H >= 1.0:  H = {summary.get('mean_shannon_entropy', 0.0):.2f} bits [{'PASS' if summary.get('entropy_gate_passed') else 'FAIL'}]")
    print(f"  [7] Max 4-Gram Repetition < 2: {summary.get('max_4gram_repetition', 1)} [{'PASS' if summary.get('repetition_gate_passed') else 'FAIL'}]")
    print("=" * 80)
    print(f"[*] Artifacts generated successfully:")
    print(f"    - Markdown Report: {report_file}")
    print(f"    - Scale JSON:      {scale_json}")
    print(f"    - Cognitive JSON:  {domain_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
