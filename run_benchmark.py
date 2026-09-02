#!/usr/bin/env python3
"""Automated Multi-Scale Benchmark Runner for Parallel Latent Reasoner (PRLR).

Evaluates reasoning latency, effective throughput, and memory footprint across
Gemma resident scales (Compact Test, 2B, 9B, 12B) on Apple Silicon Metal GPU,
emitting JSON prlr.benchmark.v1, CSV, and ASCII summary tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path for standalone script execution
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from parallel_latent_reasoner.benchmark import MultiScaleBenchmarkSuite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multi-Scale PRLR vs Autoregressive CoT Comparative Benchmark"
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="compact_test,gemma_2b,gemma_9b,gemma_12b",
        help="Comma-separated scale presets to evaluate (e.g. compact_test,gemma_2b,gemma_9b,gemma_12b).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional single preset override (e.g. compact_test).",
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
        help="Number of evaluation repetitions per scale (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save JSON and CSV artifacts (default: results/).",
    )

    args = parser.parse_args()

    if args.config:
        presets = [args.config.strip()]
    else:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    print("=" * 80)
    print("  PARALLEL LATENT REASONER (PRLR) - MULTI-SCALE BENCHMARK HARNESS")
    print("  Platform: Apple Silicon Metal GPU | Framework: Pure MLX")
    print(f"  Evaluating Presets: {presets} (Slots M={args.slots}, Steps T={args.steps})")
    print("=" * 80)

    suite = MultiScaleBenchmarkSuite(
        presets=presets,
        num_slots=args.slots,
        num_steps=args.steps,
        enable_gate=not args.no_gate,
        repeats=args.repeats,
        output_dir=args.output_dir,
    )

    suite.run()

    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY TABLE (MATCHED COMPUTE K_cot = T * M)")
    print("=" * 80)
    ascii_table = suite.to_ascii_table()
    print(ascii_table)

    json_path, csv_path = suite.save_artifacts()
    print("\n" + "=" * 80)
    print(f"[*] Artifacts saved successfully:")
    print(f"    - JSON: {json_path}")
    print(f"    - CSV:  {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
