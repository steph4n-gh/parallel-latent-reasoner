#!/usr/bin/env python3
"""CLI for Recurrent Latent Memory Kernel Microbenchmark.

Milestone 6 Requirement R9 / Feature 26:
Executes the separated kernel microbenchmark and emits:
- results/kernel_microbenchmark.json
- results/KERNEL_MICROBENCHMARK_REPORT.md
Strictly enforces Non-Negotiable Evidence Rules 4, 6, 7, 10.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prlr.eval.microbench import (
    RULE_4_DISCLAIMER,
    KernelMicrobenchConfig,
    KernelMicrobenchmarkRunner,
    get_git_metadata,
    get_hardware_metadata,
    render_markdown_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PRLR Recurrent Latent Memory Kernel Microbenchmark (Feature 26)"
    )
    parser.add_argument(
        "--preset",
        choices=["gemma_2b", "compact_test"],
        default="gemma_2b",
        help="Kernel dimension preset (default: gemma_2b).",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=16,
        help="Number of memory slots M (default: 16).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Deliberation recurrence depth T (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size B (default: 1).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark repetitions (default: 50).",
    )
    parser.add_argument(
        "--soak-runs",
        type=int,
        default=200,
        help="Number of consecutive unrolls for memory leak soak (default: 200).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run multi-dimensional sweep over M in {1, 4, 8, 16, 32} and T in {1, 2, 4, 8, 12, 16}.",
    )
    parser.add_argument(
        "--quick",
        "--smoke",
        action="store_true",
        help="Run fast smoke test with reduced iterations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
        help="Directory to store outputs (default: results/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  PARALLEL LATENT REASONER — RECURRENT KERNEL MICROBENCHMARK (FEATURE 26)")
    print("  Platform: Apple Silicon Metal GPU | Rules Enforced: 4, 6, 7, 10")
    print("=" * 80)
    print(f"\n[!] DISCLAIMER: {RULE_4_DISCLAIMER}\n")

    iterations = 5 if args.quick else args.iterations
    soak_runs = 20 if args.quick else args.soak_runs
    warmup_runs = 3 if args.quick else 10

    runner = KernelMicrobenchmarkRunner(seed=args.seed)

    if args.sweep:
        print(f"[*] Executing full parameter sweep for preset '{args.preset}'...")
        slots_to_sweep = (1, 4, 8, 16) if args.quick else (1, 4, 8, 16, 32)
        steps_to_sweep = (1, 4, 8) if args.quick else (1, 2, 4, 8, 12, 16)
        results = runner.run_sweep(
            tier=args.preset,
            slots_list=slots_to_sweep,
            steps_list=steps_to_sweep,
            runs=iterations,
            include_eager=True,
        )
    else:
        print(f"[*] Profiling condition: preset={args.preset}, M={args.slots}, T={args.steps}, B={args.batch_size}...")
        builder = (
            KernelMicrobenchConfig.gemma_2b
            if args.preset == "gemma_2b"
            else KernelMicrobenchConfig.compact_test
        )
        cfg = builder(
            slots=args.slots,
            steps=args.steps,
            compiled=True,
            batch_size=args.batch_size,
            runs=iterations,
        )
        cfg.warmup_runs = warmup_runs
        cfg.soak_runs = soak_runs
        res = runner.run_single(cfg)
        results = [res]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build output payload
    meta = {
        "benchmark_type": "recurrent_latent_memory_kernel_microbenchmark",
        "disclaimer": RULE_4_DISCLAIMER,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "git": get_git_metadata(PROJECT_DIR.parents[0]),
        "hardware": get_hardware_metadata(),
        "runtime": {
            "python_version": sys.version.split()[0],
            "mlx_version": getattr(sys.modules.get("mlx.core"), "__version__", "unknown"),
            "numpy_version": getattr(sys.modules.get("numpy"), "__version__", "unknown"),
        },
        "rules_enforced": [4, 6, 7, 10],
    }

    payload = {
        "schema_version": "1.0.0",
        "metadata": meta,
        "benchmarks": [
            {
                "condition_id": r.condition_id,
                "parameters": r.parameters,
                "timing_ms": r.timing_ms,
                "flops": r.flops,
                "memory_bandwidth": r.memory_bandwidth,
                "vram": r.vram,
                "throughput": r.throughput,
            }
            for r in results
        ],
    }

    # Write JSON artifact
    json_path = args.output_dir / "kernel_microbenchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[✓] Microbenchmark JSON saved to: {json_path}")

    # Write Markdown report
    md_content = render_markdown_report(payload)
    md_path = args.output_dir / "KERNEL_MICROBENCHMARK_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Microbenchmark Markdown report saved to: {md_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("  MICROBENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    for r in results:
        print(
            f"  {r.condition_id:<38} | Lat: {r.timing_ms['median_p50']:>6.2f} ms | "
            f"GFLOP/s: {r.flops['achieved_gflops']:>8.1f} | "
            f"BW: {r.memory_bandwidth['achieved_bandwidth_gb_s']:>6.1f} GB/s | "
            f"Peak VRAM: {str(r.vram['peak_vram_mb']):>7} MB"
        )
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
