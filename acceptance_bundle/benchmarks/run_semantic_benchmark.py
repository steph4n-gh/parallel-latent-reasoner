#!/usr/bin/env python3
"""CLI for Pretrained Gemma 2B Separated Semantic Benchmark.

Milestone 6 Requirement R9 / Feature 27:
Executes the separated semantic benchmark and emits:
- results/semantic_benchmark.json
- results/SEMANTIC_BENCHMARK_REPORT.md
Strictly enforces Non-Negotiable Evidence Rules 1, 2, 5, 8, 9, 10.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prlr.eval.semantic_bench import (
    DISCLAIMER_SEMANTIC,
    SemanticBenchmarkRunner,
    render_semantic_markdown_report,
)
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PRLR Pretrained Gemma 2B Semantic Benchmark (Feature 27)"
    )
    parser.add_argument(
        "--split",
        choices=["sealed_test", "dev", "sealed_gate", "extrapolation"],
        default="sealed_test",
        help="Dataset split to evaluate (default: sealed_test).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to trained adapter weights (.safetensors / .npz).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples.",
    )
    parser.add_argument(
        "--quick",
        "--smoke",
        action="store_true",
        help="Run fast smoke test with small sample limit.",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        default=True,
        help="Compute empirical Pareto curves (default: True).",
    )
    parser.add_argument(
        "--no-pareto",
        action="store_false",
        dest="pareto",
        help="Disable Pareto curve generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
        help="Directory to save output reports (default: results/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  PARALLEL LATENT REASONER — PRETRAINED SEMANTIC BENCHMARK (FEATURE 27)")
    print("  Backbone: google/gemma-2b-it | Rules Enforced: 1, 2, 5, 8, 9, 10")
    print("=" * 80)
    print(f"\n[!] DISCLAIMER: {DISCLAIMER_SEMANTIC}\n")

    manifest = ModelManifest.gemma_2b_it()
    print("[*] Loading official google/gemma-2b-it backbone and tokenizer...")
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)

    print("[*] Initializing recurrent adapter and causal prefix decoder...")
    adapter = GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1, deliberation_steps=4)
    if args.checkpoint and args.checkpoint.exists():
        print(f"[*] Loading adapter checkpoint from {args.checkpoint}...")
        adapter.load_weights(str(args.checkpoint))

    decoder = GemmaCausalPrefixDecoder(backbone=backbone)
    data_dir = PROJECT_DIR / "data" / "prlr_domain_v1"

    runner = SemanticBenchmarkRunner(
        backbone=backbone,
        adapter=adapter,
        decoder=decoder,
        data_dir=data_dir,
        seed=args.seed,
    )

    sample_limit = 4 if args.quick else args.limit
    run_pareto = False if args.quick else args.pareto

    results = runner.run_benchmark(
        split=args.split,
        limit=sample_limit,
        run_pareto=run_pareto,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON artifact
    json_path = args.output_dir / "semantic_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Semantic benchmark JSON saved to: {json_path}")

    # Save Markdown report
    md_content = render_semantic_markdown_report(results)
    md_path = args.output_dir / "SEMANTIC_BENCHMARK_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Semantic benchmark Markdown report saved to: {md_path}")

    # Summary table
    sm = results["summary"]
    lat = sm["stage_latencies_ms"]
    print("\n" + "=" * 80)
    print("  SEMANTIC BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Exact Match Accuracy : {sm['exact_match_accuracy'] * 100:.2f}% (95% BCa CI: [{sm['exact_match_ci_95_bca'][0] * 100:.1f}%, {sm['exact_match_ci_95_bca'][1] * 100:.1f}%])")
    print(f"  Terminal Accuracy    : {sm['terminal_tool_accuracy'] * 100:.2f}%")
    print(f"  Operational Validity : {sm['operational_validity'] * 100:.2f}%")
    print(f"  Mean Executed Depth  : {sm['mean_executed_depth']:.2f} / 12 (Depth Reduction: {sm['depth_reduction_pct']:.1f}%)")
    print(f"  Prefill Latency (p50): {lat['prefill']['median_p50']:.2f} ms")
    print(f"  Deliberation (p50)   : {lat['deliberation']['median_p50']:.2f} ms")
    print(f"  Decode Latency (p50) : {lat['decode']['median_p50']:.2f} ms")
    print(f"  Total Latency (p50)  : {lat['total']['median_p50']:.2f} ms")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
