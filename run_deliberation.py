#!/usr/bin/env python3
"""Standalone demonstration runner for Parallel Latent Reasoner (PRLR).

Executes parallel latent deliberation over working memory slots with pure MLX
and the 3-Signal Dynamic Consensus E-Gate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path for standalone script execution
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import mlx.core as mx

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.pipeline import GemmaDeliberationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRLR deliberation experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to configuration JSON.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="compact_test",
        choices=["compact_test", "gemma_2b", "gemma_9b", "gemma_12b", "gemma_e4b"],
        help="Model resident scale preset.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="If a car travels 60 mph for 2.5 hours, how far does it go?",
        help="Prompt text for deliberation.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Max deliberation unroll steps T.")
    parser.add_argument("--slots", type=int, default=16, help="Number of memory slots M.")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable 3-Signal Dynamic Consensus E-Gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = GemmaLatentConfig.from_json(args.config)
        pipeline = GemmaDeliberationPipeline(config=cfg)
    else:
        pipeline = GemmaDeliberationPipeline.from_preset(
            preset=args.preset,
            num_memory_slots=args.slots,
            deliberation_steps=args.steps,
        )

    print("=" * 65)
    print("  PARALLEL LATENT REASONER (PRLR) - DELIBERATION RUNNER")
    print(f"  Preset: {args.preset} (D={pipeline.config.dim}, M={pipeline.config.num_memory_slots} Slots)")
    print(f"  Prompt: \"{args.prompt}\"")
    print("=" * 65)

    t0 = time.perf_counter()
    out = pipeline.generate(
        prompt=args.prompt,
        max_new_tokens=16,
        deliberation_steps=args.steps,
        enable_dynamic_gate=not args.no_gate,
        return_diagnostics=True,
    )
    elapsed = time.perf_counter() - t0

    decoded_text = pipeline.decode_solution(out.token_ids)

    print(f"\nExecution Summary:")
    print(f"  • Deliberation Steps Executed: {out.deliberation_steps} / {args.steps}")
    print(f"  • Deliberation Latency: {out.metrics['deliberation_latency_ms']:.2f} ms" if out.metrics else "")
    print(f"  • Total Wall-Clock Latency: {elapsed * 1000.0:.2f} ms")
    print(f"  • Decoded Solution: \"{decoded_text.strip()}\"")
    print("-" * 65)

    if out.gate_telemetry:
        print(f"{'Step':<6} | {'Velocity':<12} | {'Rel Decay':<10} | {'erank':<8} | {'Consensus':<10} | {'Status':<8}")
        print("-" * 65)
        for tel in out.gate_telemetry:
            if tel.step == 0:
                continue
            status = "HALTED" if tel.halt else "Active"
            coda_str = f'"{tel.coda_token_str or tel.coda_token}"'
            print(f"t={tel.step:<4} | {tel.velocity:<12.6f} | {tel.rel_velocity:<10.4f} | {tel.erank:<8.2f} | {coda_str:<10} | {status:<8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
