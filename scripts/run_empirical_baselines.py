#!/usr/bin/env python3
"""Run Empirical Baselines on Full Sealed Test Split (256 samples).

Conforms strictly to:
- Evidence Rules 1-10 (blind evaluation, post-hoc scoring, zero synthetic traces).
- Urgent Research Guidance & Milestone 1 R1.

Features:
1. Two-Stage Decoupled Execution:
   - Stage 1 (generate): Reads target-free evaluation inputs only. Zero access to answer keys.
     Immediately aborts if requested checkpoint is missing (Rule 5).
     Emits atomic prediction JSON with SHA-256 sidecars.
     Halts on token 106 (<turn|>) for direct_frozen and repo_decoder.
   - Stage 2 (score): Cryptographically verifies prediction file SHA-256 before accessing
     quarantined answer keys. Computes post-hoc verification metrics and updates summary.
     Refuses to merge cross-run summaries if commits, datasets, or runtimes diverge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prlr.domain.solver_lane import ProceduralVerifier
from prlr.eval.bootstrap import run_bootstrap_analysis
from prlr.eval.harness import (
    ConditionScoredSummary,
    EvaluationInput,
    MissingCheckpointError,
    OracleLeakageError,
    PredictionIntegrityError,
    PredictionRecord,
    ScoredSummaryArtifact,
    generate_direct_frozen,
    generate_predictions,
    load_adapter_and_injection_weights,
    score_predictions,
    verify_adapter_checkpoint,
)
from prlr.gemma.adapter import GemmaNonRecurrentAdapter, GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


ADAPTER_CONDITIONS: Set[str] = {
    "adapter_t0",
    "adapter_t1",
    "adapter_t2",
    "adapter_t4",
    "adapter_recurrent",
    "control_zeroed",
    "control_random",
    "control_shuffled",
    "non_recurrent",
}

ALL_CONDITIONS: List[str] = [
    "direct_frozen",
    "repo_decoder",
    "control_zeroed",
    "control_random",
    "adapter_recurrent",
    "control_shuffled",
    "non_recurrent",
]

SUPPORTED_CONDITIONS: List[str] = [
    "direct_frozen",
    "repo_decoder",
    "adapter_t0",
    "adapter_t1",
    "adapter_t2",
    "adapter_t4",
    "adapter_recurrent",
    "control_zeroed",
    "control_random",
    "control_shuffled",
    "non_recurrent",
]


def print_decision_table(summary_path: Path) -> None:
    """Print clean comparison decision table from scored summary artifact."""
    if not summary_path.exists():
        return

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        conditions_data = data.get("conditions", {})
    except Exception:
        return

    print("\n" + "=" * 100)
    print("  DECISION TABLE: EMPIRICAL BASELINES ON SEALED TEST")
    print("=" * 100)
    print(
        f"| {'Experiment':<30} | {'Prompt contract':<15} | {'Adapter':<12} | "
        f"{'T':<5} | {'Exact':<8} | {'Terminal':<10} | {'Valid JSON':<10} | {'Max Rep':<8} |"
    )
    print(f"|{'-'*32}|{'-'*17}|{'-'*14}|{'-'*7}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*10}|")

    for cond, sm in sorted(conditions_data.items()):
        t_val = "0"
        if "_t" in cond:
            t_val = cond.split("_t")[-1]
        elif cond in ("control_zeroed", "control_shuffled", "control_random", "adapter_recurrent"):
            t_val = "4"
        elif cond == "non_recurrent":
            t_val = "1 (single-pass)"

        adapter_name = "None"
        if "adapter" in cond or "control" in cond:
            adapter_name = "Full rank"
        elif cond == "non_recurrent":
            adapter_name = "Non-rec"

        em_pct = f"{sm.get('exact_match_pct', 0.0)}%"
        term_pct = f"{sm.get('terminal_match_pct', 0.0)}%"
        json_pct = f"{sm.get('valid_json_pct', 0.0)}%"
        max_rep = str(sm.get("max_4gram_repetition", sm.get("max_repetition", 0)))

        print(
            f"| {cond:<30} | {'Official':<15} | {adapter_name:<12} | "
            f"{t_val:<5} | {em_pct:<8} | {term_pct:<10} | {json_pct:<10} | {max_rep:<8} |"
        )
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PRLR Empirical Baseline Runner (Two-Stage Target-Free Architecture)"
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "score", "all"],
        default="all",
        help="Pipeline stage: 'generate' (target-free inference), 'score' (post-hoc verified scoring), or 'all'",
    )
    parser.add_argument(
        "--action",
        choices=["generate", "score", "all"],
        default=None,
        help="Alias for --stage",
    )
    parser.add_argument(
        "--condition",
        choices=["all"] + SUPPORTED_CONDITIONS,
        default="all",
        help="Experimental condition(s) to evaluate",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROJECT_DIR / "data" / "prlr_domain_v1" / "evaluation_inputs" / "sealed_test_inputs.jsonl",
        help="Path to strictly target-free input JSONL (Rule 1)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Backward compatibility alias for --input-path",
    )
    parser.add_argument(
        "--keys-path",
        type=Path,
        default=PROJECT_DIR / "data" / "prlr_domain_v1" / "answer_keys" / "sealed_test_keys.jsonl",
        help="Path to quarantined answer keys JSONL (Rule 2)",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=None,
        help="Path to specific prediction file to score (for --stage score)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "gemma_4_12b_prlr_adapter.safetensors",
        help="Path to trained adapter checkpoint (required for adapter conditions; fails fast if missing)",
    )
    parser.add_argument(
        "--non-recurrent-checkpoint",
        type=Path,
        default=None,
        help="Optional path to separate GemmaNonRecurrentAdapter checkpoint",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Run paired bootstrap analysis (1,000 resamples) and exact permutation tests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "empirical_baselines",
        help="Output directory for predictions, sidecars, and summary artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples for rapid debugging",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Maximum generation tokens per sample",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=["cross_attention", "prefix"],
        default="cross_attention",
        help="Conditioning mode for GemmaCausalPrefixDecoder (default: 'cross_attention')",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")

    args = parser.parse_args()

    stage = args.action if args.action is not None else args.stage
    input_path = args.data_path if args.data_path is not None else args.input_path
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions_to_run = ALL_CONDITIONS if args.condition == "all" else [args.condition]

    # Pre-check checkpoints for adapter conditions before executing Stage 1
    if stage in ("generate", "all"):
        for cond in conditions_to_run:
            if cond in ADAPTER_CONDITIONS:
                cp = args.non_recurrent_checkpoint if cond == "non_recurrent" and args.non_recurrent_checkpoint is not None else args.checkpoint
                verify_adapter_checkpoint(cp, cond)

    summary_file = args.output_dir / "empirical_baselines_summary.json"

    # =========================================================================
    # STAGE 1: Target-Free Prediction Generation
    # =========================================================================
    if stage in ("generate", "all"):
        print("\n" + "#" * 80)
        print("  STAGE 1: GENERATING PREDICTIONS (TARGET-FREE INFERENCE)")
        print(f"  Input artifact: {input_path}")
        print(f"  Output dir:     {args.output_dir}")
        print(f"  Conditions:     {conditions_to_run}")
        print("#" * 80)

        # Lazy initialize models
        print("[*] Loading Gemma 4 12B backbone on Apple Silicon Metal GPU...")
        manifest = ModelManifest.gemma_4_12b_it()
        backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
        backbone.freeze()

        needs_decoder = any(c != "direct_frozen" for c in conditions_to_run)
        needs_adapter = any(c in ADAPTER_CONDITIONS for c in conditions_to_run)

        decoder = (
            GemmaCausalPrefixDecoder(
                backbone=backbone,
                prefix_dim=3840,
                hidden_dim=3840,
                conditioning_mode=args.conditioning_mode,
            )
            if needs_decoder
            else None
        )

        recurrent_adapter = None
        non_rec_adapter = None
        if needs_adapter:
            has_recurrent = any(c in ADAPTER_CONDITIONS and c != "non_recurrent" for c in conditions_to_run)
            has_non_recurrent = "non_recurrent" in conditions_to_run

            if has_recurrent:
                recurrent_adapter = GemmaRecurrentAdapter(dim=3840, num_slots=16, num_layers=1, deliberation_steps=4)
                if args.checkpoint is not None and args.checkpoint.exists():
                    print(f"[*] Loading recurrent adapter weights from {args.checkpoint}...")
                    load_adapter_and_injection_weights(args.checkpoint, recurrent_adapter, decoder=decoder)

            if has_non_recurrent:
                non_rec_adapter = GemmaNonRecurrentAdapter(dim=3840, num_slots=16, intermediate_dim=13440)
                non_rec_cp = args.non_recurrent_checkpoint if args.non_recurrent_checkpoint is not None else args.checkpoint
                if non_rec_cp is not None and non_rec_cp.exists():
                    print(f"[*] Loading non-recurrent adapter weights from {non_rec_cp}...")
                    load_adapter_and_injection_weights(non_rec_cp, non_rec_adapter, decoder=decoder)

        for cond in conditions_to_run:
            print(f"\n---> Generating predictions for condition: {cond.upper()}")
            active_adapter = non_rec_adapter if cond == "non_recurrent" else recurrent_adapter
            active_cp = args.non_recurrent_checkpoint if cond == "non_recurrent" and args.non_recurrent_checkpoint is not None else args.checkpoint
            pred_file, sidecar_file, sha256_hash = generate_predictions(
                inputs=input_path,
                condition=cond,
                output_dir=args.output_dir,
                checkpoint_path=active_cp,
                backbone=backbone,
                adapter=active_adapter,
                decoder=decoder,
                max_tokens=args.max_tokens,
                limit=args.limit,
                seed=args.seed,
                conditioning_mode=args.conditioning_mode,
            )
            print(f"  [✓] Sealed predictions: {pred_file}")
            print(f"  [✓] SHA-256 sidecar:    {sidecar_file} ({sha256_hash[:16]}...)")

    # =========================================================================
    # STAGE 2: Post-Hoc Cryptographically Verified Scoring
    # =========================================================================
    if stage in ("score", "all"):
        print("\n" + "#" * 80)
        print("  STAGE 2: SCORING PREDICTIONS (POST-HOC VERIFICATION)")
        print(f"  Answer keys:    {args.keys_path} (Quarantined)")
        print(f"  Summary file:   {summary_file}")
        print("#" * 80)

        verifier = ProceduralVerifier()

        for cond in conditions_to_run:
            if args.prediction_path is not None:
                pred_file = args.prediction_path
            else:
                pred_file = args.output_dir / f"predictions_{cond}.json"

            if not pred_file.exists():
                print(f"  [!] Prediction file not found for condition '{cond}': {pred_file}. Skipping score.")
                continue

            print(f"\n---> Scoring predictions for condition: {cond.upper()}")
            summary_artifact, summary_path = score_predictions(
                predictions_path=pred_file,
                answer_keys_path=args.keys_path,
                output_dir=args.output_dir,
                summary_path=summary_file,
                verifier=verifier,
            )
            cond_metrics = summary_artifact.conditions[cond]
            print(f"  [✓] Exact Match:  {cond_metrics.exact_match_pct}% ({cond_metrics.exact_match_count}/{cond_metrics.sample_count})")
            print(f"  [✓] Terminal:     {cond_metrics.terminal_match_pct}% ({cond_metrics.terminal_match_count}/{cond_metrics.sample_count})")
            print(f"  [✓] Valid JSON:   {cond_metrics.valid_json_pct}%")
            print(f"  [✓] Repetition:   max={cond_metrics.max_4gram_repetition}, mean={cond_metrics.mean_4gram_repetition}")
            print(f"  [✓] Latency p50:  {cond_metrics.latency.median_ms} ms")

        # Print Decision Table
        print_decision_table(summary_file)

    # =========================================================================
    # STAGE 3: Paired Bootstrap Analysis
    # =========================================================================
    if args.bootstrap:
        print("\n" + "#" * 80)
        print("  STAGE 3: PAIRED BOOTSTRAP ANALYSIS (1,000 RESAMPLES & PERMUTATION)")
        print(f"  Predictions dir: {args.output_dir}")
        print(f"  Answer keys:     {args.keys_path}")
        print("#" * 80)
        out_f, sidecar_f, art = run_bootstrap_analysis(
            predictions_dir=args.output_dir,
            keys_path=args.keys_path,
            output_file=args.output_dir / "bootstrap_analysis.json",
            summary_file=summary_file,
            seed=args.seed,
        )
        print(f"  [✓] Bootstrap analysis: {out_f}")
        print(f"  [✓] SHA-256 sidecar:    {sidecar_f}")
        print(f"  [✓] Matched samples:    {art['provenance']['matched_sample_count']}")
        print(f"  [✓] Evaluated pairs:    {len(art['pairwise_comparisons'])}")


if __name__ == "__main__":
    main()
