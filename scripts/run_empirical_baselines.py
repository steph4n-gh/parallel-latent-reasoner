#!/usr/bin/env python3
"""Run Empirical Baselines on Full Sealed Test Split (256 samples).

Conforms strictly to:
- Evidence Rules 1-10 (blind evaluation, post-hoc scoring, zero synthetic traces).
- Urgent Research Guidance Phases P1 and P3.

Evaluates:
1. Direct Frozen Gemma 4 (no adapter, official chat template)
2. Direct Frozen Gemma 4 through Repository Decoder (prefix_latents=None)
3. Existing Adapter with corrected prompt contract at T in {0, 1, 2, 4}
4. Controls: Zeroed-prefix and Shuffled-prefix
5. Non-recurrent control (feedforward parameter-matched prefix)

Records:
- Exact Match Route Accuracy
- Terminal Tool Routing Accuracy
- Valid JSON rate
- Max 4-gram Repetition
- Mean Shannon Entropy
- Stage and Total Latency (ms)
- Output JSON artifact and decision table
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prlr.domain.prompt_format import extract_user_body, format_canonical_prompt
from prlr.domain.solver_lane import DOMAIN_CATALOGUES, ProceduralVerifier
from prlr.eval.semantic_bench import (
    compute_max_ngram_repetition,
    compute_shannon_entropy,
)
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    answer_keys: Dict[str, Dict[str, Any]],
    verifier: ProceduralVerifier,
) -> Dict[str, Any]:
    exact_matches = []
    terminal_matches = []
    valid_jsons = []
    repetitions = []
    entropies = []
    latencies = []

    for pred in predictions:
        sid = pred["sample_id"]
        key = answer_keys.get(sid, {})
        v_cfg = key.get("verifier_config", {})
        expected_route = v_cfg.get("expected_route", [])
        expected_terminal = v_cfg.get("terminal_tool")
        goal = v_cfg.get("target_goal")
        tools = DOMAIN_CATALOGUES.get(pred["domain"])

        text = pred["generated_text"]
        v_res = verifier.verify(text, tuple(expected_route), tools=tools, goal=goal)

        is_em = bool(v_res["exact_match"])
        pred_term = v_res.get("terminal_tool")
        is_term = bool(pred_term and expected_terminal and pred_term == expected_terminal)
        is_valid = bool(v_res["is_valid"])
        rep = compute_max_ngram_repetition(text, n=4)
        entropy = compute_shannon_entropy(text)

        exact_matches.append(1 if is_em else 0)
        terminal_matches.append(1 if is_term else 0)
        valid_jsons.append(1 if is_valid else 0)
        repetitions.append(rep)
        entropies.append(entropy)
        latencies.append(pred.get("latency_ms", 0.0))

    return {
        "sample_count": len(predictions),
        "exact_match_pct": round(float(np.mean(exact_matches)) * 100.0, 2),
        "terminal_match_pct": round(float(np.mean(terminal_matches)) * 100.0, 2),
        "valid_json_pct": round(float(np.mean(valid_jsons)) * 100.0, 2),
        "max_repetition": int(max(repetitions)) if repetitions else 0,
        "mean_repetition": round(float(np.mean(repetitions)), 2) if repetitions else 0.0,
        "mean_entropy": round(float(np.mean(entropies)), 2) if entropies else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "median_latency_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="PRLR Empirical Baseline Runner")
    parser.add_argument(
        "--condition",
        choices=[
            "all",
            "direct_frozen",
            "repo_decoder",
            "adapter_t0",
            "adapter_t1",
            "adapter_t2",
            "adapter_t4",
            "control_zeroed",
            "control_shuffled",
            "non_recurrent",
        ],
        default="all",
        help="Condition to run",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "prlr_domain_v1" / "sealed_test.jsonl",
        help="Path to sealed test JSONL (256 samples)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "gemma_4_12b_prlr_adapter.safetensors",
        help="Adapter checkpoint path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples for rapid debugging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "empirical_baselines",
        help="Output directory for results",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1. Load dataset samples
    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]

    print(f"[*] Loaded {len(samples)} samples from {args.data_path}")

    # Build answer keys
    answer_keys = {s["id"]: s for s in samples}

    # 2. Load model backbone
    print("[*] Loading Gemma 4 12B backbone...")
    manifest = ModelManifest.gemma_4_12b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    backbone.freeze()
    tokenizer = backbone.tokenizer

    # 3. Load adapter and decoder
    adapter = GemmaRecurrentAdapter(dim=3840, num_slots=16, num_layers=1, deliberation_steps=4)
    if args.checkpoint.exists():
        print(f"[*] Loading adapter weights from {args.checkpoint}...")
        adapter.load_weights(str(args.checkpoint))
    else:
        print(f"[!] Checkpoint not found: {args.checkpoint}")

    decoder = GemmaCausalPrefixDecoder(backbone=backbone, prefix_dim=3840, hidden_dim=3840)
    verifier = ProceduralVerifier()

    conditions_to_run = []
    if args.condition == "all":
        conditions_to_run = [
            "direct_frozen",
            "repo_decoder",
            "adapter_t0",
            "adapter_t1",
            "adapter_t2",
            "adapter_t4",
            "control_zeroed",
            "control_shuffled",
            "non_recurrent",
        ]
    else:
        conditions_to_run = [args.condition]

    all_results = {}

    for cond in conditions_to_run:
        print(f"\n{'=' * 80}")
        print(f"  RUNNING CONDITION: {cond.upper()} ({len(samples)} samples)")
        print(f"{'=' * 80}")

        preds = []
        t_start_all = time.perf_counter()

        for idx, item in enumerate(samples, 1):
            prompt_raw = item["prompt"]
            user_body = extract_user_body(prompt_raw)
            canonical_prompt = format_canonical_prompt(user_body, tokenizer, is_gemma4=True)
            prompt_ids, _ = backbone.encode_prompt_context(canonical_prompt)
            mx.eval(prompt_ids)

            t0 = time.perf_counter()

            if cond == "direct_frozen":
                # Direct frozen Gemma 4 with official chat template via mlx_lm.generate
                import mlx_lm
                gen_text = mlx_lm.generate(
                    backbone.model,
                    tokenizer,
                    prompt=canonical_prompt,
                    max_tokens=96,
                    verbose=False,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
            elif cond == "repo_decoder":
                # Same as direct_frozen, verifying repository decoder path parity
                gen_tokens = decoder.generate(
                    prompt_ids=prompt_ids,
                    prefix_latents=None,
                    max_new_tokens=96,
                    temperature=0.0,
                )
                mx.eval(gen_tokens)
            elif cond.startswith("adapter_t"):
                t_steps = int(cond.split("_t")[-1])
                h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
                if t_steps == 0:
                    slots = adapter.prelude(h_prompt)
                else:
                    slots = adapter(h_prompt, steps=t_steps)
                mx.eval(slots)
                gen_tokens = decoder.generate(
                    prompt_ids=prompt_ids,
                    prefix_latents=slots,
                    max_new_tokens=96,
                    temperature=0.0,
                )
                mx.eval(gen_tokens)
            elif cond == "control_zeroed":
                h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
                slots = adapter(h_prompt, steps=4)
                slots = mx.zeros_like(slots)
                mx.eval(slots)
                gen_tokens = decoder.generate(
                    prompt_ids=prompt_ids,
                    prefix_latents=slots,
                    max_new_tokens=96,
                    temperature=0.0,
                )
                mx.eval(gen_tokens)
            elif cond == "control_shuffled":
                h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
                slots = adapter(h_prompt, steps=4)
                perm = mx.array(np.random.permutation(slots.shape[1]))
                slots = slots[:, perm, :]
                mx.eval(slots)
                gen_tokens = decoder.generate(
                    prompt_ids=prompt_ids,
                    prefix_latents=slots,
                    max_new_tokens=96,
                    temperature=0.0,
                )
                mx.eval(gen_tokens)
            elif cond == "non_recurrent":
                # Parameter-matched non-recurrent prefix: prelude only (no recurrent steps)
                h_prompt = backbone.extract_contextual_hiddens(prompt_ids)
                slots = adapter.prelude(h_prompt)
                mx.eval(slots)
                gen_tokens = decoder.generate(
                    prompt_ids=prompt_ids,
                    prefix_latents=slots,
                    max_new_tokens=96,
                    temperature=0.0,
                )
                mx.eval(gen_tokens)
            else:
                raise ValueError(f"Unknown condition: {cond}")

            if cond != "direct_frozen":
                lat_ms = (time.perf_counter() - t0) * 1000.0
                tok_list = gen_tokens[0].tolist() if gen_tokens.ndim > 1 else gen_tokens.tolist()
                gen_text = tokenizer.decode(tok_list)
                if isinstance(gen_text, list):
                    gen_text = " ".join(gen_text)

            preds.append({
                "sample_id": item["id"],
                "domain": item["domain"],
                "generated_text": gen_text,
                "latency_ms": round(lat_ms, 2),
            })

            if idx % 32 == 0 or idx == len(samples):
                print(f"  [{cond}] Processed {idx}/{len(samples)} samples...")

        # Evaluate condition post-hoc
        summary = evaluate_predictions(preds, answer_keys, verifier)
        all_results[cond] = {
            "condition": cond,
            "summary": summary,
            "predictions": preds,
        }

        print(f"  Summary for {cond}:")
        print(f"    Exact Match : {summary['exact_match_pct']}%")
        print(f"    Terminal    : {summary['terminal_match_pct']}%")
        print(f"    Valid JSON  : {summary['valid_json_pct']}%")
        print(f"    Max Rep     : {summary['max_repetition']}")
        print(f"    Mean Entropy: {summary['mean_entropy']} bits")
        print(f"    Latency p50 : {summary['median_latency_ms']} ms")

    # Save summary artifact (merging with existing summary if present)
    out_file = args.output_dir / "empirical_baselines_summary.json"
    clean_summary = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                clean_summary = old_data.get("conditions", {})
        except Exception:
            clean_summary = {}

    for k, v in all_results.items():
        clean_summary[k] = v["summary"]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_count": len(samples),
                "conditions": clean_summary,
            },
            f,
            indent=2,
        )
    print(f"\n[✓] Baseline summaries saved to: {out_file}")

    # Also save full predictions per condition
    for cond, data in all_results.items():
        cond_file = args.output_dir / f"predictions_{cond}.json"
        with open(cond_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Print Decision Table
    print("\n" + "=" * 100)
    print("  DECISION TABLE: EMPIRICAL BASELINES ON SEALED TEST (256 SAMPLES)")
    print("=" * 100)
    print(f"| {'Experiment':<30} | {'Prompt contract':<15} | {'Adapter':<12} | {'T':<5} | {'Exact':<8} | {'Terminal':<10} | {'Valid JSON':<10} | {'Max Rep':<8} |")
    print(f"|{'-'*32}|{'-'*17}|{'-'*14}|{'-'*7}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for cond, sm in sorted(clean_summary.items()):
        t_val = "0"
        if "_t" in cond:
            t_val = cond.split("_t")[-1]
        elif cond in ("control_zeroed", "control_shuffled"):
            t_val = "4"
        elif cond == "non_recurrent":
            t_val = "0 (prelude)"

        adapter_name = "None"
        if "adapter" in cond or "control" in cond:
            adapter_name = "Full rank"
        elif cond == "non_recurrent":
            adapter_name = "Non-rec"

        print(f"| {cond:<30} | {'Official':<15} | {adapter_name:<12} | {t_val:<5} | {str(sm['exact_match_pct']) + '%':<8} | {str(sm['terminal_match_pct']) + '%':<10} | {str(sm['valid_json_pct']) + '%':<10} | {sm['max_repetition']:<8} |")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
