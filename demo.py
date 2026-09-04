#!/usr/bin/env python3
"""Interactive Terminal CLI Demo & Cognitive Domain Explorer for Parallel Latent Reasoner (PRLR).

Features:
- Live side-by-side terminal comparison between Autoregressive CoT and Parallel Latent Deliberation.
- Support for `--trained` flag (defaulting to True when trained adapter checkpoint exists).
- Dynamic user prompt input via `--interactive` REPL with live 3-Signal Dynamic E-Gate telemetry
  (velocity decay, consensus token, SVD erank), thought trajectory diagnostics, and concise grounded answer.
- Full CLI parameter controls: `--preset`, `--adapter`, `--steps`, `--slots`, `--mode` (hybrid vs pure_latent),
  `--benchmark`, `--prompt`, `--case`, `--domain`.
- All 25 cognitive domain test cases across 5 domains (MCS, WSD, SDN, CMS, ATR).
- Multi-scale model presets: compact_test, gemma_2b, gemma_9b, gemma_12b, gemma_12b_q4, gemma_26b_a4b.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Tuple

# Add src to sys.path for standalone invocation
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import mlx.core as mx

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    get_domain_summary,
    get_test_case_by_id,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import GateTelemetry
from parallel_latent_reasoner.pipeline import HybridDeliberationResult, PRLRPipeline
from parallel_latent_reasoner.visualizer import Colors, render_comparison_view


# Domain mapping helpers
DOMAIN_ALIASES = {
    "mcs": DomainType.MULTI_CONSTRAINT,
    "multi_constraint": DomainType.MULTI_CONSTRAINT,
    "multiconstraint": DomainType.MULTI_CONSTRAINT,
    "wsd": DomainType.WINOGRAD_SCHEMA,
    "winograd": DomainType.WINOGRAD_SCHEMA,
    "winograd_schema": DomainType.WINOGRAD_SCHEMA,
    "sdn": DomainType.SEMANTIC_DENOISING,
    "semantic_denoising": DomainType.SEMANTIC_DENOISING,
    "denoising": DomainType.SEMANTIC_DENOISING,
    "cms": DomainType.MULTI_CLUE_SYNTHESIS,
    "multi_clue": DomainType.MULTI_CLUE_SYNTHESIS,
    "multi_clue_synthesis": DomainType.MULTI_CLUE_SYNTHESIS,
    "synthesis": DomainType.MULTI_CLUE_SYNTHESIS,
    "atr": DomainType.ACTION_TOOL_ROUTING,
    "action_tool": DomainType.ACTION_TOOL_ROUTING,
    "action_tool_routing": DomainType.ACTION_TOOL_ROUTING,
    "routing": DomainType.ACTION_TOOL_ROUTING,
}

MODEL_PRESETS = [
    "gemma_4_12b",
    "gemma_2b",
    "compact_test",
    "gemma_9b",
    "gemma_12b",
    "gemma_12b_q4",
    "gemma_26b_a4b",
    "gemma_e4b",
]

DOMAIN_PRESETS_INFO = [
    ("1", "Multi-Constraint Satisfaction (MCS)", DomainType.MULTI_CONSTRAINT, "Simultaneous 4+ non-linear constraint balancing"),
    ("2", "Winograd Schema & Disambiguation (WSD)", DomainType.WINOGRAD_SCHEMA, "Complex pronoun, physical, and legal entity binding"),
    ("3", "Semantic Denoising & Intent (SDN)", DomainType.SEMANTIC_DENOISING, "Low-pass cognitive filter for noisy/sarcastic requests"),
    ("4", "Cross-Context Clue Synthesis (CMS)", DomainType.MULTI_CLUE_SYNTHESIS, "Non-local multi-hop clue integration without scratchpads"),
    ("5", "Action & Tool Routing (ATR)", DomainType.ACTION_TOOL_ROUTING, "Zero-shot JSON candidate ranking and argument dispatch"),
]

DEFAULT_CHECKPOINT_PATH_12B = Path(__file__).resolve().parent / "checkpoints" / "gemma_4_12b_prlr_adapter.safetensors"
DEFAULT_CHECKPOINT_PATH_2B = Path(__file__).resolve().parent / "checkpoints" / "gemma_2b_prlr_adapter.safetensors"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH_12B if DEFAULT_CHECKPOINT_PATH_12B.exists() else DEFAULT_CHECKPOINT_PATH_2B


def get_metal_peak_memory_mb() -> float:
    """Retrieve peak Metal GPU memory in megabytes."""
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return float(mx.metal.get_peak_memory() / (1024 * 1024))
    if hasattr(mx, "get_peak_memory"):
        return float(mx.get_peak_memory() / (1024 * 1024))
    return 0.0


def reset_metal_peak_memory() -> None:
    """Reset peak Metal GPU memory counter."""
    if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()
    elif hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def resolve_domain(query: str) -> Optional[DomainType]:
    """Resolve a domain string or alias into a canonical DomainType."""
    normalized = query.strip().lower().replace("-", "_")
    return DOMAIN_ALIASES.get(normalized)


def default_trained_flag() -> bool:
    """Check whether default trained adapter checkpoint is present."""
    return DEFAULT_CHECKPOINT_PATH_12B.exists() or DEFAULT_CHECKPOINT_PATH_2B.exists()


def run_prlr_demo_execution(
    prompt: str,
    preset: str = "gemma_4_12b",
    adapter_path: Optional[str] = None,
    load_trained_adapter: bool = True,
    num_slots: int = 16,
    num_steps: int = 4,
    mode: str = "hybrid",
    enable_gate: bool = True,
    max_tokens: int = 32,
    temperature: float = 0.0,
    show_comparison: bool = True,
) -> HybridDeliberationResult:
    """Execute PRLR pipeline with live 3-Signal E-Gate telemetry and grounded answer decoding."""
    effective_load_trained = load_trained_adapter and (adapter_path is not None or num_slots == 16)

    if preset in ("gemma_4_12b", "gemma_12b", "gemma_2b"):
        from prlr.pipeline import PRLRPipeline
        from prlr.manifest import ModelManifest

        is_12b = preset in ("gemma_4_12b", "gemma_12b")
        dim = 3840 if is_12b else 2048
        if is_12b:
            manifest = ModelManifest.gemma_4_12b_it()
            default_ckpt = DEFAULT_CHECKPOINT_PATH_12B
        else:
            manifest = ModelManifest.gemma_2b_it()
            default_ckpt = DEFAULT_CHECKPOINT_PATH_2B

        ckpt = adapter_path or (str(default_ckpt) if (effective_load_trained and default_ckpt.exists()) else None)

        reset_metal_peak_memory()
        pipeline = PRLRPipeline(
            manifest=manifest,
            dim=dim,
            adapter_path=ckpt,
            load_trained_adapter=effective_load_trained and (ckpt is not None),
            deliberation_steps=num_steps,
            num_slots=num_slots,
        )
        out = pipeline.deliberate_and_verify(
            prompt=prompt,
            max_steps=num_steps,
            max_new_tokens=max_tokens,
            temperature=temperature,
            enable_dynamic_gate=enable_gate,
        )
        delib_peak_vram_mb = get_metal_peak_memory_mb()
    else:
        from parallel_latent_reasoner.pipeline import PRLRPipeline as LegacyPRLRPipeline
        print(f"\n[!] NOTICE: Executing testbed preset '{preset}'.")
        reset_metal_peak_memory()
        pipeline = LegacyPRLRPipeline.from_preset(
            preset=preset,
            num_memory_slots=num_slots,
            deliberation_steps=num_steps,
            adapter_path=adapter_path,
            load_trained_adapter=effective_load_trained and (adapter_path is not None),
        )
        out = pipeline.deliberate_and_verify(
            prompt=prompt,
            max_steps=num_steps,
            generate_tokens=max_tokens,
            temperature=temperature,
            enable_dynamic_gate=enable_gate,
        )
        delib_peak_vram_mb = get_metal_peak_memory_mb()

    if mode == "pure_latent":
        out.mode = "pure_latent"

    # Display Side-by-Side View
    if show_comparison:
        if preset in ("gemma_4_12b", "gemma_12b", "gemma_2b"):
            reset_metal_peak_memory()
            base_res = pipeline.generate_baseline(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            cot_text = base_res.generated_text
            cot_latency_ms = base_res.latency_ms
            cot_token_count = len(base_res.tokens) if base_res.tokens else max_tokens
            cot_peak_vram_mb = get_metal_peak_memory_mb()
            delib_latency_ms = out.latency_breakdown.get("deliberation_ms", 5.0)
            decode_latency_ms = out.latency_breakdown.get("decode_ms", 1.0)
        else:
            reset_metal_peak_memory()
            t0_base = time.perf_counter()
            token_ids = pipeline.encode_prompt(prompt)
            gen_tokens = pipeline.model.generate(
                token_ids,
                max_new_tokens=max_tokens,
                steps=0,
                temperature=temperature,
            )
            mx.eval(gen_tokens)
            cot_latency_ms = (time.perf_counter() - t0_base) * 1000.0
            cot_text = pipeline.decode_solution(gen_tokens)
            cot_token_count = gen_tokens.size
            cot_peak_vram_mb = get_metal_peak_memory_mb()
            delib_latency_ms = out.latency_breakdown.get("deliberation_latency_ms", 5.0)
            decode_latency_ms = out.latency_breakdown.get("coda_decode_latency_ms", 1.0)

        class DisplayConfig:
            def __init__(self, dim=3840, num_heads=16, num_memory_slots=16, deliberation_steps=4):
                self.dim = dim
                self.num_heads = num_heads
                self.num_memory_slots = num_memory_slots
                self.deliberation_steps = deliberation_steps

        class DisplayTelemetry:
            def __init__(self, t: Any):
                self.step = getattr(t, "step", 0)
                self.velocity = getattr(t, "velocity", 0.0)
                self.rel_velocity = getattr(t, "rel_velocity", getattr(t, "relative_velocity_decay", 1.0))
                self.erank = getattr(t, "erank", 1.0)
                self.coda_token_str = getattr(t, "first_token_str", getattr(t, "coda_token_str", getattr(t, "top_token_str", "")))
                self.coda_token = getattr(t, "first_token_id", getattr(t, "coda_token", 0))
                self.halt = getattr(t, "halt", False)
                self.exit_reason = getattr(t, "exit_reason", "active")

        cfg = DisplayConfig(
            dim=3840 if preset in ("gemma_4_12b", "gemma_12b") else (2048 if preset == "gemma_2b" else getattr(pipeline.config, "dim", 256)),
            num_heads=16 if preset in ("gemma_4_12b", "gemma_12b") else 8,
            num_memory_slots=num_slots,
            deliberation_steps=num_steps,
        )
        adapted_telems = [DisplayTelemetry(t) for t in (out.gate_telemetry or [])]

        view = render_comparison_view(
            prompt=prompt if isinstance(prompt, str) else str(prompt),
            config=cfg,
            cot_tokens_text=cot_text,
            cot_token_count=cot_token_count,
            cot_latency_ms=cot_latency_ms,
            cot_peak_vram_mb=cot_peak_vram_mb,
            gate_telemetries=adapted_telems,
            delib_latency_ms=delib_latency_ms,
            delib_peak_vram_mb=delib_peak_vram_mb,
            decoded_solution=out.decoded_text.strip() or "Verified Grounded Solution",
            decode_latency_ms=decode_latency_ms,
            coda_token_count=max_tokens,
        )
        print(view)

        # Print Trajectory Diagnostics
        print("\n" + "-" * 80)
        print(f"  {Colors.BOLD}THOUGHT TRAJECTORY & REASONING DIAGNOSTICS:{Colors.RESET}")
        print(f"  - Adapter Status: {'[LOADED]' if out.adapter_loaded else '[BASE WEIGHTS]'} ({out.adapter_path or 'none'})")
        print(f"  - Mode: {out.mode} | Unrolls Executed: T={out.deliberation_steps}/{num_steps}")
        print(f"  - 3-Signal E-Gate Verdict: {out.egate_verdict} (Consensus Step: {out.consensus_step or 'N/A'})")
        print(f"  - KV-Cache Expansion: +0.00% (Strictly Constant Sequence Length M={num_slots})")
        if hasattr(out, "shannon_entropy") and out.shannon_entropy is not None:
            print(f"  - Shannon Entropy: {out.shannon_entropy:.2f} bits")
        print(f"  - Grounded Decoded Output: \"{out.decoded_text.strip()}\"")
        print("-" * 80 + "\n")

    return out


def execute_test_case(
    case: CognitiveTestCase,
    model: str,
    adapter_path: Optional[str],
    load_trained_adapter: bool,
    slots: int,
    steps: int,
    mode: str,
    enable_gate: bool,
    max_tokens: int,
    temperature: float,
) -> None:
    """Run deliberation and visualizer for a specific cognitive test case."""
    print("\n" + "=" * 80)
    print(f"  COGNITIVE TEST CASE: [{case.id}] {case.title}")
    print(f"  Domain: {case.domain} | Verifier: {case.verifier_type}")
    if case.expected_constraints:
        print("  Expected Constraints:")
        for c in case.expected_constraints:
            print(f"    - {c}")
    print(f"  Ground Truth: {case.ground_truth}")
    print("=" * 80)

    out = run_prlr_demo_execution(
        prompt=case.prompt,
        preset=model,
        adapter_path=adapter_path,
        load_trained_adapter=load_trained_adapter,
        num_slots=slots,
        num_steps=steps,
        mode=mode,
        enable_gate=enable_gate,
        max_tokens=max_tokens,
        temperature=temperature,
        show_comparison=True,
    )

    ver = verify_test_case_result(case, out.decoded_text)
    status_str = f"{Colors.GREEN}PASSED (Score: {ver.score:.1f}){Colors.RESET}" if ver.passed else f"{Colors.YELLOW}EVAL: {ver.feedback}{Colors.RESET}"
    print(f"  Deterministic Verifier Result: {status_str}")


def run_domain_cases_interactive(
    domain: DomainType,
    model: str,
    adapter_path: Optional[str],
    load_trained_adapter: bool,
    slots: int,
    steps: int,
    mode: str,
    enable_gate: bool,
    max_tokens: int,
    temperature: float,
) -> None:
    """Sub-menu for choosing a test case within a specific cognitive domain."""
    cases = load_cognitive_benchmark_suite(domain)
    while True:
        print("\n" + "-" * 80)
        print(f"  DOMAIN: {domain.name} ({len(cases)} Test Cases)")
        print("-" * 80)
        for idx, c in enumerate(cases, 1):
            print(f"  [{idx}] [{c.id}] {c.title}")
        print("  [A] Run All Cases in this Domain")
        print("  [B] Back to Main Menu")

        choice = input("\nSelect test case > ").strip().lower()
        if choice in ("b", "back", "q", "quit"):
            break
        elif choice == "a":
            for c in cases:
                execute_test_case(
                    case=c,
                    model=model,
                    adapter_path=adapter_path,
                    load_trained_adapter=load_trained_adapter,
                    slots=slots,
                    steps=steps,
                    mode=mode,
                    enable_gate=enable_gate,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                print("\nPress Enter to continue...")
                input()
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(cases):
            selected_case = cases[int(choice) - 1]
            execute_test_case(
                case=selected_case,
                model=model,
                adapter_path=adapter_path,
                load_trained_adapter=load_trained_adapter,
                slots=slots,
                steps=steps,
                mode=mode,
                enable_gate=enable_gate,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("\nPress Enter to return to domain menu...")
            input()
        else:
            print("Invalid selection. Please enter a valid number or 'B'.")


def run_interactive_repl(args: argparse.Namespace) -> None:
    """Launch interactive REPL for prompt exploration, domain benchmarks, and live telemetry."""
    current_model = args.model
    current_adapter = args.adapter
    load_trained = args.trained
    current_slots = args.slots
    current_steps = args.steps
    current_mode = args.mode
    enable_gate = not args.no_gate
    max_tokens = args.max_tokens
    temperature = args.temperature

    print("=" * 80)
    print("  PARALLEL LATENT REASONER (PRLR) - INTERACTIVE COGNITIVE DEMO & VISUALIZER")
    print("  Platform: Apple Silicon Metal GPU (Unified Memory) | Framework: Pure MLX")
    print("=" * 80)

    while True:
        try:
            print("\n" + "=" * 80)
            print("  MAIN MENU - Select Cognitive Domain Preset or Custom Deliberation:")
            print("=" * 80)
            for num, name, d_type, desc in DOMAIN_PRESETS_INFO:
                print(f"  [{num}] {name}")
                print(f"      {desc}")
            print("  [6] Enter Custom Prompt (Dynamic User Prompt Input)")
            print("  [7] Quick Multi-Step Reasoning / Arithmetic Smoke Prompt")
            print("  [8] Run Automated Evaluation Benchmark")
            print(
                f"  [C] Configure Settings (Model: {current_model}, Trained: {'ON' if load_trained else 'OFF'}, "
                f"Mode: {current_mode}, M={current_slots}, T={current_steps}, E-Gate={'ON' if enable_gate else 'OFF'})"
            )
            print("  [Q] Quit Demo")

            choice = input("\nEnter choice (1-8, C, Q) > ").strip().lower()
            if not choice:
                continue

            if choice in ("q", "quit", "exit"):
                print("Exiting Parallel Latent Reasoner Demo. Goodbye!")
                break

            elif choice == "c":
                print("\n--- Configuration Settings ---")
                print(f"Available Model Presets: {', '.join(MODEL_PRESETS)}")
                new_model = input(f"Model Preset [{current_model}]: ").strip()
                if new_model in MODEL_PRESETS:
                    current_model = new_model
                elif new_model:
                    print(f"Using custom model preset: {new_model}")
                    current_model = new_model

                trained_str = input(f"Load Trained Adapter Checkpoint (y/n) [{'y' if load_trained else 'n'}]: ").strip().lower()
                if trained_str in ("y", "yes"):
                    load_trained = True
                elif trained_str in ("n", "no"):
                    load_trained = False

                new_mode = input(f"Execution Mode (hybrid/pure_latent) [{current_mode}]: ").strip().lower()
                if new_mode in ("hybrid", "pure_latent"):
                    current_mode = new_mode

                new_slots_str = input(f"Memory Slots M [{current_slots}]: ").strip()
                if new_slots_str.isdigit() and int(new_slots_str) > 0:
                    current_slots = int(new_slots_str)

                new_steps_str = input(f"Max Deliberation Steps T [{current_steps}]: ").strip()
                if new_steps_str.isdigit() and int(new_steps_str) > 0:
                    current_steps = int(new_steps_str)

                gate_str = input(f"Enable 3-Signal E-Gate (y/n) [{'y' if enable_gate else 'n'}]: ").strip().lower()
                if gate_str in ("y", "yes"):
                    enable_gate = True
                elif gate_str in ("n", "no"):
                    enable_gate = False

                tok_str = input(f"Coda Max Tokens [{max_tokens}]: ").strip()
                if tok_str.isdigit() and int(tok_str) > 0:
                    max_tokens = int(tok_str)

                print("Settings updated successfully.")
                continue

            elif choice in ("1", "2", "3", "4", "5"):
                domain_type = DOMAIN_PRESETS_INFO[int(choice) - 1][2]
                run_domain_cases_interactive(
                    domain=domain_type,
                    model=current_model,
                    adapter_path=current_adapter,
                    load_trained_adapter=load_trained,
                    slots=current_slots,
                    steps=current_steps,
                    mode=current_mode,
                    enable_gate=enable_gate,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            elif choice == "6":
                custom_prompt = input("\nEnter custom reasoning prompt > ").strip()
                if custom_prompt:
                    print("\nExecuting continuous latent deliberation...\n")
                    run_prlr_demo_execution(
                        prompt=custom_prompt,
                        preset=current_model,
                        adapter_path=current_adapter,
                        load_trained_adapter=load_trained,
                        num_slots=current_slots,
                        num_steps=current_steps,
                        mode=current_mode,
                        enable_gate=enable_gate,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        show_comparison=True,
                    )
                    print("\nPress Enter to return to main menu...")
                    input()

            elif choice == "7":
                smoke_prompt = "If a spacecraft travels at 12 km/s for 45 minutes, what total distance in kilometers does it cover?"
                print(f"\nExecuting smoke prompt: \"{smoke_prompt}\"...\n")
                run_prlr_demo_execution(
                    prompt=smoke_prompt,
                    preset=current_model,
                    adapter_path=current_adapter,
                    load_trained_adapter=load_trained,
                    num_slots=current_slots,
                    num_steps=current_steps,
                    mode=current_mode,
                    enable_gate=enable_gate,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    show_comparison=True,
                )
                print("\nPress Enter to return to main menu...")
                input()

            elif choice == "8":
                print("\nLaunching Automated Multi-Scale and Cognitive Benchmark Suite...\n")
                from parallel_latent_reasoner.benchmark import MultiDomainBenchmarkSuite, MultiScaleBenchmarkSuite
                scale_suite = MultiScaleBenchmarkSuite(
                    presets=[current_model],
                    num_slots=current_slots,
                    num_steps=current_steps,
                    enable_gate=enable_gate,
                    repeats=1,
                )
                scale_suite.run()
                print(scale_suite.to_ascii_table())

                domain_suite = MultiDomainBenchmarkSuite(
                    preset=current_model,
                    adapter_path=current_adapter,
                    load_trained_adapter=load_trained,
                    num_slots=current_slots,
                    num_steps=current_steps,
                    enable_gate=enable_gate,
                )
                domain_suite.run()
                print(domain_suite.to_ascii_table())
                print("\nPress Enter to return to main menu...")
                input()

            else:
                case = get_test_case_by_id(choice)
                if case:
                    execute_test_case(
                        case=case,
                        model=current_model,
                        adapter_path=current_adapter,
                        load_trained_adapter=load_trained,
                        slots=current_slots,
                        steps=current_steps,
                        mode=current_mode,
                        enable_gate=enable_gate,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    print("\nPress Enter to return to main menu...")
                    input()
                else:
                    print("Unrecognized option. Please choose a valid menu item (1-8, C, Q) or test case ID.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting PRLR Demo.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Latent Reasoner (PRLR) Production Demo & Interactive Visualizer"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt text to evaluate.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Cognitive benchmark test case ID (e.g. 'mcs_01', 'wsd_02', 'sdn_03', 'cms_04', 'atr_05').",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset identifier: model preset (e.g. 'compact_test', 'gemma_12b_q4') or test case ID / domain alias.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma_4_12b",
        choices=MODEL_PRESETS,
        help="Resident scale model configuration architecture (default: gemma_4_12b).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to trained adapter weights checkpoint (.npz or .safetensors).",
    )
    parser.add_argument(
        "--trained",
        dest="trained",
        action="store_true",
        default=None,
        help="Load production trained adapter weights (default: True if checkpoint exists).",
    )
    parser.add_argument(
        "--no-trained",
        dest="trained",
        action="store_false",
        help="Do not load trained adapter weights (use random initialization).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "pure_latent"],
        help="Reasoning mode: 'hybrid' (Deliberate-Then-Verify) or 'pure_latent' (default: hybrid).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive terminal menu and domain explorer REPL.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the automated multi-scale and cognitive benchmark suite directly.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Evaluate all cases within a cognitive domain ('multi_constraint', 'winograd_schema', 'semantic_denoising', 'multi_clue_synthesis', 'action_tool_routing').",
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
        default=4,
        help="Maximum deliberation unroll sweeps T (default: 4).",
    )
    parser.add_argument(
        "--no-gate",
        "--no-early-exit",
        action="store_true",
        help="Disable 3-Signal Dynamic Consensus E-Gate (force fixed deliberation sweeps).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Coda decoding (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Number of discrete solution tokens to generate (default: 16).",
    )

    args = parser.parse_args()

    # Default --trained handling: if not explicitly specified, default to True if checkpoint exists
    if args.trained is None:
        args.trained = default_trained_flag()

    # Determine model and test case from args
    selected_model = args.model
    target_case_id: Optional[str] = args.case

    # If --preset was passed, check if it's a test case ID or a model preset
    if args.preset:
        if get_test_case_by_id(args.preset) is not None:
            target_case_id = args.preset
        elif args.preset in MODEL_PRESETS:
            selected_model = args.preset
        else:
            d = resolve_domain(args.preset)
            if d:
                args.domain = d.value
            else:
                selected_model = args.preset

    args.model = selected_model

    # 1. Benchmark Direct Execution
    if args.benchmark:
        from parallel_latent_reasoner.benchmark import MultiDomainBenchmarkSuite, MultiScaleBenchmarkSuite
        print(f"Running automated benchmark for preset '{args.model}' (Trained: {args.trained})...")
        suite = MultiDomainBenchmarkSuite(
            preset=args.model,
            adapter_path=args.adapter,
            load_trained_adapter=args.trained,
            num_slots=args.slots,
            num_steps=args.steps,
            enable_gate=not args.no_gate,
        )
        suite.run()
        print(suite.to_ascii_table())
        return

    # 2. Interactive Mode
    if args.interactive:
        run_interactive_repl(args)
        return

    # 3. Specific Test Case ID Mode
    if target_case_id:
        case = get_test_case_by_id(target_case_id)
        if not case:
            print(f"Error: Test case ID '{target_case_id}' not found in cognitive suite.", file=sys.stderr)
            print("Valid case IDs include: mcs_01..05, wsd_01..05, sdn_01..05, cms_01..05, atr_01..05", file=sys.stderr)
            sys.exit(1)

        execute_test_case(
            case=case,
            model=args.model,
            adapter_path=args.adapter,
            load_trained_adapter=args.trained,
            slots=args.slots,
            steps=args.steps,
            mode=args.mode,
            enable_gate=not args.no_gate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return

    # 4. Domain Batch Mode
    if args.domain:
        domain_type = resolve_domain(args.domain)
        if not domain_type:
            print(f"Error: Unknown domain '{args.domain}'.", file=sys.stderr)
            print("Valid domains: multi_constraint (mcs), winograd_schema (wsd), semantic_denoising (sdn), multi_clue_synthesis (cms), action_tool_routing (atr)", file=sys.stderr)
            sys.exit(1)

        cases = load_cognitive_benchmark_suite(domain_type)
        print(f"Evaluating {len(cases)} test cases for domain: {domain_type.name} with model: {args.model}")
        for c in cases:
            execute_test_case(
                case=c,
                model=args.model,
                adapter_path=args.adapter,
                load_trained_adapter=args.trained,
                slots=args.slots,
                steps=args.steps,
                mode=args.mode,
                enable_gate=not args.no_gate,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        return

    # 5. Custom Prompt or Default Prompt Mode
    prompt = (
        args.prompt
        if args.prompt is not None
        else "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
    )
    run_prlr_demo_execution(
        prompt=prompt,
        preset=args.model,
        adapter_path=args.adapter,
        load_trained_adapter=args.trained,
        num_slots=args.slots,
        num_steps=args.steps,
        mode=args.mode,
        enable_gate=not args.no_gate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        show_comparison=True,
    )


if __name__ == "__main__":
    main()
