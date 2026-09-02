#!/usr/bin/env python3
"""Interactive Terminal CLI Demo & Cognitive Domain Explorer for Parallel Latent Reasoner (PRLR).

Features:
- Full support for all 25 cognitive domain test cases across 5 domains (MCS, WSD, SDN, CMS, ATR).
- Multi-scale model architectures: Gemma 4 12B Q4, 26B A4B MoE, Gemma 2B, 9B, 12B, and compact test.
- Interactive terminal menu for browsing domains, selecting test cases, adjusting M/T/E-Gate parameters,
  or testing custom prompts.
- Live side-by-side terminal comparison between Autoregressive CoT and Parallel Latent Deliberation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to sys.path for standalone invocation
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    get_domain_summary,
    get_test_case_by_id,
    load_cognitive_benchmark_suite,
)
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.visualizer import run_visualizer_demo

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
    "compact_test",
    "gemma_2b",
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


def resolve_domain(query: str) -> Optional[DomainType]:
    """Resolve a domain string or alias into a canonical DomainType."""
    normalized = query.strip().lower().replace("-", "_")
    return DOMAIN_ALIASES.get(normalized)


def execute_test_case(
    case: CognitiveTestCase,
    model: str,
    slots: int,
    steps: int,
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
    print("=" * 80)

    run_visualizer_demo(
        prompt=case.prompt,
        preset=model,
        num_slots=slots,
        num_steps=steps,
        enable_gate=enable_gate,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_domain_cases_interactive(
    domain: DomainType,
    model: str,
    slots: int,
    steps: int,
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
        print("  [A] Run All 5 Cases in this Domain")
        print("  [B] Back to Main Menu")

        choice = input("\nSelect test case > ").strip().lower()
        if choice in ("b", "back", "q", "quit"):
            break
        elif choice == "a":
            for c in cases:
                execute_test_case(
                    case=c,
                    model=model,
                    slots=slots,
                    steps=steps,
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
                slots=slots,
                steps=steps,
                enable_gate=enable_gate,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("\nPress Enter to return to domain menu...")
            input()
        else:
            print("Invalid selection. Please enter a valid number or 'B'.")


def run_interactive_repl(args: argparse.Namespace) -> None:
    """Launch interactive REPL for prompt exploration and domain benchmarks."""
    current_model = args.model
    current_slots = args.slots
    current_steps = args.steps
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
            print("  [6] Enter Custom Prompt")
            print("  [7] Quick Arithmetic / Logic Smoke Prompt")
            print(
                f"  [C] Configure Settings (Model: {current_model}, M={current_slots}, "
                f"T={current_steps}, E-Gate={'ON' if enable_gate else 'OFF'}, MaxTok={max_tokens})"
            )
            print("  [Q] Quit Demo")

            choice = input("\nEnter choice (1-7, C, Q) > ").strip().lower()
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
                    slots=current_slots,
                    steps=current_steps,
                    enable_gate=enable_gate,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            elif choice == "6":
                custom_prompt = input("\nEnter custom reasoning prompt > ").strip()
                if custom_prompt:
                    print("\nExecuting continuous latent deliberation...\n")
                    run_visualizer_demo(
                        prompt=custom_prompt,
                        preset=current_model,
                        num_slots=current_slots,
                        num_steps=current_steps,
                        enable_gate=enable_gate,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    print("\nPress Enter to return to main menu...")
                    input()

            elif choice == "7":
                smoke_prompt = "If a car travels 60 mph for 2.5 hours, how far does it go?"
                print(f"\nExecuting smoke prompt: \"{smoke_prompt}\"...\n")
                run_visualizer_demo(
                    prompt=smoke_prompt,
                    preset=current_model,
                    num_slots=current_slots,
                    num_steps=current_steps,
                    enable_gate=enable_gate,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                print("\nPress Enter to return to main menu...")
                input()

            else:
                # Check if user typed a direct test case ID (e.g. mcs_01)
                case = get_test_case_by_id(choice)
                if case:
                    execute_test_case(
                        case=case,
                        model=current_model,
                        slots=current_slots,
                        steps=current_steps,
                        enable_gate=enable_gate,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    print("\nPress Enter to return to main menu...")
                    input()
                else:
                    print("Unrecognized option. Please choose a valid menu item or test case ID.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting PRLR Demo.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Latent Reasoner (PRLR) Cognitive Benchmark CLI & Live Visualizer"
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
        help="Preset identifier: either a model preset (e.g. 'gemma_12b_q4', 'compact_test') or a test case ID (e.g. 'mcs_01').",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Evaluate all cases within a cognitive domain ('multi_constraint', 'winograd_schema', 'semantic_denoising', 'multi_clue_synthesis', 'action_tool_routing').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="compact_test",
        choices=MODEL_PRESETS,
        help="Resident scale model configuration architecture (default: compact_test).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive terminal menu and domain explorer REPL.",
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
            # Check domain alias
            d = resolve_domain(args.preset)
            if d:
                args.domain = d.value
            else:
                selected_model = args.preset

    args.model = selected_model

    # 1. Interactive Mode
    if args.interactive:
        run_interactive_repl(args)
        return

    # 2. Specific Test Case ID Mode
    if target_case_id:
        case = get_test_case_by_id(target_case_id)
        if not case:
            print(f"Error: Test case ID '{target_case_id}' not found in cognitive suite.", file=sys.stderr)
            print("Valid case IDs include: mcs_01..05, wsd_01..05, sdn_01..05, cms_01..05, atr_01..05", file=sys.stderr)
            sys.exit(1)

        execute_test_case(
            case=case,
            model=args.model,
            slots=args.slots,
            steps=args.steps,
            enable_gate=not args.no_gate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return

    # 3. Domain Batch Mode
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
                slots=args.slots,
                steps=args.steps,
                enable_gate=not args.no_gate,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        return

    # 4. Custom Prompt or Default Prompt Mode
    prompt = args.prompt if args.prompt is not None else "If a car travels 60 mph for 2.5 hours, how far does it go?"
    run_visualizer_demo(
        prompt=prompt,
        preset=args.model,
        num_slots=args.slots,
        num_steps=args.steps,
        enable_gate=not args.no_gate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
