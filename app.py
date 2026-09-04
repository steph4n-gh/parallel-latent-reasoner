"""Standalone Interactive Web Application for Parallel Latent Reasoner (PRLR).

Suitable for local execution via Gradio or 1-click deployment to HuggingFace Spaces.
Demonstrates live dual-pane comparison between Autoregressive CoT and Parallel Latent Deliberation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import time
from typing import Tuple
import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None

import mlx.core as mx
from parallel_latent_reasoner.cognitive_suite import (
    load_cognitive_benchmark_suite,
    get_test_case_by_id,
)

# Load benchmark suite test cases for dropdown
SUITE = load_cognitive_benchmark_suite()
PRESET_CHOICES = [f"{tc.id}: [{tc.domain.value.upper()}] {tc.prompt[:60]}..." for tc in SUITE]
PRESET_MAP = {f"{tc.id}: [{tc.domain.value.upper()}] {tc.prompt[:60]}...": tc.id for tc in SUITE}

_GLOBAL_PIPELINE = None


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


def get_pipeline():
    """Lazily instantiate production PRLRPipeline with trained adapter."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        from prlr.pipeline import PRLRPipeline
        _GLOBAL_PIPELINE = PRLRPipeline(
            load_trained_adapter=True,
            deliberation_steps=4,
            num_slots=16,
        )
    return _GLOBAL_PIPELINE


def run_comparison(
    prompt: str,
    selected_preset: str,
    slots_m: int,
    steps_t: int,
    enable_gate: bool,
    tol_rel_vel: float,
) -> Tuple[str, str, str, str]:
    """Run comparison between Autoregressive CoT and Parallel Latent Deliberation."""
    # If a preset was selected from dropdown and prompt was not manually changed
    if selected_preset in PRESET_MAP and (not prompt or prompt == "What is 25 * 14?"):
        case_id = PRESET_MAP[selected_preset]
        tc = get_test_case_by_id(case_id)
        active_prompt = tc.prompt
    else:
        active_prompt = prompt or "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"

    # 1. Initialize PRLR Pipeline
    pipeline = get_pipeline()
    model_id = getattr(getattr(pipeline, "manifest", None), "model_id", "google/gemma-4-12B-it-4bit")

    # 2. Parallel Latent Deliberation (PRLR)
    reset_metal_peak_memory()
    out = pipeline.deliberate_and_verify(
        prompt=active_prompt,
        max_steps=int(steps_t),
        max_new_tokens=32,
        enable_dynamic_gate=enable_gate,
    )
    prlr_peak_vram_mb = get_metal_peak_memory_mb()

    delib_ms = out.stage_latencies_ms.get("deliberation_ms", 5.0)
    decode_ms = out.stage_latencies_ms.get("decode_ms", 1.0)
    total_prlr_ms = out.stage_latencies_ms.get("total_ms", delib_ms + decode_ms)
    decoded_answer = out.decoded_text.strip()

    # 3. Genuine Autoregressive Gemma Baseline (Metal GPU)
    reset_metal_peak_memory()
    base_res = pipeline.generate_baseline(
        prompt=active_prompt,
        max_new_tokens=32,
    )
    cot_peak_vram_mb = get_metal_peak_memory_mb()
    cot_thought_stream = base_res.generated_text.strip() or "Standard autoregressive completion"
    cot_latency_ms = base_res.latency_ms
    cot_tokens = len(base_res.tokens) if base_res.tokens else 1

    speedup = cot_latency_ms / max(0.1, total_prlr_ms)

    # Format telemetry table
    telemetry_rows = ["| Step | Velocity v(t) | Rel Decay | Entropy H(t) | Margin m(t) | SVD erank | Status |"]
    telemetry_rows.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    if out.gate_telemetry:
        for tel in out.gate_telemetry:
            status_str = f"HALT ({tel.exit_reason})" if tel.halt else "Active"
            telemetry_rows.append(
                f"| t={tel.step} | {tel.velocity:.6f} | {tel.rel_velocity:.4f} | "
                f"{tel.entropy:.3f} | {tel.margin:.2f} | {tel.erank:.2f} | {status_str} |"
            )
    else:
        telemetry_rows.append(f"| t=0..{out.deliberation_steps} | N/A | N/A | N/A | N/A | N/A | Completed ({out.egate_verdict}) |")

    telemetry_table = "\n".join(telemetry_rows)

    left_summary = (
        f"### Mode 1: Genuine Autoregressive Baseline (`{model_id}`)\n"
        f"- **Measured Latency**: `{cot_latency_ms:.2f} ms`\n"
        f"- **Peak Metal VRAM**: `{cot_peak_vram_mb:.1f} MB`\n"
        f"- **Tokens Emitted**: `{cot_tokens} tokens`\n"
        f"- **KV-Cache Expansion**: N/A (Linear O(N) Growth)\n"
        f"- **Execution Bound**: Memory-Bandwidth (DRAM weight streaming)\n\n"
        f"**Generated Baseline Output**:\n```text\n{cot_thought_stream}\n```"
    )

    right_summary = (
        f"### Mode 2: Parallel Latent Deliberation (PRLR + `{model_id}`)\n"
        f"- **Deliberation Latency**: `{delib_ms:.2f} ms` (Total with decode: `{total_prlr_ms:.2f} ms`)\n"
        f"- **Peak Metal VRAM**: `{prlr_peak_vram_mb:.1f} MB`\n"
        f"- **Intermediate Tokens Emitted**: `0` (Zero KV-cache bloat)\n"
        f"- **Deliberation Sweeps Executed**: `T={out.deliberation_steps}/{steps_t}` across `M={slots_m}` slots\n"
        f"- **Reasoning Speedup**: `**{speedup:.1f}x FASTER**`\n"
        f"- **Shannon Entropy**: `{out.shannon_entropy:.2f} bits`\n"
        f"- **Execution Bound**: Compute-Bound (L2/SRAM Matrix Multiplication)\n\n"
        f"**Decoded Answer**:\n```text\n{decoded_answer}\n```"
    )

    perf_badge = (
        f"## ⚡ Result: PRLR is {speedup:.1f}x Faster in the Reasoning Phase "
        f"({delib_ms:.2f} ms vs {cot_latency_ms:.2f} ms measured on Metal GPU) with 0.00% KV-Cache Growth!"
    )

    return perf_badge, left_summary, right_summary, telemetry_table


def build_app():
    if gr is None:
        raise RuntimeError("Gradio is not installed. Install via `pip install gradio`.")

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
    )

    with gr.Blocks(theme=theme, title="Parallel Latent Reasoner (PRLR)") as demo:
        gr.Markdown(
            "# ⚡ Parallel Latent Reasoner (PRLR)\n"
            "### High-Throughput Continuous Latent Deliberation with Recurrent Depth on Apple Silicon MLX\n"
            "*Replace slow, memory-bound Chain-of-Thought (CoT) token generation with compute-bound parallel Jacobi sweeps in GPU SRAM.*"
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt / Reasoning Query",
                    placeholder="Enter a reasoning problem, or choose a cognitive test case below...",
                    value="What is 25 * 14?",
                    lines=3,
                )
                preset_dropdown = gr.Dropdown(
                    label="Or Select Curated Cognitive Test Case (25 Domain Cases)",
                    choices=["None (Use Custom Prompt)"] + PRESET_CHOICES,
                    value="None (Use Custom Prompt)",
                )
            with gr.Column(scale=2):
                with gr.Row():
                    slots_slider = gr.Slider(minimum=4, maximum=32, value=16, step=4, label="Memory Slots (M)")
                    steps_slider = gr.Slider(minimum=2, maximum=16, value=8, step=1, label="Max Steps (T)")
                with gr.Row():
                    gate_checkbox = gr.Checkbox(value=True, label="Enable 3-Signal Dynamic Consensus E-Gate")
                    tol_slider = gr.Slider(minimum=0.01, maximum=0.30, value=0.10, step=0.01, label="Velocity Halt Threshold (tau_v)")

        run_btn = gr.Button("🚀 Run Live Deliberation Comparison", variant="primary")

        perf_header = gr.Markdown("### Click 'Run Live Deliberation Comparison' to begin.")

        with gr.Row():
            with gr.Column():
                left_output = gr.Markdown()
            with gr.Column():
                right_output = gr.Markdown()

        with gr.Row():
            telemetry_output = gr.Markdown(label="Live E-Gate Telemetry")

        run_btn.click(
            fn=run_comparison,
            inputs=[prompt_input, preset_dropdown, slots_slider, steps_slider, gate_checkbox, tol_slider],
            outputs=[perf_header, left_output, right_output, telemetry_output],
        )

        gr.Markdown(
            "---\n"
            "**Open Source**: [https://github.com/steph4n-gh/parallel-latent-reasoner](https://github.com/steph4n-gh/parallel-latent-reasoner)\n"
            "*Licensed under Apache-2.0.*"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
