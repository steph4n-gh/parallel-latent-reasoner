"""End-to-End Hybrid Deliberate-Then-Verify Pipeline for Parallel Latent Reasoner.

Orchestrates the complete dual-mode latent deliberation pipeline on Apple Silicon:
1. Mode 1 (Pure Latent Deliberation):
   - Prelude working memory slot initialization (M=16 slots, constant SRAM sequence length)
   - High-speed parallel Jacobi sweeps with AdaRMSNorm step conditioning and ReZero residual scaling
   - 3-Signal Dynamic Consensus E-Gate (velocity decay, Coda prediction consensus, SVD erank plateau)
2. Mode 2 (Hybrid 'Deliberate-Then-Verify' Execution):
   - Phase 1: High-speed parallel Jacobi sweeps in SRAM cache to resolve constraints and prune hypothesis space
   - Phase 2: Concise grounded discrete token decoding directly conditioned on the deliberated thought vector without intermediate CoT token bloat
3. Automatic loading of trained BPTT adapter checkpoints (.npz / .safetensors)
4. MLX JIT compilation (@mx.compile) for maximum Apple Silicon Metal GPU throughput
5. Structured telemetry with HybridDeliberationResult dataclass
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from prlr.compact.config import GemmaLatentConfig
from prlr.kernel.gates import DynamicDeliberationGate, GateTelemetry
from prlr.kernel.engine import DeliberationResult
from prlr.compact.scratch_model import MLXCompactGemmaModel
from prlr.kernel.telemetry import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    detect_limit_cycle,
)


@dataclass
class HybridDeliberationResult:
    """Structured result container for Hybrid Deliberate-Then-Verify execution."""

    prompt: str | list[int] | mx.array
    token_ids: mx.array
    decoded_text: str
    deliberation_steps: int
    final_states: mx.array
    consensus_step: int | None = None
    egate_verdict: str = "active"
    gate_telemetry: list[GateTelemetry] | None = None
    coda_logits: mx.array | None = None
    coda_tokens: list[int] | None = None
    trajectory_states: list[mx.array] | None = None
    effective_ranks: list[float] | None = None
    trajectory_analysis: TrajectoryAnalysis | None = None
    diagnostics: dict[str, Any] | None = None
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    memory_stats: dict[str, float] = field(default_factory=dict)
    adapter_loaded: bool = False
    adapter_path: str | None = None
    mode: str = "hybrid_deliberate_then_verify"

    @property
    def verified_response_text(self) -> str:
        """Contract property alias for decoded_text."""
        return self.decoded_text

    @property
    def deliberation_trajectory(self) -> list[mx.array] | None:
        """Contract property alias for trajectory_states."""
        return self.trajectory_states

    @property
    def latency_metrics(self) -> dict[str, float]:
        """Contract property alias for latency_breakdown."""
        return self.latency_breakdown

    @property
    def metrics(self) -> dict[str, float]:
        """Backward-compatibility alias for DeliberationPipelineOutput.metrics."""
        return self.latency_breakdown

    @property
    def thought_trajectory(self) -> list[mx.array] | None:
        """Alias for trajectory_states."""
        return self.trajectory_states


# Backwards compatibility alias
DeliberationPipelineOutput = HybridDeliberationResult


def _find_adapter_checkpoint(adapter_path: str | Path | None = None) -> Path | None:
    """Resolve adapter checkpoint path from explicit path or default project locations."""
    if adapter_path is not None:
        p = Path(adapter_path)
        if p.exists():
            return p
        # Check relative to current working directory
        p_cwd = Path.cwd() / adapter_path
        if p_cwd.exists():
            return p_cwd
        # Check relative to module package
        pkg_root = Path(__file__).resolve().parents[2]
        p_pkg = pkg_root / adapter_path
        if p_pkg.exists():
            return p_pkg
        p_pkg_checkpoints = pkg_root / "checkpoints" / Path(adapter_path).name
        if p_pkg_checkpoints.exists():
            return p_pkg_checkpoints
        return None

    # Search default candidates for trained adapter
    pkg_root = Path(__file__).resolve().parents[2]
    candidates = [
        pkg_root / "checkpoints/prlr_latent_adapter.npz",
        pkg_root / "checkpoints/prlr_latent_adapter.safetensors",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


class PRLRPipeline:
    """Production Hybrid Deliberate-Then-Verify and Pure Latent Reasoning Pipeline.

    Modes:
    1. Mode 1 (Pure Latent Deliberation):
       - Prelude projection initializes M working memory slots S^(0) in SRAM.
       - High-speed parallel Jacobi sweeps unroll across T steps.
       - 3-Signal Dynamic Consensus E-Gate autonomously detects representation convergence.
    2. Mode 2 (Hybrid Deliberate-Then-Verify):
       - Phase 1: Parallel Jacobi sweeps resolve constraints and prune hypothesis space in continuous latent space.
       - Phase 2: Concise grounded discrete token decoding directly conditioned on deliberated thought vector.
    """

    def __init__(
        self,
        model: MLXCompactGemmaModel | None = None,
        config: GemmaLatentConfig | None = None,
        tokenizer: Any | None = None,
        adapter_path: str | Path | None = None,
        load_trained_adapter: bool = False,
        compile_engine: bool = True,
        compile_decoder: bool = True,
    ):
        if model is not None:
            self.model = model
            self.config = model.config
        elif config is not None:
            self.config = config
            self.model = MLXCompactGemmaModel(config)
        else:
            self.config = GemmaLatentConfig.compact_test()
            self.model = MLXCompactGemmaModel(self.config)

        self.tokenizer = tokenizer
        self.adapter_loaded: bool = False
        self.adapter_path: str | None = None
        self.compile_engine: bool = compile_engine
        self.compile_decoder: bool = compile_decoder

        # Initialize MLX JIT compiled execution paths
        self._compiled_step_fn: Callable[..., mx.array] | None = None
        if self.compile_engine:
            self._compiled_step_fn = mx.compile(self.model.engine.step)

        # Automatic adapter loading
        if adapter_path is not None or load_trained_adapter:
            self.load_adapter(adapter_path=adapter_path)

    @classmethod
    def from_preset(
        cls,
        preset: str = "compact_test",
        tokenizer: Any | None = None,
        adapter_path: str | Path | None = None,
        load_trained_adapter: bool = False,
        compile_engine: bool = True,
        compile_decoder: bool = True,
        **kwargs: Any,
    ) -> PRLRPipeline:
        """Create a deliberation pipeline from a standard configuration preset."""
        preset_lower = preset.lower()
        if preset_lower in ("compact_test", "compact", "test"):
            config = GemmaLatentConfig.compact_test(**kwargs)
        elif preset_lower in ("gemma_2b", "2b"):
            config = GemmaLatentConfig.gemma_2b(**kwargs)
        elif preset_lower in ("gemma_9b", "9b"):
            config = GemmaLatentConfig.gemma_9b(**kwargs)
        elif preset_lower in ("gemma_12b", "12b"):
            config = GemmaLatentConfig.gemma_12b(**kwargs)
        elif preset_lower in ("gemma_e4b", "e4b", "4b"):
            config = GemmaLatentConfig.gemma_e4b(**kwargs)
        elif preset_lower in ("gemma_26b_a4b", "26b_a4b", "26b"):
            config = GemmaLatentConfig.gemma_26b_a4b(**kwargs)
        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Expected 'compact_test', 'gemma_2b', 'gemma_9b', 'gemma_12b', 'gemma_e4b', or 'gemma_26b_a4b'."
            )

        model = MLXCompactGemmaModel(config)
        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            adapter_path=adapter_path,
            load_trained_adapter=load_trained_adapter,
            compile_engine=compile_engine,
            compile_decoder=compile_decoder,
        )

    def load_adapter(self, adapter_path: str | Path | None = None) -> dict[str, mx.array]:
        """Load and bind trained PRLR adapter weights from .npz or .safetensors checkpoint."""
        resolved = _find_adapter_checkpoint(adapter_path)
        if resolved is None or not resolved.exists():
            target_str = str(adapter_path) if adapter_path is not None else "default checkpoint"
            raise FileNotFoundError(f"PRLR adapter checkpoint not found for: {target_str}")

        loaded_weights = self.model.load_adapter_weights(resolved)
        self.adapter_loaded = True
        self.adapter_path = str(resolved)

        # Refresh compiled step function after weight loading if engine JIT enabled
        if self.compile_engine:
            self._compiled_step_fn = mx.compile(self.model.engine.step)

        return loaded_weights

    def encode_prompt(self, prompt: str | mx.array | Sequence[int]) -> mx.array:
        """Parse prompt string, sequence of integers, or array into 2D [B, P] int32 array."""
        if isinstance(prompt, str):
            if self.tokenizer is not None:
                token_ids = mx.array([self.tokenizer.encode(prompt)], dtype=mx.int32)
            else:
                token_ids = mx.array([[ord(c) % self.config.vocab_size for c in prompt]], dtype=mx.int32)
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) > 0 and isinstance(prompt[0], (list, tuple)):
                token_ids = mx.array(list(prompt), dtype=mx.int32)
            else:
                token_ids = mx.array([list(prompt)], dtype=mx.int32)
        elif isinstance(prompt, mx.array):
            if prompt.ndim == 1:
                token_ids = prompt[None, :].astype(mx.int32)
            else:
                token_ids = prompt.astype(mx.int32)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if token_ids.shape[1] == 0:
            raise ValueError("Prompt cannot be empty (0 tokens).")

        return token_ids

    def deliberate(
        self,
        prompt_tokens: mx.array | str | Sequence[int],
        steps: int | None = None,
        enable_dynamic_gate: bool = True,
        gate: DynamicDeliberationGate | None = None,
        tol_rel_vel: float = 0.10,
        tol_erank_delta: float = 0.005,
        min_steps: int = 2,
        max_steps: int | None = None,
        patience: int = 1,
        return_trajectory: bool = False,
        compute_probes: bool = False,
        use_jit: bool | None = None,
    ) -> tuple[DeliberationResult, list[GateTelemetry] | None]:
        """Execute Mode 1: Pure Continuous Latent Deliberation with 3-Signal Dynamic Consensus E-Gate.

        Args:
            prompt_tokens: Input prompt token IDs [B, P] or text string.
            steps: Maximum deliberation sweeps T (defaults to config.deliberation_steps).
            enable_dynamic_gate: If True, uses the 3-Signal Dynamic Consensus E-Gate.
            gate: Optional pre-existing DynamicDeliberationGate instance.
            tol_rel_vel: Velocity decay threshold (default 0.10).
            tol_erank_delta: SVD effective rank plateau threshold (default 0.005).
            min_steps: Minimum deliberation steps before early exit is allowed (default 2).
            max_steps: Hard upper bound on unroll steps.
            patience: Number of consecutive consensus steps required to halt (default 1).
            return_trajectory: Whether to record all intermediate memory states S^(t).
            compute_probes: Whether to compute diagnostic probes on trajectory.
            use_jit: Explicit flag to use JIT compiled step (defaults to self.compile_engine).

        Returns:
            Tuple of (DeliberationResult, list of GateTelemetry or None).
        """
        token_ids = self.encode_prompt(prompt_tokens)
        max_T = (
            steps
            if steps is not None
            else (max_steps if max_steps is not None else self.config.deliberation_steps)
        )

        should_jit = self.compile_engine if use_jit is None else use_jit
        step_fn = self._compiled_step_fn if (should_jit and self._compiled_step_fn is not None) else self.model.engine.step

        # 1. Prelude projection: initialize M slots S^(0)
        slots, prompt_hiddens = self.model.prelude(token_ids)
        prompt_len = prompt_hiddens.shape[1]
        prompt_kv = self.model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

        curr = slots
        trajectory: list[mx.array] | None = (
            [curr] if (return_trajectory or enable_dynamic_gate or compute_probes) else None
        )

        active_gate: DynamicDeliberationGate | None = None
        if enable_dynamic_gate:
            if gate is not None:
                active_gate = gate
                active_gate.reset()
            else:
                active_gate = DynamicDeliberationGate(
                    tol_rel_vel=tol_rel_vel,
                    tol_erank_delta=tol_erank_delta,
                    min_steps=min_steps,
                    max_steps=max_T,
                    patience=patience,
                )

            # Evaluate initial Coda prediction at step 0
            coda_logits_0 = self.model.coda(curr, pool=True)
            coda_tok_0 = int(mx.argmax(coda_logits_0, axis=-1)[0].item())
            coda_str_0 = (
                self.tokenizer.decode([coda_tok_0])
                if self.tokenizer is not None
                else chr(coda_tok_0 % 128)
            )
            active_gate.update(curr, step=0, coda_token=coda_tok_0, coda_token_str=coda_str_0)

        steps_executed = 0
        for t in range(1, max_T + 1):
            curr = step_fn(
                curr,
                step_idx=t,
                prompt_kv=prompt_kv,
                prompt_len=prompt_len,
            )
            steps_executed = t

            if trajectory is not None:
                trajectory.append(curr)

            if active_gate is not None:
                # Read discrete coda token prediction at step t
                coda_logits_t = self.model.coda(curr, pool=True)
                coda_tok_t = int(mx.argmax(coda_logits_t, axis=-1)[0].item())
                coda_str_t = (
                    self.tokenizer.decode([coda_tok_t])
                    if self.tokenizer is not None
                    else chr(coda_tok_t % 128)
                )

                telemetry = active_gate.update(
                    curr,
                    step=t,
                    coda_token=coda_tok_t,
                    coda_token_str=coda_str_t,
                )
                if telemetry.halt:
                    break

        mx.eval(curr)
        final_trajectory = trajectory if return_trajectory else None
        gate_telemetry = active_gate.telemetry_history if active_gate is not None else None

        delib_result = DeliberationResult(
            final_states=curr,
            trajectory_states=final_trajectory,
            steps_executed=steps_executed,
        )
        return delib_result, gate_telemetry

    def deliberate_and_verify(
        self,
        prompt: str | mx.array | Sequence[int],
        max_steps: int = 8,
        generate_tokens: int = 64,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        return_diagnostics: bool = False,
        min_steps: int = 2,
        tol_rel_vel: float = 0.10,
        tol_erank_delta: float = 0.005,
        patience: int = 1,
        **gate_kwargs: Any,
    ) -> HybridDeliberationResult:
        """Execute Mode 2: Hybrid 'Deliberate-Then-Verify' Pipeline.

        Phase 1: High-speed parallel Jacobi sweeps in SRAM cache to resolve constraints and
                 prune hypothesis space with 3-Signal Dynamic Consensus E-Gate.
        Phase 2: Concise grounded discrete token decoding directly conditioned on the
                 deliberated thought vector without intermediate CoT token bloat.

        Args:
            prompt: Text prompt string, integer token ID list, or mx.array [B, P].
            max_steps: Maximum unroll sweeps T in SRAM working memory.
            generate_tokens: Number of discrete grounded solution tokens to decode.
            temperature: Sampling temperature (0.0 for greedy argmax).
            enable_dynamic_gate: Whether to allow early termination on 3-signal consensus.
            return_diagnostics: Whether to compute diagnostic probes, erank, and trajectory analysis.
            min_steps: Minimum deliberation steps before early exit is permitted.
            tol_rel_vel: Velocity decay threshold.
            tol_erank_delta: SVD effective rank plateau threshold.
            patience: Number of consecutive consensus steps required to halt.
            gate_kwargs: Additional parameters passed to deliberate().

        Returns:
            HybridDeliberationResult containing decoded solution text, token IDs,
            deliberation steps, consensus metrics, and detailed latency breakdowns.
        """
        t0 = time.perf_counter()

        # Track memory baseline
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        # 1. Parse prompt inputs
        token_ids = self.encode_prompt(prompt)
        mx.eval(token_ids)
        t_prefill_end = time.perf_counter()

        # Phase 1: High-Speed Parallel Continuous Latent Deliberation
        delib_res, gate_telemetry = self.deliberate(
            token_ids,
            steps=max_steps,
            enable_dynamic_gate=enable_dynamic_gate,
            tol_rel_vel=tol_rel_vel,
            tol_erank_delta=tol_erank_delta,
            min_steps=min_steps,
            max_steps=max_steps,
            patience=patience,
            return_trajectory=(return_diagnostics or enable_dynamic_gate),
            compute_probes=return_diagnostics,
            **gate_kwargs,
        )
        mx.eval(delib_res.final_states)
        t_delib_end = time.perf_counter()

        # Consensus step and exit reason extraction
        consensus_step: int | None = None
        egate_verdict = "disabled" if not enable_dynamic_gate else "active"
        if gate_telemetry is not None and len(gate_telemetry) > 0:
            last_tel = gate_telemetry[-1]
            egate_verdict = last_tel.exit_reason
            if last_tel.halt and last_tel.exit_reason == "3_signal_consensus":
                consensus_step = last_tel.step

        # Phase 2: Concise Grounded Discrete Token Decoding
        readout = self.model.coda.pool_readout(delib_res.final_states)
        B = token_ids.shape[0]
        generated_tokens: list[mx.array] = []
        curr_hidden = readout
        final_coda_logits: mx.array | None = None

        for i in range(generate_tokens):
            logits = self.model.coda.project_logits(curr_hidden)
            if i == 0:
                final_coda_logits = logits

            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                next_tok = mx.random.categorical(logits / temperature)[:, None]
            generated_tokens.append(next_tok)

            tok_embed = self.model.prelude.embed_prompt(next_tok)[:, 0, :]
            curr_hidden = self.model.coda.final_norm(curr_hidden + 0.1 * tok_embed)

        if len(generated_tokens) > 0:
            solution_ids = mx.concatenate(generated_tokens, axis=-1)
        else:
            solution_ids = mx.zeros((B, 0), dtype=mx.int32)

        mx.eval(solution_ids)
        t_decode_end = time.perf_counter()

        # Decode solution text
        decoded_text = self.decode_solution(solution_ids)

        # Diagnostic probes calculation
        erank_history: list[float] | None = None
        trajectory_analysis: TrajectoryAnalysis | None = None
        diagnostics: dict[str, Any] | None = None

        if return_diagnostics and delib_res.trajectory_states is not None:
            trajectory_analysis = analyze_deliberation_trajectory(
                delib_res.trajectory_states,
                compute_erank=True,
            )
            erank_history = trajectory_analysis.effective_ranks
            diagnostics = detect_limit_cycle(
                delib_res.trajectory_states,
                erank_history=erank_history,
            )

        # Extract predicted Coda tokens across trajectory if recorded
        coda_tokens: list[int] | None = None
        if gate_telemetry is not None:
            coda_tokens = [t.coda_token for t in gate_telemetry if t.coda_token is not None]

        # Latency breakdown
        prefill_ms = (t_prefill_end - t0) * 1000.0
        delib_ms = (t_delib_end - t_prefill_end) * 1000.0
        decode_ms = (t_decode_end - t_delib_end) * 1000.0
        total_ms = (t_decode_end - t0) * 1000.0
        tok_per_sec = (float(generate_tokens) / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        latency_breakdown = {
            "prefill_latency_ms": prefill_ms,
            "deliberation_latency_ms": delib_ms,
            "coda_decode_latency_ms": decode_ms,
            "total_latency_ms": total_ms,
            "throughput_tok_per_sec": tok_per_sec,
            "steps_executed": float(delib_res.steps_executed),
            "tokens_generated": float(generate_tokens),
        }

        # Memory statistics
        peak_bytes = float(getattr(mx, "get_peak_memory", lambda: 0.0)())
        active_bytes = float(getattr(mx, "get_active_memory", lambda: 0.0)())
        memory_stats = {
            "peak_memory_mb": peak_bytes / (1024.0 * 1024.0),
            "active_memory_mb": active_bytes / (1024.0 * 1024.0),
            "kv_cache_growth_pct": 0.0,
            "num_memory_slots": float(self.config.num_memory_slots),
        }

        final_trajectory = delib_res.trajectory_states if return_diagnostics else None

        return HybridDeliberationResult(
            prompt=prompt,
            token_ids=solution_ids,
            decoded_text=decoded_text,
            deliberation_steps=delib_res.steps_executed,
            final_states=delib_res.final_states,
            consensus_step=consensus_step,
            egate_verdict=egate_verdict,
            gate_telemetry=gate_telemetry,
            coda_logits=final_coda_logits,
            coda_tokens=coda_tokens,
            trajectory_states=final_trajectory,
            effective_ranks=erank_history,
            trajectory_analysis=trajectory_analysis,
            diagnostics=diagnostics,
            latency_breakdown=latency_breakdown,
            memory_stats=memory_stats,
            adapter_loaded=self.adapter_loaded,
            adapter_path=self.adapter_path,
            mode="hybrid_deliberate_then_verify",
        )

    def deliberate_then_verify(
        self,
        prompt: str | mx.array | Sequence[int],
        max_steps: int = 8,
        generate_tokens: int = 64,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        return_diagnostics: bool = False,
        **gate_kwargs: Any,
    ) -> HybridDeliberationResult:
        """Alias for deliberate_and_verify for seamless interface parity."""
        return self.deliberate_and_verify(
            prompt=prompt,
            max_steps=max_steps,
            generate_tokens=generate_tokens,
            temperature=temperature,
            enable_dynamic_gate=enable_dynamic_gate,
            return_diagnostics=return_diagnostics,
            **gate_kwargs,
        )

    def generate(
        self,
        prompt: str | mx.array | Sequence[int],
        max_new_tokens: int = 16,
        deliberation_steps: int | None = None,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        return_diagnostics: bool = False,
        **gate_kwargs: Any,
    ) -> HybridDeliberationResult:
        """Run complete end-to-end deliberation and discrete token decoding.

        Maintains full backwards compatibility with DeliberationPipelineOutput.
        """
        max_T = (
            deliberation_steps
            if deliberation_steps is not None
            else self.config.deliberation_steps
        )
        return self.deliberate_and_verify(
            prompt=prompt,
            max_steps=max_T,
            generate_tokens=max_new_tokens,
            temperature=temperature,
            enable_dynamic_gate=enable_dynamic_gate,
            return_diagnostics=return_diagnostics,
            **gate_kwargs,
        )

    def decode_solution(self, token_ids: mx.array | Sequence[int]) -> str:
        """Decode discrete token IDs into human-readable text string."""
        if isinstance(token_ids, mx.array):
            mx.eval(token_ids)
            if token_ids.ndim == 2:
                ids = token_ids[0].tolist()
            else:
                ids = token_ids.tolist()
        elif isinstance(token_ids, (list, tuple)):
            if len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
                ids = list(token_ids[0])
            else:
                ids = list(token_ids)
        else:
            ids = [int(token_ids)]

        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)

        # Clean fallback ascii character decoding
        return "".join(chr(i % 128) for i in ids if 32 <= (i % 128) <= 126)


# Backwards compatibility alias
GemmaDeliberationPipeline = PRLRPipeline

__all__ = [
    "HybridDeliberationResult",
    "DeliberationPipelineOutput",
    "PRLRPipeline",
    "GemmaDeliberationPipeline",
]
