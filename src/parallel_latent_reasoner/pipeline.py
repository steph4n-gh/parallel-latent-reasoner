"""End-to-End Deliberation Pipeline for Standalone Parallel Latent Reasoner.

Orchestrates the entire latent deliberation pipeline:
1. Tokenization and prompt prefill
2. Prelude working memory slot initialization (M=16 slots)
3. Recurrent Jacobi unroll sweeps with 3-Signal Dynamic Consensus E-Gate
4. Discrete Coda LM Head solution decoding without intermediate CoT tokens
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.egate import DynamicDeliberationGate, GateTelemetry
from parallel_latent_reasoner.engine import DeliberationResult
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.probes import (
    TrajectoryAnalysis,
    analyze_deliberation_trajectory,
    compute_effective_rank,
    detect_limit_cycle,
)


@dataclass
class DeliberationPipelineOutput:
    """Output container for end-to-end deliberation and solution generation."""

    token_ids: mx.array
    deliberation_steps: int
    final_states: mx.array
    trajectory_states: list[mx.array] | None = None
    effective_ranks: list[float] | None = None
    trajectory_analysis: TrajectoryAnalysis | None = None
    diagnostics: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    gate_telemetry: list[GateTelemetry] | None = None


class GemmaDeliberationPipeline:
    """End-to-end continuous latent deliberation and discrete generation pipeline.

    Stages:
    1. Prelude Projection: Embeds prompt tokens scaled by sqrt(D) and initializes M working memory slots S^(0).
    2. Parallel Latent Deliberation: Executes T Jacobi sweeps with AdaRMSNorm step conditioning and ReZero scaling.
    3. 3-Signal Dynamic Consensus E-Gate: Evaluates velocity decay, Coda prediction consensus, and SVD erank plateau.
    4. Discrete Coda Decoding: Autoregressively generates discrete solution tokens conditioned on refined latent state S^(T).
    """

    def __init__(
        self,
        model: MLXCompactGemmaModel | None = None,
        config: GemmaLatentConfig | None = None,
        tokenizer: Any | None = None,
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

    @classmethod
    def from_preset(
        cls,
        preset: str = "compact_test",
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> GemmaDeliberationPipeline:
        """Create a deliberation pipeline from a standard config preset."""
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
        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Expected 'compact_test', 'gemma_2b', 'gemma_9b', 'gemma_12b', or 'gemma_e4b'."
            )

        model = MLXCompactGemmaModel(config)
        return cls(model=model, config=config, tokenizer=tokenizer)

    def deliberate(
        self,
        prompt_tokens: mx.array,
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
    ) -> tuple[DeliberationResult, list[GateTelemetry] | None]:
        """Execute continuous latent deliberation over working memory slots with optional E-Gate.

        Args:
            prompt_tokens: Prompt token IDs of shape [B, P].
            steps: Maximum deliberation sweeps T (defaults to config.deliberation_steps).
            enable_dynamic_gate: If True, uses the 3-Signal Dynamic Consensus E-Gate.
            gate: Optional pre-existing DynamicDeliberationGate instance.
            tol_rel_vel: Velocity decay threshold (default 0.10).
            tol_erank_delta: SVD effective rank plateau threshold (default 0.005).
            min_steps: Minimum deliberation steps before early exit is allowed (default 2).
            max_steps: Hard upper bound on unroll steps (default config.max_steps or steps).
            patience: Number of consecutive consensus steps required to halt (default 1).
            return_trajectory: Whether to record all intermediate memory states S^(t).
            compute_probes: Whether to compute diagnostic probes on trajectory.

        Returns:
            Tuple of (DeliberationResult, list of GateTelemetry or None).
        """
        max_T = (
            steps
            if steps is not None
            else (max_steps if max_steps is not None else self.config.deliberation_steps)
        )

        # 1. Prelude projection: initialize M slots S^(0)
        slots, prompt_hiddens = self.model.prelude(prompt_tokens)
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
            curr = self.model.engine.step(
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

    def generate(
        self,
        prompt: str | mx.array | Sequence[int],
        max_new_tokens: int = 16,
        deliberation_steps: int | None = None,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        return_diagnostics: bool = False,
        **gate_kwargs: Any,
    ) -> DeliberationPipelineOutput:
        """Run complete end-to-end deliberation and discrete token decoding.

        Args:
            prompt: Text prompt string, integer token ID list, or mx.array [B, P].
            max_new_tokens: Number of discrete solution tokens to generate.
            deliberation_steps: Maximum unroll sweeps T.
            temperature: Sampling temperature (0.0 for greedy argmax).
            enable_dynamic_gate: Whether to allow early termination on 3-signal consensus.
            return_diagnostics: Whether to calculate full probes, Gram matrix, and erank.
            gate_kwargs: Additional parameters passed to deliberate() / E-Gate.

        Returns:
            DeliberationPipelineOutput containing generated tokens, states, metrics, and diagnostics.
        """
        t0 = time.perf_counter()

        # 1. Parse prompt inputs
        if isinstance(prompt, str):
            if self.tokenizer is not None:
                token_ids = mx.array([self.tokenizer.encode(prompt)], dtype=mx.int32)
            else:
                token_ids = mx.array([[ord(c) % self.config.vocab_size for c in prompt]], dtype=mx.int32)
        elif isinstance(prompt, (list, tuple)):
            token_ids = mx.array([list(prompt)], dtype=mx.int32)
        elif isinstance(prompt, mx.array):
            if prompt.ndim == 1:
                token_ids = prompt[None, :]
            else:
                token_ids = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        mx.eval(token_ids)
        t_prefill_end = time.perf_counter()

        # 2. Deliberate in continuous latent space
        delib_res, gate_telemetry = self.deliberate(
            token_ids,
            steps=deliberation_steps,
            enable_dynamic_gate=enable_dynamic_gate,
            return_trajectory=return_diagnostics,
            compute_probes=return_diagnostics,
            **gate_kwargs,
        )
        mx.eval(delib_res.final_states)
        t_delib_end = time.perf_counter()

        # 3. Decode into discrete solution tokens
        readout = self.model.coda.pool_readout(delib_res.final_states)
        B = token_ids.shape[0]
        generated_tokens: list[mx.array] = []
        curr_hidden = readout

        for _ in range(max_new_tokens):
            logits = self.model.coda.project_logits(curr_hidden)
            if temperature <= 1e-5:
                next_tok = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                next_tok = mx.random.categorical(logits / temperature)[:, None]
            generated_tokens.append(next_tok)

            tok_embed = self.model.prelude.embed_prompt(next_tok)[:, 0, :]
            curr_hidden = self.model.coda.final_norm(curr_hidden + 0.1 * tok_embed)

        solution_ids = mx.concatenate(generated_tokens, axis=-1)
        mx.eval(solution_ids)
        t_decode_end = time.perf_counter()

        # 4. Compute diagnostics if requested
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

        metrics = {
            "prefill_latency_ms": (t_prefill_end - t0) * 1000.0,
            "deliberation_latency_ms": (t_delib_end - t_prefill_end) * 1000.0,
            "coda_decode_latency_ms": (t_decode_end - t_delib_end) * 1000.0,
            "total_latency_ms": (t_decode_end - t0) * 1000.0,
            "steps_executed": float(delib_res.steps_executed),
            "tokens_generated": float(max_new_tokens),
        }

        return DeliberationPipelineOutput(
            token_ids=solution_ids,
            deliberation_steps=delib_res.steps_executed,
            final_states=delib_res.final_states,
            trajectory_states=delib_res.trajectory_states,
            effective_ranks=erank_history,
            trajectory_analysis=trajectory_analysis,
            diagnostics=diagnostics,
            metrics=metrics,
            gate_telemetry=gate_telemetry,
        )

    def decode_solution(self, token_ids: mx.array) -> str:
        """Decode discrete token IDs into text string."""
        mx.eval(token_ids)
        if token_ids.ndim == 2:
            ids = token_ids[0].tolist()
        else:
            ids = token_ids.tolist()

        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        # Fallback ascii decoding
        return "".join(chr(i % 128) for i in ids if 32 <= (i % 128) <= 126)


__all__ = [
    "DeliberationPipelineOutput",
    "GemmaDeliberationPipeline",
]
