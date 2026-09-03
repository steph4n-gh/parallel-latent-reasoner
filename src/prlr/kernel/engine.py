"""Parallel Latent Deliberation Engine for Recurrent Transformer Core.

Executes non-causal parallel Jacobi deliberation sweeps across M working memory slots,
providing step transitions, unroll loops, and JIT-compiled Metal GPU graphs.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from prlr.kernel.recurrent_core import MLXRecurrentBlock, MLXRecurrentGemmaBlock


class DeliberationResult(NamedTuple):
    """Result container for parallel latent deliberation sweeps."""

    final_states: mx.array
    trajectory_states: list[mx.array] | None
    steps_executed: int


class MLXParallelLatentEngine(nn.Module):
    """Parallel Latent Deliberation Engine executing Jacobi sweeps over memory slots.

    Manages recurrent unrolls across T deliberation steps, ensuring constant sequence
    length (M working slots), zero KV-cache growth during deliberation, and JIT compilation support.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.layers = [
            MLXRecurrentGemmaBlock(config)
            for _ in range(config.num_layers)
        ]

    def step(
        self,
        memory_slots: mx.array,
        step_idx: int | float | mx.array,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        prompt_len: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Execute one Jacobi deliberation step across all working memory slots in parallel."""
        h = memory_slots
        for layer in self.layers:
            h = layer(
                h,
                step=step_idx,
                prompt_kv=prompt_kv,
                prompt_len=prompt_len,
                mask=mask,
            )
        return h

    def deliberate(
        self,
        initial_memory: mx.array,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        steps: int | None = None,
        prompt_len: int = 0,
        return_trajectory: bool = False,
    ) -> DeliberationResult:
        """Unroll deliberation for T steps with constant sequence length and zero KV growth."""
        T = steps if steps is not None else self.config.deliberation_steps
        curr = initial_memory
        trajectory: list[mx.array] | None = [curr] if return_trajectory else None

        for t in range(1, T + 1):
            curr = self.step(
                curr,
                step_idx=t,
                prompt_kv=prompt_kv,
                prompt_len=prompt_len,
            )
            mx.eval(curr)
            if return_trajectory and trajectory is not None:
                trajectory.append(curr)

        return DeliberationResult(
            final_states=curr,
            trajectory_states=trajectory,
            steps_executed=T,
        )

    def compile_unroll(
        self,
        steps: int | None = None,
        prompt_kv: tuple[mx.array, mx.array] | mx.array | None = None,
        prompt_len: int = 0,
    ) -> Callable[[mx.array], mx.array]:
        """Return an @mx.compile JIT-compiled unroll function for Metal GPU acceleration."""
        T = steps if steps is not None else self.config.deliberation_steps

        @mx.compile
        def _compiled_loop(memory_slots: mx.array) -> mx.array:
            h = memory_slots
            for t in range(1, T + 1):
                for layer in self.layers:
                    h = layer(
                        h,
                        step=t,
                        prompt_kv=prompt_kv,
                        prompt_len=prompt_len,
                    )
            return h

        return _compiled_loop


__all__ = [
    "DeliberationResult",
    "MLXParallelLatentEngine",
]
