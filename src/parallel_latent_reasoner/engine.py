"""Parallel Latent Deliberation Engine for Recurrent Transformer Core.

Executes non-causal parallel Jacobi deliberation sweeps across M working memory slots,
providing step transitions, unroll loops, and JIT-compiled Metal GPU graphs.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXRecurrentGemmaBlock


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

    def __init__(self, config: GemmaLatentConfig):
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
        """Execute one Jacobi deliberation step across all working memory slots in parallel.

        Args:
            memory_slots: Working memory slots of shape [B, M, D].
            step_idx: Deliberation step index t (1-indexed).
            prompt_kv: Static prompt key/value cache tuple.
            prompt_len: Context prefix length for RoPE offset.
            mask: Attention mask (defaults to None for non-causal bidirectional).

        Returns:
            Updated working memory slots of shape [B, M, D].
        """
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
        """Unroll deliberation for T steps with constant sequence length and zero KV growth.

        Args:
            initial_memory: Initial slot embeddings S^(0) of shape [B, M, D].
            prompt_kv: Static prompt key/value cache tuple.
            steps: Number of unroll steps T (defaults to config.deliberation_steps).
            prompt_len: Context prefix length for RoPE offset.
            return_trajectory: Whether to retain intermediate states S^(t).

        Returns:
            DeliberationResult with final memory state, trajectory list, and step count.
        """
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
        """Return an @mx.compile JIT-compiled unroll function for Metal GPU acceleration.

        Args:
            steps: Fixed unroll depth T.
            prompt_kv: Static prompt key/value cache tuple.
            prompt_len: Context prefix length for RoPE offset.

        Returns:
            Compiled function mapping S^(0) -> S^(T).
        """
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
