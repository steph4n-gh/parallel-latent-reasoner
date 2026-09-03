"""3-Signal Dynamic Consensus E-Gate for Latent Deliberation.

Monitors three independent, orthogonal indicators of representation maturity:
1. Relative velocity decay: v(t) / v(1) < 0.10 (dissipation of kinetic energy)
2. Coda discrete prediction consensus: y_hat^(t) == y_hat^(t-1) (symbolic invariance)
3. SVD effective rank plateau: |erank(t) - erank(t-1)| < 0.005 (subspace saturation)

Halts when all 3 signals agree for t >= T_min (default T_min=2, T_max=12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from prlr.kernel.telemetry import (
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
)


@dataclass
class GateTelemetry:
    """Full telemetry recorded at each deliberation step by the E-Gate."""

    step: int
    velocity: float
    rel_velocity: float
    coda_token: int | None
    coda_token_str: str | None
    erank: float
    delta_erank: float
    signal_velocity: bool
    signal_coda: bool
    signal_erank: bool
    halt: bool
    exit_reason: str


@dataclass
class GateDecision:
    """Decision output of the 3-Signal Dynamic Consensus E-Gate."""

    halt: bool
    step: int
    rel_velocity: float
    coda_pred: int | None
    erank: float
    delta_erank: float
    signal_velocity: bool
    signal_coda: bool
    signal_erank: bool
    exit_reason: str


class DynamicDeliberationGate:
    """3-Signal Dynamic Consensus E-Gate for autonomous deliberation halting.

    Halts when relative velocity drops below tolerance (tol_rel_vel < 0.10),
    top-1 Coda discrete prediction stabilizes (y^(t) == y^(t-1)), and SVD effective
    rank growth plateaus (|delta erank| < 0.005) for t >= min_steps.
    """

    def __init__(
        self,
        tol_rel_vel: float = 0.10,
        tol_erank_delta: float = 0.005,
        min_steps: int = 2,
        max_steps: int = 12,
        patience: int = 1,
    ):
        self.tol_rel_vel = tol_rel_vel
        self.tol_erank_delta = tol_erank_delta
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.patience = patience

        self._consecutive_consensus = 0
        self._history_states: list[mx.array] = []
        self._history_coda_tokens: list[int | None] = []
        self._history_coda_strs: list[str | None] = []
        self._history_eranks: list[float] = []
        self._history_velocities: list[float] = []
        self._history_rel_velocities: list[float] = []
        self._v1: float | None = None
        self._telemetry: list[GateTelemetry] = []

    def reset(self) -> None:
        """Reset internal history and state counters."""
        self._consecutive_consensus = 0
        self._history_states.clear()
        self._history_coda_tokens.clear()
        self._history_coda_strs.clear()
        self._history_eranks.clear()
        self._history_velocities.clear()
        self._history_rel_velocities.clear()
        self._v1 = None
        self._telemetry.clear()

    def update(
        self,
        curr_state: mx.array,
        step: int,
        coda_token: int | None = None,
        coda_token_str: str | None = None,
    ) -> GateTelemetry:
        """Record step state, evaluate 3 signals, and return telemetry with halting decision."""
        curr_erank = compute_effective_rank(curr_state)
        self._history_eranks.append(curr_erank)
        self._history_coda_tokens.append(coda_token)
        self._history_coda_strs.append(coda_token_str)

        if len(self._history_states) == 0:
            self._history_states.append(curr_state)
            telemetry = GateTelemetry(
                step=step,
                velocity=0.0,
                rel_velocity=1.0,
                coda_token=coda_token,
                coda_token_str=coda_token_str,
                erank=curr_erank,
                delta_erank=0.0,
                signal_velocity=False,
                signal_coda=False,
                signal_erank=False,
                halt=False,
                exit_reason="initialization",
            )
            self._telemetry.append(telemetry)
            return telemetry

        prev_state = self._history_states[-1]
        self._history_states.append(curr_state)

        vel = compute_slot_velocity(prev_state, curr_state)
        self._history_velocities.append(vel)

        if self._v1 is None or self._v1 <= 1e-9:
            self._v1 = vel if vel > 1e-9 else 1e-6

        rel_vel = vel / (self._v1 + 1e-9)
        self._history_rel_velocities.append(rel_vel)

        prev_erank = self._history_eranks[-2] if len(self._history_eranks) >= 2 else curr_erank
        delta_erank = abs(curr_erank - prev_erank)

        prev_coda = (
            self._history_coda_tokens[-2]
            if len(self._history_coda_tokens) >= 2
            else None
        )
        coda_consensus = (
            coda_token is not None
            and prev_coda is not None
            and coda_token == prev_coda
        )

        sig_velocity = bool(rel_vel < self.tol_rel_vel)
        sig_coda = bool(coda_consensus)
        sig_erank = bool(delta_erank < self.tol_erank_delta)

        all_signals = sig_velocity and sig_coda and sig_erank

        if all_signals and step >= self.min_steps:
            self._consecutive_consensus += 1
        else:
            self._consecutive_consensus = 0

        halt = False
        exit_reason = "active"

        if step >= self.max_steps:
            halt = True
            exit_reason = "max_steps_timeout"
        elif step >= self.min_steps and self._consecutive_consensus >= self.patience:
            halt = True
            exit_reason = "3_signal_consensus"

        telemetry = GateTelemetry(
            step=step,
            velocity=vel,
            rel_velocity=rel_vel,
            coda_token=coda_token,
            coda_token_str=coda_token_str,
            erank=curr_erank,
            delta_erank=delta_erank,
            signal_velocity=sig_velocity,
            signal_coda=sig_coda,
            signal_erank=sig_erank,
            halt=halt,
            exit_reason=exit_reason,
        )
        self._telemetry.append(telemetry)
        return telemetry

    def step(
        self,
        t: int,
        current_slots: mx.array,
        prev_slots: mx.array | None = None,
        current_pred: int | None = None,
        prev_pred: int | None = None,
    ) -> GateDecision:
        """Functional evaluation interface for step t."""
        if prev_slots is not None and len(self._history_states) == 0:
            self.update(prev_slots, step=max(0, t - 1), coda_token=prev_pred)

        tel = self.update(current_slots, step=t, coda_token=current_pred)
        return GateDecision(
            halt=tel.halt,
            step=tel.step,
            rel_velocity=tel.rel_velocity,
            coda_pred=tel.coda_token,
            erank=tel.erank,
            delta_erank=tel.delta_erank,
            signal_velocity=tel.signal_velocity,
            signal_coda=tel.signal_coda,
            signal_erank=tel.signal_erank,
            exit_reason=tel.exit_reason,
        )

    @property
    def telemetry_history(self) -> list[GateTelemetry]:
        return self._telemetry

    @property
    def step_velocities(self) -> list[float]:
        return self._history_velocities

    @property
    def rel_velocities(self) -> list[float]:
        return self._history_rel_velocities

    @property
    def erank_history(self) -> list[float]:
        return self._history_eranks

    @property
    def coda_history(self) -> list[int | None]:
        return self._history_coda_tokens


DynamicConsensusEGate = DynamicDeliberationGate

__all__ = [
    "GateTelemetry",
    "GateDecision",
    "DynamicDeliberationGate",
    "DynamicConsensusEGate",
]
