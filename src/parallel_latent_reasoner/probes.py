"""Diagnostic Probes and Trajectory Telemetry for Latent Deliberation.

Provides mathematical probes to quantify representation capacity and dynamical properties:
- SVD spectral Shannon entropy effective rank erank(S) in [1.0, M]
- Step-to-step slot-wise cosine similarity rho(t) and velocity v(t) = 1.0 - rho(t)
- Full symmetric Gram history matrix G in R^((T+1) x (T+1))
- Phase classification: fixed point, limit cycle (period-2 orbit), monotonic refinement, divergent
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx
import numpy as np


def compute_effective_rank(
    states: mx.array | np.ndarray,
    center: bool = False,
    eps: float = 1e-12,
) -> float:
    """Compute SVD spectral Shannon entropy effective rank of memory states.

    The effective rank (Roy & Vetterli, 2007) is defined as:
        erank(S) = exp( - sum_{i=1}^r p_i ln(p_i) ) in [1, min(M, D)]
    where p_i = sigma_i / sum_{j=1}^r sigma_j are the normalized singular values
    of the memory state matrix S in R^(M x D).

    To ensure strict numerical stability and avoid GPU LAPACK faults on Apple Silicon,
    singular value decomposition is safely evaluated via CPU stream (NumPy).

    Args:
        states: Memory state tensor of shape [M, D] or batched [B, M, D].
        center: If True, subtract the slot-wise mean before SVD.
        eps: Small numerical epsilon to prevent log(0).

    Returns:
        float: Effective rank in [1.0, min(M, D)].
    """
    if isinstance(states, mx.array):
        mx.eval(states)
        arr = np.array(states.astype(mx.float32))
    else:
        arr = np.asarray(states, dtype=np.float32)

    if arr.ndim == 3:
        b_size = arr.shape[0]
        if b_size == 0:
            return 1.0
        eranks = [
            compute_effective_rank(arr[b], center=center, eps=eps)
            for b in range(b_size)
        ]
        return float(np.mean(eranks))

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D [M, D] or 3D [B, M, D] array, got shape {arr.shape}")

    M, D = arr.shape
    if M <= 1:
        return 1.0

    matrix = arr
    if center:
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)

    norm_val = float(np.linalg.norm(matrix))
    if norm_val < 1e-9:
        return 1.0

    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return 1.0

    s_pos = s[s > 1e-9]
    if s_pos.size == 0:
        return 1.0

    s_sum = float(np.sum(s_pos))
    if s_sum <= 0.0:
        return 1.0

    p = s_pos / s_sum
    entropy = -float(np.sum(p * np.log(p + eps)))
    erank = float(np.exp(entropy))

    max_rank = float(min(M, D))
    return max(1.0, min(erank, max_rank))


def compute_slot_cosine_similarity(
    state_a: mx.array,
    state_b: mx.array,
    eps: float = 1e-8,
) -> float:
    """Compute average slot-wise cosine similarity between two memory state tensors."""
    if state_a.ndim == 3:
        # [B, M, D]
        norm_a = mx.sqrt(mx.sum(mx.square(state_a), axis=-1, keepdims=True) + eps)
        norm_b = mx.sqrt(mx.sum(mx.square(state_b), axis=-1, keepdims=True) + eps)
        cos = mx.sum((state_a / norm_a) * (state_b / norm_b), axis=-1)  # [B, M]
        mean_cos = mx.mean(cos)
    elif state_a.ndim == 2:
        # [M, D]
        norm_a = mx.sqrt(mx.sum(mx.square(state_a), axis=-1, keepdims=True) + eps)
        norm_b = mx.sqrt(mx.sum(mx.square(state_b), axis=-1, keepdims=True) + eps)
        cos = mx.sum((state_a / norm_a) * (state_b / norm_b), axis=-1)  # [M]
        mean_cos = mx.mean(cos)
    else:
        norm_a = mx.sqrt(mx.sum(mx.square(state_a)) + eps)
        norm_b = mx.sqrt(mx.sum(mx.square(state_b)) + eps)
        mean_cos = mx.sum((state_a / norm_a) * (state_b / norm_b))

    mx.eval(mean_cos)
    return float(mean_cos.item())


def compute_slot_velocity(
    state_a: mx.array,
    state_b: mx.array,
    eps: float = 1e-8,
) -> float:
    """Compute normalized step velocity v(t) = 1.0 - cosine_similarity."""
    sim = compute_slot_cosine_similarity(state_a, state_b, eps=eps)
    return max(0.0, 1.0 - sim)


@dataclass
class TrajectoryAnalysis:
    """Container for deliberation trajectory diagnostics and Gram matrix history."""

    step_similarities: list[float]
    step_velocities: list[float]
    gram_matrix: mx.array
    is_converged: bool
    is_limit_cycle: bool
    is_divergent: bool
    optimal_exit_step: int
    effective_ranks: list[float] | None = None
    phase: str = "monotonic_refinement"


def analyze_deliberation_trajectory(
    trajectory: Sequence[mx.array],
    tol_conv: float = 0.995,
    tol_vel: float = 5e-3,
    compute_erank: bool = False,
) -> TrajectoryAnalysis:
    """Analyze a sequence of deliberation memory states S^(0), S^(1), ..., S^(T).

    Computes:
    - Step-to-step cosine similarity: rho(t) = cos(S^(t), S^(t-1))
    - Step trajectory velocity: v(t) = 1.0 - rho(t)
    - Full Gram history matrix: G_(i, j) = cos(S^(i), S^(j))
    - Phase detection: fixed-point convergence, period-2 limit cycles, divergence.

    Args:
        trajectory: List of memory state tensors of length T+1.
        tol_conv: Cosine similarity threshold for convergence (e.g. >= 0.995).
        tol_vel: Velocity threshold for convergence (e.g. <= 0.005).
        compute_erank: Whether to calculate effective rank at each step.

    Returns:
        TrajectoryAnalysis dataclass.
    """
    if len(trajectory) < 2:
        raise ValueError("Trajectory must contain at least 2 states (initial and step 1).")

    num_states = len(trajectory)
    T = num_states - 1

    sims: list[float] = []
    vels: list[float] = []

    # 1. Step-to-step metrics
    for t in range(1, num_states):
        prev = trajectory[t - 1]
        curr = trajectory[t]
        sim = compute_slot_cosine_similarity(prev, curr)
        vel = max(0.0, 1.0 - sim)
        sims.append(sim)
        vels.append(vel)

    # 2. Symmetric Gram history matrix G in R^((T+1) x (T+1))
    gram = np.zeros((num_states, num_states), dtype=np.float32)
    for i in range(num_states):
        gram[i, i] = 1.0
        for j in range(i + 1, num_states):
            c = compute_slot_cosine_similarity(trajectory[i], trajectory[j])
            gram[i, j] = c
            gram[j, i] = c

    gram_mx = mx.array(gram)

    # 3. Effective rank trajectory
    eranks: list[float] | None = None
    if compute_erank:
        eranks = [compute_effective_rank(s) for s in trajectory]

    # 4. Phase classification
    is_converged = bool(sims[-1] >= tol_conv or vels[-1] <= tol_vel)

    # Period-2 limit cycle: G(t, t-2) > 0.95 and G(t, t-1) < 0.92
    is_limit_cycle = False
    if T >= 2:
        g_t_t2 = float(gram[T, T - 2])
        g_t_t1 = float(gram[T, T - 1])
        if g_t_t2 > 0.95 and g_t_t1 < 0.92:
            is_limit_cycle = True
        elif g_t_t2 > g_t_t1 + 0.05:
            is_limit_cycle = True

    # Divergence detection
    is_divergent = any(math.isnan(s) or s < -0.5 for s in sims) or (
        len(vels) >= 3 and vels[-1] > vels[0] * 3.0 and vels[-1] > 0.1
    )

    # Find earliest optimal exit step
    optimal_exit = T
    for idx, (s, v) in enumerate(zip(sims, vels)):
        if s >= tol_conv or v <= tol_vel:
            optimal_exit = idx + 1
            break

    phase = "monotonic_refinement"
    if is_converged:
        phase = "fixed_point"
    elif is_limit_cycle:
        phase = "limit_cycle"
    elif is_divergent:
        phase = "divergent"

    return TrajectoryAnalysis(
        step_similarities=sims,
        step_velocities=vels,
        gram_matrix=gram_mx,
        is_converged=is_converged,
        is_limit_cycle=is_limit_cycle,
        is_divergent=is_divergent,
        optimal_exit_step=optimal_exit,
        effective_ranks=eranks,
        phase=phase,
    )


def detect_limit_cycle(
    trajectory: Sequence[mx.array],
    erank_history: list[float] | None = None,
    tol_limit_cycle: float = 0.05,
) -> dict[str, Any]:
    """Analyze trajectory for periodic orbits, fixed-point convergence, and rank collapse.

    Args:
        trajectory: Sequence of memory states across deliberation unrolls.
        erank_history: Optional precomputed list of effective ranks per step.
        tol_limit_cycle: Threshold for limit cycle detection.

    Returns:
        Dictionary containing phase classification, limit cycle flags, and rank status.
    """
    analysis = analyze_deliberation_trajectory(
        trajectory,
        compute_erank=(erank_history is None),
    )
    eranks = erank_history if erank_history is not None else analysis.effective_ranks

    rank_collapsed = False
    if eranks is not None and len(eranks) > 0:
        final_rank = eranks[-1]
        initial_rank = eranks[0]
        if final_rank < 1.1 and initial_rank > 2.0:
            rank_collapsed = True
        elif final_rank < 0.25 * initial_rank and initial_rank > 4.0:
            rank_collapsed = True

    return {
        "phase": "rank_collapse" if rank_collapsed else analysis.phase,
        "is_converged": analysis.is_converged,
        "is_limit_cycle": analysis.is_limit_cycle,
        "is_divergent": analysis.is_divergent,
        "rank_collapsed": rank_collapsed,
        "optimal_exit_step": analysis.optimal_exit_step,
        "final_step_similarity": analysis.step_similarities[-1] if analysis.step_similarities else 1.0,
        "final_step_velocity": analysis.step_velocities[-1] if analysis.step_velocities else 0.0,
        "erank_history": eranks,
    }


__all__ = [
    "compute_effective_rank",
    "compute_slot_cosine_similarity",
    "compute_slot_velocity",
    "analyze_deliberation_trajectory",
    "detect_limit_cycle",
    "TrajectoryAnalysis",
]
