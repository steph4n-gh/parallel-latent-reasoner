"""Unit tests for Diagnostic Probes and the 3-Signal Dynamic Consensus E-Gate."""

import mlx.core as mx
import numpy as np
import pytest

from parallel_latent_reasoner.egate import DynamicConsensusEGate, DynamicDeliberationGate
from parallel_latent_reasoner.probes import (
    analyze_deliberation_trajectory,
    compute_effective_rank,
    compute_slot_cosine_similarity,
    compute_slot_velocity,
    detect_limit_cycle,
)


def test_effective_rank_orthogonal_basis():
    """Verify orthogonal basis of M slots achieves maximal erank = M."""
    M, D = 16, 64
    # Create orthonormal basis via QR decomposition
    rnd = np.random.randn(D, M).astype(np.float32)
    q, _ = np.linalg.qr(rnd)
    ortho_matrix = q.T  # [M, D]
    erank = compute_effective_rank(ortho_matrix)
    assert abs(erank - M) < 0.05, f"Orthonormal matrix of size {M} must have erank ~ {M}, got {erank}"


def test_effective_rank_collinear_collapse():
    """Verify rank-1 collinear state collapses to erank ~ 1.0."""
    M, D = 16, 64
    base = np.random.randn(1, D).astype(np.float32)
    collinear = np.repeat(base, M, axis=0)  # [M, D] all rows identical
    erank = compute_effective_rank(collinear)
    assert abs(erank - 1.0) < 1e-3, f"Collinear matrix must have erank ~ 1.0, got {erank}"


def test_effective_rank_edge_cases():
    """Verify all-zero matrices, single slot M=1, and batched inputs are handled safely."""
    # Zero matrix
    zeros = np.zeros((16, 64), dtype=np.float32)
    assert compute_effective_rank(zeros) == 1.0

    # Single slot
    single = np.random.randn(1, 64).astype(np.float32)
    assert compute_effective_rank(single) == 1.0

    # Batched inputs [B, M, D]
    batched = mx.random.normal((4, 16, 64))
    erank_b = compute_effective_rank(batched)
    assert 1.0 <= erank_b <= 16.0


def test_cosine_similarity_and_velocity():
    """Verify slot cosine similarity and velocity metrics."""
    a = mx.random.normal((1, 16, 64))
    # Identical states -> similarity 1.0, velocity 0.0
    sim_id = compute_slot_cosine_similarity(a, a)
    vel_id = compute_slot_velocity(a, a)
    assert abs(sim_id - 1.0) < 1e-4
    assert vel_id < 1e-4

    # Opposite states -> similarity -1.0, velocity 2.0
    neg_a = -a
    sim_neg = compute_slot_cosine_similarity(a, neg_a)
    vel_neg = compute_slot_velocity(a, neg_a)
    assert abs(sim_neg - (-1.0)) < 1e-4
    assert abs(vel_neg - 2.0) < 1e-4


def test_gram_matrix_symmetry_and_unit_diagonal():
    """Verify Gram history matrix is symmetric with unit diagonal."""
    states = [mx.random.normal((1, 16, 64)) for _ in range(5)]
    analysis = analyze_deliberation_trajectory(states, compute_erank=True)
    gram = np.array(analysis.gram_matrix)

    assert gram.shape == (5, 5)
    # Unit diagonal
    diag = np.diag(gram)
    assert np.allclose(diag, 1.0, atol=1e-5)
    # Symmetry
    assert np.allclose(gram, gram.T, atol=1e-5)


def test_period2_limit_cycle_detection():
    """Verify limit cycle detector correctly identifies period-2 orbits."""
    state_a = mx.random.normal((1, 16, 64))
    state_b = mx.random.normal((1, 16, 64))

    # Construct oscillating trajectory: A -> B -> A -> B -> A
    traj = [state_a, state_b, state_a, state_b, state_a]
    diag = detect_limit_cycle(traj)
    assert diag["is_limit_cycle"] is True
    assert diag["phase"] == "limit_cycle"


def test_3_signal_dynamic_consensus_egate_logic():
    """Verify 3-Signal Dynamic Consensus E-Gate requires all 3 signals to halt."""
    gate = DynamicDeliberationGate(
        tol_rel_vel=0.10,
        tol_erank_delta=0.005,
        min_steps=2,
        max_steps=12,
        patience=1,
    )

    base = mx.random.normal((1, 16, 64))
    # Step 0: Prelude
    gate.update(base, step=0, coda_token=42)

    # Step 1: Initial jump
    s1 = base + 0.5 * mx.random.normal((1, 16, 64))
    tel1 = gate.update(s1, step=1, coda_token=42)
    assert not tel1.halt, "Gate must not halt at step 1 (min_steps=2)."

    # Case A: Signal 2 false (Coda token flips: 99 != 42) -> Must NOT halt even if velocity is low
    s2_sub = s1 + 0.001 * mx.random.normal((1, 16, 64))
    tel2_flip = gate.update(s2_sub, step=2, coda_token=99)
    assert not tel2_flip.signal_coda
    assert not tel2_flip.halt, "Gate must not halt when Coda prediction is flipping."

    # Case B: All 3 signals satisfied at Step 3 (low rel vel, same token 99, stable erank)
    s3_converged = s2_sub + 0.00001 * mx.random.normal((1, 16, 64))
    tel3 = gate.update(s3_converged, step=3, coda_token=99)
    assert tel3.signal_velocity
    assert tel3.signal_coda
    assert tel3.signal_erank
    assert tel3.halt
    assert tel3.exit_reason == "3_signal_consensus"


def test_egate_timeout_at_max_steps():
    """Verify E-Gate enforces hard exit when max_steps is reached."""
    gate = DynamicDeliberationGate(max_steps=5)
    # Feed non-converging states with alternating coda tokens
    curr = mx.random.normal((1, 16, 64))
    gate.update(curr, step=0, coda_token=1)

    for t in range(1, 5):
        curr = curr + mx.random.normal((1, 16, 64))
        tel = gate.update(curr, step=t, coda_token=t)
        assert not tel.halt

    # Step 5 (max_steps)
    curr = curr + mx.random.normal((1, 16, 64))
    tel5 = gate.update(curr, step=5, coda_token=5)
    assert tel5.halt
    assert tel5.exit_reason == "max_steps_timeout"
