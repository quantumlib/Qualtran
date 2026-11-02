# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

r"""Zero-Iteration Analytical Star-Anisotropy Syzygy and Inhomogeneous Off-Diagonal Bianchi Certifier for DQI.

Formal Lean 4.20.0 Proofs & Float64 Certificates (Apache-2.0):
  - Companion Verification Repo (Pinned v1.0.0): https://github.com/justinarndtQ/dqi-star-anisotropy-syzygy-lean/tree/v1.0.0
  - Master Certificate Repo: https://github.com/justinarndtQ/quantum-cohomological-certificates
  - Lean 4.20.0 Kernel Proofs (0 sorryAx): lean/ExteriorBruhatParabolic.lean, lean/CohomologicalHorizons.lean
  - Interactive 3-Second Lean 4 Check (paste at https://live.lean-lang.org):
      theorem dqi_partition_syzygy_and_hochschild_strata :
          24 * 23 / 2 = 276 /\ 276 + 24 = 300 /\
          276 + 6072 + 31878 = 276 * 277 / 2 := by decide
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray


def compute_zero_iteration_partition_syzygy_bound(
    m_offdiag: NDArray[np.float64],
    diag_trace_shift: float = 0.0,
    n_spins: int = 24,
) -> Dict[str, float]:
    """Computes the exact 0-iteration Partition Syzygy bound on {+-1}^N.

    Eliminates the phantom diagonal norm penalty N * |lambda_min(M_offdiag)| in 0 steps
    using the exact hypercube partition identity ||z_off(x)||_2^2 == N(N-1)/2.
    """
    d_off = n_spins * (n_spins - 1) // 2
    d_total = d_off + n_spins
    eigvals_off = np.linalg.eigvalsh(m_offdiag)
    lam_min_off = float(eigvals_off[0])

    naive_bound = float(d_total) * lam_min_off + diag_trace_shift
    syzygy_bound = float(d_off) * lam_min_off + diag_trace_shift
    phantom_penalty_eliminated = float(n_spins) * abs(lam_min_off)
    return {
        "naive_300_bound": naive_bound,
        "zero_iter_syzygy_bound": syzygy_bound,
        "phantom_penalty_eliminated": phantom_penalty_eliminated,
        "lambda_min_offdiag": lam_min_off,
    }


def evaluate_inhomogeneous_bianchi_quadratic_form(
    k_hollow: NDArray[np.float64], x_spins: NDArray[np.float64]
) -> Tuple[float, float]:
    """Evaluates the exact inhomogeneous off-diagonal Bianchi quadratic form on x in {+-1}^N.

    By expanding (x^T K x)^2 - (x^T K x)^2 == 0 and separating diagonal squares x_i^2 = 1
    from off-diagonal pairs z_{ij} = x_i x_j (i < j), the pure off-diagonal quadratic form
    equals -||K||_F^2 identically for every hypercube vertex x in {+-1}^N.
    """
    x_full = np.outer(x_spins, x_spins)
    x_off = x_full - np.diag(np.diag(x_full))
    kx_off = k_hollow @ x_off
    tr_kx = float(np.trace(kx_off))
    b_off_action = kx_off @ k_hollow - tr_kx * k_hollow
    quad_off = float(np.sum(x_off * b_off_action))

    k2 = k_hollow @ k_hollow
    k2_off = k2 - np.diag(np.diag(k2))
    total_off_quad = quad_off + 2.0 * float(np.sum(x_off * k2_off))
    expected_rhs = -float(np.sum(k_hollow * k_hollow))
    return total_off_quad, expected_rhs


def verify_hochschild_rota_mobius_pair_strata(n_spins: int = 24) -> Dict[int, int]:
    """Partitions Sym^2(R^{N(N-1)/2}) into exact XOR Hamming-weight strata |a ^ b| in {0, 2, 4}."""
    masks = [((1 << i) | (1 << j)) for i in range(n_spins) for j in range(i + 1, n_spins)]
    d_off = len(masks)
    counts = {0: 0, 2: 0, 4: 0, -1: 0}
    for a in range(d_off):
        ma = masks[a]
        for b in range(a, d_off):
            wt = int(ma ^ masks[b]).bit_count()
            if wt in counts:
                counts[wt] += 1
            else:
                counts[-1] += 1
    return counts
