# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
try:
    from qualtran.bloqs.algorithms.dqi.star_anisotropy_syzygy import (
        compute_zero_iteration_partition_syzygy_bound,
        evaluate_inhomogeneous_bianchi_quadratic_form,
        verify_hochschild_rota_mobius_pair_strata,
    )
except ImportError:
    try:
        from qualtran_dropin.star_anisotropy_syzygy import (
            compute_zero_iteration_partition_syzygy_bound,
            evaluate_inhomogeneous_bianchi_quadratic_form,
            verify_hochschild_rota_mobius_pair_strata,
        )
    except ImportError:
        from star_anisotropy_syzygy import (
            compute_zero_iteration_partition_syzygy_bound,
            evaluate_inhomogeneous_bianchi_quadratic_form,
            verify_hochschild_rota_mobius_pair_strata,
        )


def test_zero_iteration_partition_syzygy_eliminates_598_penalty():
    rng = np.random.default_rng(42)
    n_spins = 24
    d_off = n_spins * (n_spins - 1) // 2
    x_planted = rng.choice([-1.0, 1.0], size=n_spins)
    z_off = np.array(
        [x_planted[i] * x_planted[j] for i in range(n_spins) for j in range(i + 1, n_spins)],
        dtype=np.float64,
    )

    proj = np.eye(d_off) - np.outer(z_off, z_off) / float(d_off)
    m_off = -24.930073 * (np.outer(z_off, z_off) / float(d_off)) + 5.0 * proj
    res = compute_zero_iteration_partition_syzygy_bound(m_off, n_spins=n_spins)

    assert np.isclose(res["phantom_penalty_eliminated"], 24.0 * 24.930073, atol=1e-6)
    assert np.isclose(
        res["zero_iter_syzygy_bound"] - res["naive_300_bound"], 598.321752, atol=1e-5
    )


def test_inhomogeneous_offdiag_bianchi_identity_exact_on_hypercube():
    rng = np.random.default_rng(101)
    n_spins = 24
    k_raw = rng.standard_normal((n_spins, n_spins))
    k_hollow = 0.5 * (k_raw + k_raw.T)
    np.fill_diagonal(k_hollow, 0.0)

    for _ in range(25):
        x_spins = rng.choice([-1.0, 1.0], size=n_spins)
        lhs, rhs = evaluate_inhomogeneous_bianchi_quadratic_form(k_hollow, x_spins)
        assert abs(lhs - rhs) < 1e-12


def test_hochschild_pair_strata_counts():
    counts = verify_hochschild_rota_mobius_pair_strata(24)
    assert counts[0] == 276
    assert counts[2] == 6072
    assert counts[4] == 31878
    assert counts[-1] == 0
    assert counts[0] + counts[2] + counts[4] == 38226
