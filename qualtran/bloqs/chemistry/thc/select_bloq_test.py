#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import QFxp
from qualtran.bloqs.chemistry.thc.select_bloq import (
    _leaf_tensor_to_givens_rotations,
    _thc_rotations,
    _thc_sel,
    SelectTHC,
    THCRotations,
)
from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type


def test_thc_rotations_autotester(bloq_autotester):
    bloq_autotester(_thc_rotations)


def test_thc_select_autotester(bloq_autotester):
    bloq_autotester(_thc_sel)


def test_leaf_tensor_to_givens_rotations():
    """Tests that the computed thetas rotate from basis set to THC basis."""
    num_mu = 5
    num_spatial = 4
    num_bits_theta = 25

    rng = np.random.default_rng(42)
    eta = rng.normal(size=(num_mu, num_spatial))

    dtype = QFxp(num_bits_theta, num_bits_theta, signed=False)
    thetas = _leaf_tensor_to_givens_rotations(eta, num_bits_theta)

    for mu in range(num_mu):
        U = np.eye(num_spatial)
        eta_mu = eta[mu]
        u = eta_mu / np.linalg.norm(eta_mu)
        # build full rotation operator from Givens rotations angles
        for k, theta in enumerate(thetas[:, mu]):
            theta_f = dtype.float_from_fixed_width_int(theta) * 4 * np.pi
            G_2d = np.array(
                [[np.cos(theta_f), -np.sin(theta_f)], [np.sin(theta_f), np.cos(theta_f)]]
            )
            G_full = np.eye(num_spatial)
            G_full[k : k + 2, k : k + 2] = G_2d
            U = G_full @ U

        # test U * (1,0,0...) = eta/norm(eta)
        testvec = np.zeros(num_spatial)
        testvec[0] = 1.0
        result = U @ testvec - u
        assert np.allclose(result, np.zeros(num_spatial), atol=1e-6)


@pytest.mark.parametrize(
    "num_mu, num_spatial, num_bits_theta, kr1, kr2",
    ((2, 4, 12, 1, 1), (4, 8, 15, 2, 2), (8, 16, 15, 2, 2)),
)
def test_thc_select_cost(num_mu, num_spatial, num_bits_theta, kr1, kr2):
    num_spin_orb = num_spatial * 2
    rng = np.random.default_rng(42)
    eta = rng.normal(size=(num_mu, num_spatial))  # THC vectors
    tpq = rng.normal(size=(num_spatial, num_spatial))
    tpq = 0.5 * (tpq + tpq.T)

    thc_sel = SelectTHC(
        num_mu=num_mu,
        num_spin_orb=num_spin_orb,
        num_bits_theta=num_bits_theta,
        keep_bitsize=10,
        kr1=kr1,
        kr2=kr2,
        eta=eta,
        tpq=tpq,
    )

    binned_counts_t = classify_t_count_by_bloq_type(thc_sel.decompose_bloq())
    binned_counts_toffoli = {k: v / 4 for k, v in binned_counts_t.items()}
    tot_counts_toffoli = sum(binned_counts_toffoli.values())

    # Toffoli cost according to the formula in the paper
    paper_cost_toffoli = num_spin_orb  # swaps controlled on spin (doubled for mu/nu)
    paper_cost_toffoli += np.ceil((num_mu + num_spatial) / kr1) - 2  # QROM load mu
    paper_cost_toffoli += np.ceil(num_mu / kr2) - 2  # QROM load nu
    paper_cost_toffoli += 2 * num_spin_orb * (num_bits_theta - 2)  # rotations (doubled for mu/nu)
    paper_cost_toffoli += (
        2 * num_spin_orb * (num_bits_theta - 2)
    )  # invert rotations (doubled for mu/nu)
    paper_cost_toffoli += np.ceil(1.0 * (num_mu + num_spatial) / kr1) + kr1  # QROM erase mu
    paper_cost_toffoli += np.ceil(1.0 * num_mu / kr2) + kr2  # QROM erase nu
    paper_cost_toffoli += num_spin_orb  # swaps controlled by ancilla (doubled for mu/nu)
    paper_cost_toffoli += 2  # extra gates (CSwap on plus_a, plus_b and controlled Z)

    # Corrections for the implementation in qualtran, not in the paper
    cost_correction = 0

    # correction - rotations include phase gradient state preparation
    rotation_correct = 4 * (num_spin_orb - 2) * (num_bits_theta - 1) + 22 * num_bits_theta - 64
    rotation_incorrect = 4 * num_spin_orb * (num_bits_theta - 2)
    cost_correction += rotation_correct - rotation_incorrect

    # correction - QROAM load costs are slightly different according to docstrings
    b = (num_spatial - 1) * num_bits_theta
    load_mu_term_correct = (kr1 - 1) * b
    load_nu_term_correct = (kr2 - 1) * b
    cost_correction += load_mu_term_correct + load_nu_term_correct

    # correction - QROAM erase always uses the optimal batch size
    k1 = int(np.round(0.5 * np.log2(num_mu + num_spatial)))
    k2 = int(np.round(0.5 * np.log2(num_mu)))
    kr1_optimal = 2**k1
    kr2_optimal = 2**k2
    erase_mu_incorrect = np.ceil(1.0 * (num_mu + num_spatial) / kr1) + kr1
    erase_nu_incorrect = np.ceil(1.0 * num_mu / kr2) + kr2
    erase_mu_correct = max(0, np.ceil((num_mu + num_spatial) / kr1_optimal) + (kr1_optimal - 4))
    erase_nu_correct = max(0, np.ceil(num_mu / kr2_optimal) + (kr2_optimal - 4))
    cost_correction += erase_mu_correct - erase_mu_incorrect
    cost_correction += erase_nu_correct - erase_nu_incorrect

    # correction - CSwap between mu and nu
    cost_correction += num_mu.bit_length()

    corrected_paper_cost_toffoli = paper_cost_toffoli + cost_correction
    assert tot_counts_toffoli == corrected_paper_cost_toffoli


def test_thc_rotations_from_hamiltonian_coeffs():
    rng = np.random.default_rng(42)
    num_mu = 6
    num_spatial = 4
    eta = rng.normal(size=(num_mu, num_spatial))
    tpq = rng.normal(size=(num_spatial, num_spatial))
    bloq = THCRotations.from_hamiltonian_coeffs(eta, tpq=tpq, num_bits_theta=10)
    assert bloq.num_mu == num_mu
    assert bloq.num_spin_orb == 2 * num_spatial
    assert bloq.num_bits_theta == 10
    assert bloq.angles_data is not None
    assert len(bloq.angles_data) == num_spatial - 1
    assert hash(bloq) == hash(bloq)
    qlt_testing.assert_valid_bloq_decomposition(bloq)
    qlt_testing.assert_valid_bloq_decomposition(bloq.adjoint())
