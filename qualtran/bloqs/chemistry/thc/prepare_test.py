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
from typing import Optional

import networkx as nx
import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.chemistry.thc.notebook_utils import GENERALIZERS as THC_GENERALIZERS
from qualtran.bloqs.chemistry.thc.prepare import (
    _thc_prep,
    _thc_uni,
    PrepareTHC,
    UniformSuperpositionTHC,
)
from qualtran.drawing.musical_score import get_musical_score_data, MusicalScoreData
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type
from qualtran.resource_counting.generalizers import generalize_cswap_approx, ignore_split_join
from qualtran.testing import execute_notebook


def test_thc_uniform_prep(bloq_autotester):
    bloq_autotester(_thc_uni)


def test_thc_prepare(bloq_autotester):
    bloq_autotester(_thc_prep)


def build_random_test_integrals(num_mu: int, num_spat: int, seed: Optional[int] = None):
    """Build random THC integrals for testing / demonstration purposes.

    Args:
        num_mu: The THC auxiliary dimension.
        num_spat: The number of spatial orbitals.
        seed: If not None then seed the rng with this value. Otherwise seed is
            not set explcitly in this function.

    Retuns:
        t_l: The eigenvalues of the one-body Hamiltonian.
        eta: The THC leaf tensors (matrix) of size num_mu x num_spat.
        zeta: The central tensor (matrix) of size num_mu x num_mu.
    """
    rs = np.random.RandomState(seed)
    tpq = rs.normal(size=(num_spat, num_spat))
    tpq = 0.5 * (tpq + tpq.T)
    t_l = np.linalg.eigvalsh(tpq)
    zeta = rs.normal(size=(num_mu, num_mu))
    zeta = 0.5 * (zeta + zeta.T)
    eta = rs.normal(size=(num_mu, num_spat))
    return t_l, eta, zeta


@pytest.mark.parametrize("num_mu, num_spat, mu", ((10, 4, 10), (40, 10, 17), (72, 31, 27)))
def test_prepare_alt_keep_vals(num_mu, num_spat, mu):
    t_l, eta, zeta = build_random_test_integrals(num_mu, num_spat, seed=7)
    prep = PrepareTHC.from_hamiltonian_coeffs(t_l, eta, zeta, num_bits_state_prep=mu)
    qlt_testing.assert_valid_bloq_decomposition(prep)
    # Test that the alt / keep values are correct
    qlt_testing.assert_valid_bloq_decomposition(prep)
    triu_indices = np.triu_indices(num_mu)
    enlarged_matrix = np.zeros((num_mu + 1, num_mu + 1))
    # THC paper uses column major ordering of the data it seems
    enlarged_matrix[:num_mu, :num_mu] = np.abs(zeta)
    enlarged_matrix[:num_spat, num_mu] = np.abs(t_l)
    flat_data = np.abs(np.concatenate([zeta[triu_indices], t_l]))
    eps = 2**-mu / len(flat_data)
    alternates, keep_numers, mu = preprocess_probabilities_for_reversible_sampling(
        flat_data, sub_bit_precision=mu
    )
    keep_denom = 2**mu
    data_len = len(flat_data)
    num_ut = len(triu_indices[0])
    # Test alt_mu / alt_nu vales
    # stolen from openfermion unit test
    out_distribution = [1 / data_len * numer / keep_denom for numer in keep_numers]
    unraveled_dist = np.zeros_like(enlarged_matrix)
    unraveled_dist[triu_indices] = out_distribution[:num_ut]
    unraveled_dist[:num_spat, num_mu] = out_distribution[num_ut:]
    total = np.sum(flat_data)
    for i in range(data_len):
        switch_probability = 1 - keep_numers[i] / keep_denom
        unraveled_dist[prep.alt_mu[i], prep.alt_nu[i]] += 1 / data_len * switch_probability
    assert np.allclose(unraveled_dist[triu_indices], flat_data[:num_ut] / total, atol=eps)
    assert np.allclose(unraveled_dist[:num_spat, num_mu], flat_data[num_ut:] / total, atol=eps)


def test_prepare_graph():
    num_mu = 10
    num_spin_orb = 4
    uniform_bloq = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    graph, sigma = uniform_bloq.call_graph(generalizer=THC_GENERALIZERS)
    assert isinstance(graph, nx.DiGraph)
    assert isinstance(sigma, dict)


def test_prepare_qrom_counts():
    num_spat = 4
    num_mu = 8
    t_l, eta, zeta = build_random_test_integrals(num_mu, num_spat, seed=7)
    thc_prep = PrepareTHC.from_hamiltonian_coeffs(t_l, eta, zeta, num_bits_state_prep=8)
    binned_counts = classify_t_count_by_bloq_type(thc_prep)
    qroam = thc_prep.build_qrom_bloq()

    counts = get_cost_value(
        qroam, QECGatesCost(), generalizer=generalize_cswap_approx
    ).total_t_and_ccz_count()
    assert binned_counts['data_loading'] == counts['n_ccz'] * 4, binned_counts['data_loading']


def test_equivalent_bloq_counts():
    prepare = _thc_prep.make()
    qlt_testing.assert_equivalent_bloq_counts(prepare, ignore_split_join)


def test_musical_score():
    uni = _thc_uni()
    msd = get_musical_score_data(uni)
    assert isinstance(msd, MusicalScoreData)
    prep = _thc_prep()
    msd = get_musical_score_data(prep)
    assert isinstance(msd, MusicalScoreData)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('thc')
