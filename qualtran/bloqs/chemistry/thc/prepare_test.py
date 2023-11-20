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
import networkx as nx
import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.chemistry.thc import PrepareTHC, UniformSuperpositionTHC
from qualtran.bloqs.chemistry.thc.notebook_utils import generalize as thc_generalize
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling
from qualtran.testing import execute_notebook


def _make_uniform_superposition():
    from qualtran.bloqs.chemistry.thc import UniformSuperpositionTHC

    num_mu = 10
    num_spin_orb = 4
    return UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)


def _make_prepare():
    from qualtran.bloqs.chemistry.thc import PrepareTHC

    num_spat = 4
    num_mu = 8
    t_l = np.random.normal(0, 1, size=num_spat)
    zeta = np.random.normal(0, 1, size=(num_mu, num_mu))
    zeta = 0.5 * (zeta + zeta.T)
    return PrepareTHC.from_hamiltonian_coeffs(t_l, zeta, num_bits_state_prep=8)


def test_uniform_superposition():
    num_mu = 10
    num_spin_orb = 4
    usup = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(usup)


@pytest.mark.parametrize("num_mu, num_spat, mu", ((10, 4, 10), (40, 10, 17), (72, 31, 27)))
def test_prepare_alt_keep_vals(num_mu, num_spat, mu):
    np.random.seed(7)
    t_l = np.random.normal(0, 1, size=num_spat)
    zeta = np.random.normal(0, 1, size=(num_mu, num_mu))
    zeta = 0.5 * (zeta + zeta.T)
    prep = PrepareTHC.from_hamiltonian_coeffs(t_l, zeta, num_bits_state_prep=mu)
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
    alternates, keep_numers, mu = preprocess_lcu_coefficients_for_reversible_sampling(
        flat_data, eps
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
    graph, sigma = uniform_bloq.call_graph(generalizer=thc_generalize)
    assert isinstance(graph, nx.DiGraph)
    assert isinstance(sigma, dict)


def test_notebook():
    execute_notebook('thc')
