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
from cirq_ft.algos.arithmetic_gates import LessThanEqualGate, LessThanGate
from cirq_ft.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, Register
from qualtran.bloqs.chemistry.thc import (
    add_from_bloq_register_flat_qubits,
    PrepareTHC,
    UniformSuperpositionTHC,
)
from qualtran.cirq_interop import CirqGateAsBloq
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
    eps = 1e-3
    return PrepareTHC.build(t_l, zeta, probability_epsilon=eps)


def test_split_join_arithmetic_gates():
    bb = BloqBuilder()
    bitsize = 9
    val = bb.add_register(Register("val", bitsize=bitsize))
    res = bb.add_register(Register("res", bitsize=1))
    val, res = add_from_bloq_register_flat_qubits(
        bb, CirqGateAsBloq(LessThanGate(bitsize, 7)), val=val, res=res
    )
    cbloq = bb.finalize(val=val, res=res)
    assert cbloq.t_complexity() == CirqGateAsBloq(LessThanGate(bitsize, 7)).t_complexity()
    bb = BloqBuilder()
    x = bb.add_register(Register("x", bitsize=bitsize))
    y = bb.add_register(Register("y", bitsize=bitsize))
    res = bb.add_register(Register("res", bitsize=1))
    x, y, res = add_from_bloq_register_flat_qubits(
        bb, CirqGateAsBloq(LessThanEqualGate(bitsize, bitsize)), x=x, y=y, res=res
    )
    cbloq = bb.finalize(x=x, y=y, res=res)
    assert (
        cbloq.t_complexity() == CirqGateAsBloq(LessThanEqualGate(bitsize, bitsize)).t_complexity()
    )


def test_uniform_superposition():
    num_mu = 10
    num_spin_orb = 4
    usup = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(usup)


@pytest.mark.parametrize("num_mu, num_spat, eps", ((10, 4, 1e-3), (40, 10, 1e-5), (72, 31, 1e-8)))
def test_prepare_alt_keep_vals(num_mu, num_spat, eps):
    np.random.seed(7)
    t_l = np.random.normal(0, 1, size=num_spat)
    zeta = np.random.normal(0, 1, size=(num_mu, num_mu))
    zeta = 0.5 * (zeta + zeta.T)
    prep = PrepareTHC.build(t_l, zeta, probability_epsilon=eps)
    qlt_testing.assert_valid_bloq_decomposition(prep)
    # Test that the alt / keep values are correct
    qlt_testing.assert_valid_bloq_decomposition(prep)
    triu_indices = np.triu_indices(num_mu)
    enlarged_matrix = np.zeros((num_mu + 1, num_mu + 1))
    # THC paper uses column major ordering of the data it seems
    enlarged_matrix[:num_mu, :num_mu] = np.abs(zeta)
    enlarged_matrix[:num_spat, num_mu] = np.abs(t_l)
    flat_data = np.abs(np.concatenate([zeta[triu_indices], t_l]))
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
    from qualtran.bloqs.chemistry.thc_notebook_utils import generalize
    from qualtran.drawing import GraphvizCounts
    from qualtran.resource_counting import get_bloq_counts_graph

    num_mu = 10
    num_spin_orb = 4
    uniform_bloq = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    graph, sigma = get_bloq_counts_graph(uniform_bloq, generalizer=generalize)


def test_notebook():
    execute_notebook('thc')
