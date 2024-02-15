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
import attrs
import numpy as np
import pytest
from openfermion.resource_estimates.sparse.costing_sparse import cost_sparse
from openfermion.resource_estimates.utils import power_two, QI

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.testing import execute_notebook


def make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa):
    tpq = np.random.random((num_spin_orb // 2, num_spin_orb // 2))
    tpq = 0.5 * (tpq + tpq.T)
    eris = np.random.random((num_spin_orb // 2,) * 4)
    eris += np.transpose(eris, (0, 1, 3, 2))
    eris += np.transpose(eris, (1, 0, 2, 3))
    eris += np.transpose(eris, (2, 3, 0, 1))
    prep_sparse = PrepareSparse.from_hamiltonian_coeffs(
        num_spin_orb,
        tpq,
        eris,
        qroam_block_size=32,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
    )
    num_nnz = len(prep_sparse.alt_pqrs[0])
    return prep_sparse, num_nnz


@pytest.mark.parametrize("num_spin_orb, num_bits_rot_aa", ((8, 3), (20, 3), (57, 3)))
def test_sparse_costs_against_openfermion(num_spin_orb, num_bits_rot_aa):
    num_bits_state_prep = 12
    cost = 0
    bloq = SelectSparse(num_spin_orb)
    _, sigma = bloq.call_graph()
    cost += sigma[TGate()]
    bloq, num_non_zero = make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa)
    _, sigma = bloq.call_graph()
    cost += sigma[TGate()]
    bloq = attrs.evolve(bloq, adjoint=True, qroam_block_size=2 ** QI(num_non_zero)[0])
    _, sigma = bloq.call_graph()
    cost += sigma[TGate()]
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 10
    logd = (num_non_zero - 1).bit_length()
    refl_cost = 4 * (num_bits_state_prep + logd + 4)  # page 40 listing 4
    delta_swap = 8 * (7 - 4) * (num_spin_orb // 2 - 1).bit_length()
    cost_of = cost_sparse(
        num_spin_orb, unused_lambda, num_non_zero, unused_de, num_bits_state_prep, unused_stps
    )[0]
    # correct the expected cost by using a different uniform superposition algorithm
    # see: https://github.com/quantumlib/Qualtran/issues/611
    eta = power_two(bloq.num_non_zero)
    cost_uni_prep = (
        4 * 2 * (3 * (bloq.num_non_zero - 1).bit_length() + 3 * eta + 2 * num_bits_rot_aa - 9)
    )
    prep = PrepareUniformSuperposition(bloq.num_non_zero)
    cost1a_mod = prep.call_graph()[1][TGate()]
    cost1a_mod += prep.adjoint().call_graph()[1][TGate()]
    adjusted_cost_qualtran = (cost - cost1a_mod + cost_uni_prep + refl_cost - delta_swap) // 4
    assert adjusted_cost_qualtran == cost_of


def test_notebook():
    execute_notebook("sparse")
