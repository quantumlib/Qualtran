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
import pytest
from openfermion.resource_estimates.df.compute_cost_df import compute_cost
from openfermion.resource_estimates.utils import QI, QR

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.double_factorization import (
    DoubleFactorization,
    get_num_bits_lxi,
    get_qroam_cost,
    InnerPrepare,
    OuterPrepare,
    OutputIndexedData,
    ProgRotGateArray,
)
from qualtran.resource_counting import get_bloq_counts_graph
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _make_double_factorization():
    from qualtran.bloqs.chemistry.double_factorization import DoubleFactorization

    return DoubleFactorization(10, 20, 8)


def test_double_factorization():
    df = DoubleFactorization(10, 12, 8)
    assert_valid_bloq_decomposition(df)


def test_double_factorization_counts_graph():
    graph, sigma = get_bloq_counts_graph(DoubleFactorization(4, 10, 4))
    assert sigma[TGate()] == 4656


@pytest.mark.parametrize("data_size, bitsize", ((100, 10), (100, 3), (1_000, 13), (1_000_000, 20)))
def test_qroam_factors(data_size, bitsize):
    assert get_qroam_cost(data_size, bitsize) == QR(data_size, bitsize)[-1]
    assert get_qroam_cost(data_size, bitsize, adjoint=True) == QI(data_size)[-1]


def test_outerprep_bloq_counts():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 1
    outer_prep = OuterPrepare(
        num_aux,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot=num_bits_rot,  # computed by of?
    )
    _, counts = get_bloq_counts_graph(outer_prep)
    toff = counts[TGate()] // 4
    outer_prep = OuterPrepare(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot=num_bits_rot, adjoint=True
    )
    _, counts = get_bloq_counts_graph(outer_prep)
    toff += counts[TGate()] // 4
    assert toff == 117


def test_indexed_data_bloq_counts():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 1  # decided by OF
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    in_l_data_l = OutputIndexedData(
        num_aux=num_aux, num_spin_orb=num_spin_orb, num_xi=num_eig, num_bits_rot_aa=num_bits_rot
    )
    _, counts = get_bloq_counts_graph(in_l_data_l)
    toff = counts[TGate()] // 4
    in_l_data_l = OutputIndexedData(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_xi=num_eig,
        num_bits_rot_aa=num_bits_rot,
        adjoint=True,
    )
    _, counts = get_bloq_counts_graph(in_l_data_l)
    toff += counts[TGate()] // 4
    assert toff == 54


def test_inner_prepare():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 7  # decided by OF
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_lxi = get_num_bits_lxi(num_aux, num_eig, num_spin_orb)
    in_prep = InnerPrepare(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_xi=num_eig,
        num_bits_rot_aa=num_bits_rot,
        num_bits_offset=num_bits_lxi,
        num_bits_state_prep=num_bits_state_prep,
        adjoint=False,
    )
    _, counts = get_bloq_counts_graph(in_prep)
    toff = counts[TGate()] // 4
    in_prep = InnerPrepare(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_xi=num_eig,
        num_bits_rot_aa=num_bits_rot,
        num_bits_offset=num_bits_lxi,
        num_bits_state_prep=num_bits_state_prep,
        adjoint=True,
    )
    _, counts = get_bloq_counts_graph(in_prep)
    of_cost = compute_cost(
        num_spin_orb, 10, 10, num_aux, num_eig * num_aux, num_bits_state_prep, num_bits_rot, 10
    )
    toff += counts[TGate()] // 4
    toff *= 2  # cost is for the two applications of the in-prep, in-prep^
    # + 1 for accounting for openfermion not outputting one-body ham for second
    # application of ciruit.
    assert toff == 497 + 1


def test_rotations():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 7  # decided by OF
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_lxi = get_num_bits_lxi(num_aux, num_eig, num_spin_orb)
    rot = ProgRotGateArray(
        num_aux=num_aux,
        num_xi=num_eig,
        num_spin_orb=num_spin_orb,
        num_bits_rot=num_bits_rot,
        num_bits_offset=num_bits_lxi,
        adjoint=False,
    )
    _, counts = get_bloq_counts_graph(rot)
    toff = counts[TGate()] // 4
    rot = ProgRotGateArray(
        num_aux=num_aux,
        num_xi=num_eig,
        num_spin_orb=num_spin_orb,
        num_bits_rot=num_bits_rot,
        num_bits_offset=num_bits_lxi,
        adjoint=True,
    )
    _, counts = get_bloq_counts_graph(rot)
    of_cost = compute_cost(
        num_spin_orb, 10, 10, num_aux, num_eig * num_aux, num_bits_state_prep, num_bits_rot, 10
    )
    toff += counts[TGate()] // 4
    toff *= 2  # cost is for the two applications of the (rot, rot^) pair
    # the rot gate array includes the offset addition, qrom and cost for applying the rotations.
    # it does not include the swaps and the controlled Z which is included in the openfermion costs.
    # cost4ah cost4bg cost4df
    # the + 3 is the QROM difference when not loading the one-body part after
    # the reflection in the middle of the circuit.
    assert toff == 615 + 3


def test_compare_cost_to_openfermion():
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 12
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    bloq = DoubleFactorization(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=1,
        num_bits_rot_aa_inner=7,
        num_bits_rot=num_bits_rot,
    )
    _, counts = get_bloq_counts_graph(bloq)
    # https://github.com/quantumlib/OpenFermion/issues/839
    of_cost = compute_cost(
        num_spin_orb,
        unused_lambda,
        unused_de,
        num_aux,
        num_eig * num_aux,
        num_bits_state_prep,
        num_bits_rot,
        unused_stps,
    )[0]
    print(of_cost + 6 - 2, counts[TGate()] // 4 + 22 + 2)


def test_notebook():
    execute_notebook("double_factorization")
