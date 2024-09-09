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


from openfermion.resource_estimates.utils import power_two, QI, QR

from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.df.prepare import (
    _indexed_data,
    _prep_inner,
    _prep_outer,
    InnerPrepareDoubleFactorization,
    OuterPrepareDoubleFactorization,
    OutputIndexedData,
)
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_prep_inner(bloq_autotester):
    bloq_autotester(_prep_inner)


def test_prep_outer(bloq_autotester):
    bloq_autotester(_prep_outer)


def test_indexed_data(bloq_autotester):
    bloq_autotester(_indexed_data)


def test_outerprep_t_counts():
    num_aux = 360
    num_bits_rot_aa = 7
    num_bits_state_prep = 10
    outer_prep = OuterPrepareDoubleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    )
    toff = get_cost_value(outer_prep, QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    outer_prep = OuterPrepareDoubleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    ).adjoint()
    toff += get_cost_value(outer_prep, QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    # The output size for the QROM for the first state preparation in Eq. (C27)
    eta = power_two(num_aux + 1)
    nl = num_aux.bit_length()
    bp1 = nl + num_bits_state_prep
    cost1a = 2 * (3 * nl + 2 * num_bits_rot_aa - 3 * eta - 9)
    cost1b = QR(num_aux + 1, bp1)[1] + QI(num_aux + 1)[1]
    cost1cd = 2 * (num_bits_state_prep + nl)
    # correct the expected cost by using a different uniform superposition algorithm
    # https://github.com/quantumlib/Qualtran/issues/611
    prep = PrepareUniformSuperposition(num_aux + 1)
    cost1a_mod = get_cost_value(prep, QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    cost1a_mod += get_cost_value(prep.adjoint(), QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    assert cost1a != cost1a_mod
    assert toff == cost1a_mod + cost1b + cost1cd


def test_indexed_data_toffoli_counts():
    num_spin_orb = 108
    num_aux = 360
    num_bits_rot_aa = 7
    num_eig = 13031
    in_l_data_l = OutputIndexedData(
        num_aux=num_aux, num_spin_orb=num_spin_orb, num_eig=num_eig, num_bits_rot_aa=num_bits_rot_aa
    )
    _, counts = in_l_data_l.call_graph()
    toff = counts[Toffoli()]
    in_l_data_l = OutputIndexedData(
        num_aux=num_aux, num_spin_orb=num_spin_orb, num_eig=num_eig, num_bits_rot_aa=num_bits_rot_aa
    ).adjoint()
    _, counts = in_l_data_l.call_graph()
    toff += counts[Toffoli()]
    # captured from cost2 in openfermion df.compute_cost
    nxi = (num_spin_orb // 2 - 1).bit_length()
    nlxi = (num_eig + num_spin_orb // 2 - 1).bit_length()
    bo = nxi + nlxi + num_bits_rot_aa + 1
    of_cost = QR(num_aux + 1, bo)[1] + QI(num_aux + 1)[1]
    assert toff == of_cost


def test_inner_prepare_t_counts():
    num_spin_orb = 108
    num_aux = 360
    num_bits_rot_aa = 7
    num_bits_state_prep = 10
    num_eig = 13031
    in_prep = InnerPrepareDoubleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_eig=num_eig,
        num_bits_rot_aa=num_bits_rot_aa,
        num_bits_state_prep=num_bits_state_prep,
    )
    toff = get_cost_value(in_prep, QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    in_prep = InnerPrepareDoubleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_eig=num_eig,
        num_bits_rot_aa=num_bits_rot_aa,
        num_bits_state_prep=num_bits_state_prep,
    ).adjoint()
    toff += get_cost_value(in_prep, QECGatesCost()).total_t_and_ccz_count()['n_ccz']
    toff *= 2  # cost is for the two applications of the in-prep, in-prep^
    # application of ciruit.
    # captured from cost3 in openfermion df.compute_cost
    nxi = (num_spin_orb // 2 - 1).bit_length()
    nlxi = (num_eig + num_spin_orb // 2 - 1).bit_length()
    cost3a = 4 * (7 * nxi + 2 * num_bits_rot_aa - 6)
    cost3b = 4 * (nlxi - 1)
    bp2 = nxi + num_bits_state_prep + 2
    cost3c = (
        QR(num_eig + num_spin_orb // 2, bp2)[1]
        + QI(num_eig + num_spin_orb // 2)[1]
        + QR(num_eig, bp2)[1]
        + QI(num_eig)[1]
    )
    delta_qr = QR(num_eig + num_spin_orb // 2, bp2)[1] - QR(num_eig, bp2)[1]
    delta_qi = QI(num_eig + num_spin_orb // 2)[1] - QI(num_eig)[1]
    cost3d = 4 * (nxi + num_bits_state_prep)
    cost3 = cost3a + cost3b + cost3c + cost3d
    toff -= delta_qr + delta_qi
    assert toff == cost3
