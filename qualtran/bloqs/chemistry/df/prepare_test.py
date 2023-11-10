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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.df.common_bitsize import get_num_bits_lxi
from qualtran.bloqs.chemistry.df.prepare import (
    InnerPrepareDoubleFactorization,
    OuterPrepareDoubleFactorization,
    OutputIndexedData,
)
from qualtran.resource_counting import get_bloq_counts_graph


def test_outerprep_bloq_counts():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 1
    outer_prep = OuterPrepareDoubleFactorization(
        num_aux,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot,  # computed by of?
    )
    _, counts = get_bloq_counts_graph(outer_prep)
    toff = counts[TGate()] // 4
    outer_prep = OuterPrepareDoubleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot, adjoint=True
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
    in_prep = InnerPrepareDoubleFactorization(
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
    in_prep = InnerPrepareDoubleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_xi=num_eig,
        num_bits_rot_aa=num_bits_rot,
        num_bits_offset=num_bits_lxi,
        num_bits_state_prep=num_bits_state_prep,
        adjoint=True,
    )
    _, counts = get_bloq_counts_graph(in_prep)
    toff += counts[TGate()] // 4
    toff *= 2  # cost is for the two applications of the in-prep, in-prep^
    # + 1 for accounting for openfermion not outputting one-body ham for second
    # application of ciruit.
    assert toff == 497 + 1
