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
from qualtran.bloqs.chemistry.sf.prepare import (
    _prep_inner,
    _prep_outer,
    InnerPrepareSingleFactorization,
    OuterPrepareSingleFactorization,
)


def test_prep_inner(bloq_autotester):
    bloq_autotester(_prep_inner)


def test_prep_outer(bloq_autotester):
    bloq_autotester(_prep_outer)


def test_outerprep_t_counts():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 1
    outer_prep = OuterPrepareSingleFactorization(
        num_aux,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot,  # computed by of?
    )
    _, counts = outer_prep.call_graph()
    toff = counts[TGate()] // 4
    outer_prep = OuterPrepareSingleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot, adjoint=True
    )
    _, counts = outer_prep.call_graph()
    toff += counts[TGate()] // 4
    # captured from OF
    assert toff == 121


def test_inner_prepare_t_counts():
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 4  # decided by OF
    num_spin_orb = 10
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot,
        adjoint=False,
        kp1=2**2,
        kp2=2**1,
    )
    _, counts = in_prep.call_graph()
    toff = counts[TGate()] // 4
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_rot_aa=num_bits_rot,
        num_bits_state_prep=num_bits_state_prep,
        adjoint=True,
        kp1=2**3,
        kp2=2**2,
    )
    _, counts = in_prep.call_graph()
    # factor of two from squaring
    toff += counts[TGate()] // 4
    toff *= 2
    # cost captured from OF correcting issues of OF inconsistencies
    assert toff == 840
