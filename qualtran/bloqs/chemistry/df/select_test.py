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
from qualtran.bloqs.chemistry.df.select import ProgRotGateArray


def test_rotations():
    num_aux = 50
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
        adjoint=False,
    )
    _, counts = rot.call_graph()
    toff = counts[TGate()] // 4
    rot = ProgRotGateArray(
        num_aux=num_aux,
        num_xi=num_eig,
        num_spin_orb=num_spin_orb,
        num_bits_rot=num_bits_rot,
        adjoint=True,
    )
    _, counts = rot.call_graph()
    toff += counts[TGate()] // 4
    toff *= 2  # cost is for the two applications of the (rot, rot^) pair
    # the rot gate array includes the offset addition, qrom and cost for applying the rotations.
    # it does not include the swaps and the controlled Z which is included in the openfermion costs.
    # cost4ah cost4bg cost4df
    # the + 3 is the QROM difference when not loading the one-body part after
    # the reflection in the middle of the circuit.
    assert toff == 615 + 3
