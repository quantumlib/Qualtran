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

from openfermion.resource_estimates.utils import QI, QR

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray


def test_rotations():
    num_spin_orb = 108
    num_aux = 360
    num_bits_rot = 10
    num_eig = 13031
    rot = ProgRotGateArray(
        num_aux=num_aux, num_eig=num_eig, num_spin_orb=num_spin_orb, num_bits_rot=num_bits_rot
    )
    _, counts = rot.call_graph()
    toff = counts[TGate()] // 4
    rot = ProgRotGateArray(
        num_aux=num_aux, num_eig=num_eig, num_spin_orb=num_spin_orb, num_bits_rot=num_bits_rot
    ).adjoint()
    _, counts = rot.call_graph()
    toff += counts[TGate()] // 4
    toff *= 2  # cost is for the two applications of the (rot, rot^) pair
    # the rot gate array includes the offset addition, qrom and cost for applying the rotations.
    # it does not include the swaps and the controlled Z which is included in the openfermion costs.
    # cost4ah cost4bg cost4df
    # the + 3 is the QROM difference when not loading the one-body part after
    # the reflection in the middle of the circuit.
    nlxi = (num_eig + num_spin_orb // 2 - 1).bit_length()
    cost4ah = 4 * (nlxi - 1)
    # The costs of the QROMs and their inverses in steps 4 (b) and (g).
    cost4bg = (
        QR(num_eig + num_spin_orb // 2, num_spin_orb * num_bits_rot // 2)[1]
        + QI(num_eig + num_spin_orb // 2)[1]
        + QR(num_eig, num_spin_orb * num_bits_rot // 2)[1]
        + QI(num_eig)[1]
    )
    delta_qr = (
        QR(num_eig + num_spin_orb // 2, num_spin_orb * num_bits_rot // 2)[1]
        - QR(num_eig, num_spin_orb * num_bits_rot // 2)[1]
    )
    delta_qi = QI(num_eig + num_spin_orb // 2)[1] - QI(num_eig)[1]
    # The controlled rotations in steps 4 (d) and (f).
    cost4df = 4 * num_spin_orb * (num_bits_rot - 2)
    # The controlled Z operations in the middle for step 4 (e).
    of_cost = cost4ah + cost4bg + cost4df
    toff -= delta_qr + delta_qi
    assert toff == of_cost
