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
from qualtran import Bloq
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates.t_gate import TGate
from qualtran.bloqs.chemistry.trotter.hubbard.trotter_step import (
    build_plaq_unitary_second_order_suzuki,
)
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.resource_counting.generalizers import PHI


def catch_rotations(bloq) -> Bloq:
    if isinstance(bloq, Rz):
        if abs(bloq.angle) < 1e-12:
            return ArbitraryClifford(1)
        else:
            return Rz(angle=PHI)
    return bloq


def test_second_order_suzuki_costs():
    length = 8
    u = 4
    dt = 0.1
    unitary = build_plaq_unitary_second_order_suzuki(length, u, dt)
    _, sigma = unitary.call_graph(generalizer=catch_rotations)
    # there are 3 hopping unitaries contributing 8 Ts from from the F gate
    assert sigma[TGate()] == (3 * length**2 // 2) * 8
    # 3 hopping unitaries and 2 interaction unitaries
    assert sigma[Rz(PHI)] == (3 * length**2 + 2 * length**2)
