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
import itertools

import cirq

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import TGate, Toffoli, ZeroState
from qualtran.resource_counting import get_bloq_counts_graph


def _make_Toffoli():
    from qualtran.bloqs.basic_gates import Toffoli

    return Toffoli()


def test_toffoli_t_count():
    counts = Toffoli().bloq_counts()
    assert counts == {(4, TGate())}

    _, sigma = get_bloq_counts_graph(Toffoli())
    assert sigma == {TGate(): 4}


def test_toffoli_cirq():
    bb = BloqBuilder()
    c0, c1, trg = [bb.add(ZeroState()) for _ in range(3)]
    ctrl, target = bb.add(Toffoli(), ctrl=[c0, c1], target=trg)
    ctrl, target = bb.add(Toffoli(), ctrl=ctrl, target=target)
    cbloq = bb.finalize(q0=ctrl[0], q1=ctrl[1], q2=target)

    circuit, qubits = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        """\
_c(0): ───@───@───
          │   │
_c(1): ───@───@───
          │   │
_c(2): ───X───X───""",
    )


def test_classical_sim():
    tof = Toffoli()

    for c0, c1 in itertools.product([0, 1], repeat=2):
        ctrl, target = tof.call_classically(ctrl=[c0, c1], target=0)
        assert ctrl.tolist() == [c0, c1]
        if c0 == 1 and c1 == 1:
            assert target == 1
        else:
            assert target == 0


def test_classical_sim_2():
    bb = BloqBuilder()
    c0, c1, trg = [bb.add(ZeroState()) for _ in range(3)]
    ctrl, target = bb.add(Toffoli(), ctrl=[c0, c1], target=trg)
    ctrl, target = bb.add(Toffoli(), ctrl=ctrl, target=target)
    cbloq = bb.finalize(q0=ctrl[0], q1=ctrl[1], q2=target)

    b0, b1, b2 = cbloq.call_classically()
    assert b0 == 0
    assert b1 == 0
    assert b2 == 0
