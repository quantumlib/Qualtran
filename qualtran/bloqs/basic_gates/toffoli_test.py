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
import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Toffoli, ZeroState
from qualtran.bloqs.basic_gates.toffoli import _toffoli
from qualtran.drawing.musical_score import Circle, ModPlus
from qualtran.testing import assert_wire_symbols_match_expected


def test_toffoli(bloq_autotester):
    bloq_autotester(_toffoli)


def test_toffoli_sigma():
    _, sigma = Toffoli().call_graph()
    assert sigma == {Toffoli(): 1}


def test_toffoli_cirq():
    bb = BloqBuilder()
    c0, c1, trg = [bb.add(ZeroState()) for _ in range(3)]
    ctrl, target = bb.add(Toffoli(), ctrl=[c0, c1], target=trg)
    ctrl, target = bb.add(Toffoli(), ctrl=ctrl, target=target)
    cbloq = bb.finalize(q0=ctrl[0], q1=ctrl[1], q2=target)

    circuit, qubits = cbloq.to_cirq_circuit_and_quregs()
    cirq.testing.assert_has_diagram(
        circuit,
        """\
_c(0): ───@───@───
          │   │
_c(1): ───@───@───
          │   │
_c(2): ───X───X───""",
    )
    assert_wire_symbols_match_expected(
        Toffoli(), [Circle(filled=True), Circle(filled=True), ModPlus()]
    )


def test_classical_sim():
    tof = Toffoli()

    for c0, c1 in itertools.product([0, 1], repeat=2):
        ctrl, target = tof.call_classically(ctrl=np.asarray([c0, c1]), target=0)
        assert isinstance(ctrl, np.ndarray)
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


def test_toffoli_tensors():
    tof = Toffoli()
    unitary = tof.tensor_contract()
    cirq_unitary = cirq.unitary(cirq.TOFFOLI)
    np.testing.assert_allclose(cirq_unitary, unitary)
