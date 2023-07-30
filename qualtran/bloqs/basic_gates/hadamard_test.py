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
import cirq
import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Hadamard


def _make_Hadamard():
    from qualtran.bloqs.basic_gates import Hadamard

    return Hadamard()


def test_hadamard_cirq():
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(Hadamard(), q=q)
    cbloq = bb.finalize(q=q)

    circuit, _ = cbloq.to_cirq_circuit(q=[cirq.LineQubit(0)])
    cirq.testing.assert_has_diagram(
        circuit,
        """\
0: ───H───
""",
    )
    bb = BloqBuilder()
    q = bb.add_register('q', 3)
    q = bb.add(Hadamard(bitsize=3), q=q)
    cbloq = bb.finalize(q=q)

    circuit, _ = cbloq.to_cirq_circuit(q=[cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2)])
    cirq.testing.assert_has_diagram(
        circuit,
        """\
0: ───H───

1: ───H───

2: ───H───
""",
    )


def test_tensor_contract():
    bloq = Hadamard()
    np.testing.assert_allclose(bloq.tensor_contract(), cirq.unitary(cirq.H))
    bloq = Hadamard(bitsize=3)
    q0, q1, q2 = cirq.LineQubit.range(3)
    circ = cirq.Circuit()
    circ.append(cirq.H(q0))
    circ.append(cirq.H(q1))
    circ.append(cirq.H(q2))
    np.testing.assert_allclose(bloq.tensor_contract(), cirq.unitary(circ))
