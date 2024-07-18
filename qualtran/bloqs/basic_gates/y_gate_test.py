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
from qualtran.bloqs.basic_gates import MinusState, OneEffect, OneState, PlusState, YGate
from qualtran.bloqs.basic_gates.y_gate import _cy_gate, _y_gate, CYGate
from qualtran.cirq_interop import cirq_gate_to_bloq
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity


def test_y_gate(bloq_autotester):
    bloq_autotester(_y_gate)


def test_cy_gate(bloq_autotester):
    bloq_autotester(_cy_gate)


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(PlusState())
    q = bb.add(YGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───H───Y───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)

    bb = BloqBuilder()
    q = bb.add(MinusState())
    q = bb.add(YGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───[ _c(0): ───X───H─── ]───Y───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_cy_vs_cirq():
    bloq = YGate().controlled()
    assert bloq == CYGate()

    gate = cirq.Y.controlled()
    np.testing.assert_allclose(cirq.unitary(gate), bloq.tensor_contract())


def test_cirq_interop():
    circuit = CYGate().as_composite_bloq().to_cirq_circuit()
    should_be = cirq.Circuit(
        [cirq.Moment(cirq.Y(cirq.NamedQubit('target')).controlled_by(cirq.NamedQubit('ctrl')))]
    )
    assert circuit == should_be

    (op,) = list(should_be.all_operations())
    assert op.gate is not None
    assert cirq_gate_to_bloq(op.gate) == CYGate()


def test_active_cy_is_y():
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    ctrl_on = bb.add(OneState())
    ctrl_on, q = bb.add(CYGate(), ctrl=ctrl_on, target=q)
    bb.add(OneEffect(), q=ctrl_on)
    cbloq = bb.finalize(q=q)

    np.testing.assert_allclose(YGate().tensor_contract(), cbloq.tensor_contract())


def test_cy_adjoint():
    bb = BloqBuilder()
    ctrl = bb.add_register('ctrl', 1)
    q = bb.add_register('q', 1)
    ctrl, q = bb.add(CYGate(), ctrl=ctrl, target=q)
    ctrl, q = bb.add(CYGate().adjoint(), ctrl=ctrl, target=q)
    cbloq = bb.finalize(ctrl=ctrl, q=q)

    np.testing.assert_allclose(np.eye(4), cbloq.tensor_contract(), atol=1e-12)


def test_cy_t_complexity():
    assert t_complexity(CYGate()) == TComplexity(clifford=1)
