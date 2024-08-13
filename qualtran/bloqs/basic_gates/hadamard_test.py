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
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Hadamard, OneEffect, OneState
from qualtran.bloqs.basic_gates.hadamard import _hadamard, CHadamard
from qualtran.cirq_interop import cirq_gate_to_bloq


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(OneState())
    q = bb.add(Hadamard(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───X───H───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_hadamard(bloq_autotester):
    bloq_autotester(_hadamard)


def test_unitary_vs_cirq():
    h = Hadamard()
    unitary = h.tensor_contract()
    cirq_unitary = cirq.unitary(cirq.H)
    np.testing.assert_allclose(unitary, cirq_unitary)


def test_not_classical():
    h = Hadamard()
    with pytest.raises(NotImplementedError, match=r'.*is not classically simulable\.'):
        h.call_classically(q=0)


def test_chadamard_vs_cirq():
    bloq = Hadamard().controlled()
    assert bloq == CHadamard()

    gate = cirq.H.controlled()
    np.testing.assert_allclose(cirq.unitary(gate), bloq.tensor_contract())


def test_cirq_interop():
    circuit = CHadamard().as_composite_bloq().to_cirq_circuit()
    should_be = cirq.Circuit(
        [cirq.Moment(cirq.H(cirq.NamedQubit('target')).controlled_by(cirq.NamedQubit('ctrl')))]
    )
    assert circuit == should_be

    (op,) = list(should_be.all_operations())
    assert op.gate is not None
    assert cirq_gate_to_bloq(op.gate) == CHadamard()


def test_active_chadamard_is_hadamard():
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    ctrl_on = bb.add(OneState())
    ctrl_on, q = bb.add(CHadamard(), ctrl=ctrl_on, target=q)
    bb.add(OneEffect(), q=ctrl_on)
    cbloq = bb.finalize(q=q)

    np.testing.assert_allclose(Hadamard().tensor_contract(), cbloq.tensor_contract())


def test_chadamard_adjoint():
    bb = BloqBuilder()
    ctrl = bb.add_register('ctrl', 1)
    q = bb.add_register('q', 1)
    ctrl, q = bb.add(CHadamard(), ctrl=ctrl, target=q)
    ctrl, q = bb.add(CHadamard().adjoint(), ctrl=ctrl, target=q)
    cbloq = bb.finalize(ctrl=ctrl, q=q)

    np.testing.assert_allclose(np.eye(4), cbloq.tensor_contract(), atol=1e-12)
