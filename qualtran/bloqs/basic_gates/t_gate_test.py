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
from functools import cached_property
from typing import Dict

import cirq
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, PlusState, TGate
from qualtran.bloqs.basic_gates.t_gate import _t_gate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


def test_t_gate(bloq_autotester):
    bloq_autotester(_t_gate)


def test_call_graph():
    g, simga = TGate().call_graph()
    assert simga == {TGate(): 1}
    assert TGate().t_complexity() == TComplexity(t=1)


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(PlusState())
    q = bb.add(TGate(), q=q)
    q = bb.add(TGate(is_adjoint=True), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───H───T───T^-1───")


def test_tensors():
    from_cirq = cirq.unitary(cirq.Circuit(cirq.T(cirq.LineQubit(0))))
    from_tensors = TGate().tensor_contract()
    np.testing.assert_allclose(from_cirq, from_tensors)

    from_cirq = cirq.unitary(cirq.Circuit(cirq.T(cirq.LineQubit(0)) ** -1))
    from_tensors = TGate(is_adjoint=True).tensor_contract()
    np.testing.assert_allclose(from_cirq, from_tensors)


class TestTStateMaker(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> Dict[str, 'SoquetT']:
        x = bb.add(Hadamard(), q=x)
        x = bb.add(TGate(), q=x)
        return {'x': x}


def test_test_t_state():
    q = cirq.LineQubit(0)
    from_cirq = cirq.unitary(cirq.Circuit(cirq.H(q), cirq.T(q)))
    from_tensors = TestTStateMaker().tensor_contract()
    np.testing.assert_allclose(from_cirq, from_tensors)

    q = cirq.LineQubit(0)
    from_cirq = cirq.unitary(cirq.inverse(cirq.Circuit(cirq.H(q), cirq.T(q))))
    from_tensors = TestTStateMaker().adjoint().tensor_contract()
    np.testing.assert_allclose(from_cirq, from_tensors)


def test_test_t_state_tensor_adjoint():
    unitary = TestTStateMaker().tensor_contract()
    adj_unitary = TestTStateMaker().adjoint().tensor_contract()
    np.testing.assert_allclose(unitary.conj().T, adj_unitary)
