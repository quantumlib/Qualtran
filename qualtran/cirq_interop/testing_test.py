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

from qualtran import QBit, Register, Side, Signature
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.cirq_interop import testing
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


def test_assert_circuit_inp_out_cirqsim():
    qubits = cirq.LineQubit.range(4)
    initial_state = [0, 1, 0, 0]
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    final_state = [0, 1, 0, 1]

    testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)

    final_state = [0, 0, 0, 1]
    with pytest.raises(AssertionError):
        testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)


def test_gate_helper():
    g = testing.GateHelper(MultiAnd(cvs=(1, 0, 1, 0)))
    assert g.gate == MultiAnd(cvs=(1, 0, 1, 0))
    assert g.r == Signature(
        [
            Register('ctrl', QBit(), shape=4),
            Register('junk', QBit(), shape=2, side=Side.RIGHT),
            Register('target', QBit(), side=Side.RIGHT),
        ]
    )
    expected_quregs = {
        'ctrl': np.array([[cirq.q(f'ctrl[{i}]')] for i in range(4)]),
        'junk': np.array([[cirq.q(f'junk[{i}]')] for i in range(2)]),
        'target': [cirq.NamedQubit('target')],
    }
    for key in expected_quregs:
        assert np.array_equal(g.quregs[key], expected_quregs[key])
    assert g.operation.qubits == tuple(g.all_qubits)
    assert len(g.circuit) == 1


class DoesNotDecompose(cirq.Operation):
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1, clifford=2, rotations=3)

    @property
    def qubits(self):
        return []

    def with_qubits(self, _):
        pass


class InconsistentDecompostion(cirq.Operation):
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(rotations=1)

    def _decompose_(self) -> cirq.OP_TREE:
        yield cirq.X(self.qubits[0])

    @property
    def qubits(self):
        return tuple(cirq.LineQubit(3).range(3))

    def with_qubits(self, _):
        pass


def test_assert_decompose_is_consistent_with_t_complexity():
    testing.assert_decompose_is_consistent_with_t_complexity(And())


def test_assert_decompose_is_consistent_with_t_complexity_raises():
    with pytest.raises(AssertionError):
        testing.assert_decompose_is_consistent_with_t_complexity(InconsistentDecompostion())
