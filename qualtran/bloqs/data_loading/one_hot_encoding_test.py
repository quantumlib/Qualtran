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
import attrs
import cirq
import pytest
from attr import field
from numpy._typing import NDArray

from qualtran import GateWithRegisters, QUInt, Signature
from qualtran.bloqs.data_loading.one_hot_encoding import OneHotEncoding
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim


@attrs.frozen
class OneHotEncodingTest(GateWithRegisters):
    integer: int = field()
    size: int = field()

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(a=QUInt(self.size), b=QUInt(2**self.size))

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        a = quregs['a']
        b = quregs['b']
        binary_repr = list(iter_bits(self.integer, self.size))[::-1]
        for i in range(self.size):
            if binary_repr[i] == 1:
                yield cirq.X(a[i])
        yield OneHotEncoding(binary_reg_size=self.size).on_registers(a=a, b=b)


@pytest.mark.parametrize('integer', list(range(8)))
def test_one_hot_encoding(integer):
    gate = OneHotEncodingTest(integer, 3)
    qubits = cirq.LineQubit.range(3 + 2**3)
    op = gate.on_registers(a=qubits[:3], b=qubits[3:])
    circuit0 = cirq.Circuit(op)
    initial_state = [0] * (3 + 2**3)
    final_state = [0] * (3 + 2**3)
    final_state[:3] = list(iter_bits(integer, 3))[::-1]
    final_state[3 + integer] = 1
    assert_circuit_inp_out_cirqsim(circuit0, qubits, initial_state, final_state)
