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
from typing import List

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
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid] # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        a = quregs['a']
        b = quregs['b']
        binary_repr = list(iter_bits(self.integer, self.size))
        op_tree: List[cirq.Operation] = []
        for i in range(self.size):
            if binary_repr[i] == 1:
                op_tree.append(cirq.X(a[i]))
        op_tree.append(OneHotEncoding(binary_bitsize=self.size).on_registers(a=a, b=b))
        return op_tree


@pytest.mark.parametrize('integer', list(range(8)))
def test_one_hot_encoding(integer):
    # Tests that the second register has a 1 in the 'integer' index and zeroes elsewhere.
    # For example, if integer=4, then second register should a 1 in the 4th index and zeroes else.
    bitsize = 3
    gate = OneHotEncodingTest(integer, bitsize)
    qubits = cirq.LineQubit.range(bitsize + 2**bitsize)
    op = gate.on_registers(a=qubits[:bitsize], b=qubits[bitsize:])
    circuit0 = cirq.Circuit(op)
    initial_state = [0] * (bitsize + 2**bitsize)
    final_state = [0] * (bitsize + 2**bitsize)
    final_state[:bitsize] = list(iter_bits(integer, bitsize))
    final_state[bitsize + integer] = 1
    assert_circuit_inp_out_cirqsim(circuit0, qubits, initial_state, final_state)


@pytest.mark.parametrize('integer', list(range(8)))
def test_one_hot_encoding_classical(integer):
    bitsize = 3
    gate = OneHotEncoding(bitsize)
    vals = gate.call_classically(a=integer, b=0)
    assert vals == (integer, 2**integer)
