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

from qualtran import GateWithRegisters, Register, Signature
from qualtran.testing import execute_notebook


class _TestGate(GateWithRegisters):
    @property
    def signature(self) -> Signature:
        r1 = Register("r1", 5)
        r2 = Register("r2", 2)
        r3 = Register("r3", 1)
        regs = Signature([r1, r2, r3])
        return regs

    def decompose_from_registers(self, *, context, **quregs) -> cirq.OP_TREE:
        yield cirq.H.on_each(quregs['r1'])
        yield cirq.X.on_each(quregs['r2'])
        yield cirq.X.on_each(quregs['r3'])


def test_gate_with_registers():
    tg = _TestGate()
    assert tg._num_qubits_() == 8
    qubits = cirq.LineQubit.range(8)
    circ = cirq.Circuit(tg._decompose_(qubits))
    assert circ.operation_at(cirq.LineQubit(3), 0).gate == cirq.H

    op1 = tg.on_registers(r1=qubits[:5], r2=qubits[6:], r3=qubits[5])
    op2 = tg.on(*qubits[:5], *qubits[6:], qubits[5])
    assert op1 == op2


def test_notebook():
    execute_notebook('gate_with_registers')
