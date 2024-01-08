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

from typing import Dict

import cirq

from qualtran import GateWithRegisters, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import XGate, YGate, ZGate
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


class BloqWithDecompose(GateWithRegisters):
    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('l', 1, side=Side.LEFT),
                Register('t', 1, side=Side.THRU),
                Register('r', 1, side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', l: 'SoquetT', t: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        l = bb.add(XGate(), q=l)
        bb.free(l)
        t = bb.add(YGate(), q=t)
        r = bb.allocate(1)
        r = bb.add(ZGate(), q=r)
        return {'t': t, 'r': r}


def test_gate_with_registers_decompose_from_context_auto_generated():
    gate = BloqWithDecompose()
    op = gate.on(cirq.q('l'), cirq.q('t'), cirq.q('r'))
    circuit = cirq.Circuit(op, cirq.decompose(op))
    cirq.testing.assert_has_diagram(
        circuit,
        """
l: ───BloqWithDecompose───X──────────Free───
      │
r: ───r───────────────────Allocate───Z──────
      │
t: ───t───────────────────Y─────────────────
""",
    )


def test_notebook():
    execute_notebook('gate_with_registers')
