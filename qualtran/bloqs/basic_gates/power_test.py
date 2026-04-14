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
import subprocess

import cirq
import pytest

from qualtran._infra.gate_with_registers import GateWithRegisters
from qualtran.bloqs.basic_gates import Power
from qualtran.bloqs.for_testing import TestAtom, TestMultiRegister
from qualtran.bloqs.for_testing.atom import TestGWRAtom


def test_power():
    with pytest.raises(ValueError, match="THRU"):
        from qualtran.bloqs.mcmt import And

        _ = Power(And(), 2)

    bloq = TestMultiRegister()
    with pytest.raises(ValueError, match="positive"):
        _ = Power(bloq, -2)

    bloq_raised_to_power = Power(bloq, 10)
    assert bloq_raised_to_power.signature == bloq.signature
    cbloq = bloq_raised_to_power.decompose_bloq()
    assert [binst.bloq for binst, _, _ in cbloq.iter_bloqnections()] == [bloq] * 10


def test_power_of_power():
    bloq = TestAtom()
    assert Power(bloq, 6) == Power(bloq, 2) ** 3

    gate = TestGWRAtom()
    assert gate**-3 == gate.adjoint() ** 3
    assert gate**6 == (gate**2) ** 3
    assert gate**6 == (gate**-2) ** -3


def test_power_circuit_diagram():
    def to_cirq_circuit(bloq: GateWithRegisters) -> cirq.Circuit:
        op = bloq.on(*cirq.LineQubit.range(bloq.num_qubits()))
        return cirq.Circuit(op)

    power_atom = Power(TestGWRAtom(), 4)

    cirq.testing.assert_has_diagram(to_cirq_circuit(power_atom), '0: ───TestGWRAtom^4───')

    power_multi_reg = Power(TestMultiRegister(), 4)
    cirq.testing.assert_has_diagram(
        to_cirq_circuit(power_multi_reg),
        '''
0: ────Power^4───
       │
1: ────yy^4──────
       │
2: ────yy^4──────
       │
3: ────yy^4──────
       │
4: ────yy^4──────
       │
5: ────yy^4──────
       │
6: ────yy^4──────
       │
7: ────yy^4──────
       │
8: ────yy^4──────
       │
9: ────zz^4──────
       │
10: ───zz^4──────
       │
11: ───zz^4──────''',
    )


@pytest.mark.slow
def test_no_circular_import():
    subprocess.check_call(['python', '-c', 'from qualtran.bloqs.basic_gates import power'])


def test_power_wire_symbol():
    from qualtran import Signature
    from qualtran.drawing import LarrowTextBox, RarrowTextBox, Text, TextBox

    class MockBloq(GateWithRegisters):
        @property
        def signature(self):
            return Signature([])

        def wire_symbol(self, reg, idx=tuple()):
            if reg is None:
                return Text("Mock")
            if reg.name == 'a':
                return Text("A")
            if reg.name == 'b':
                return TextBox("B")
            if reg.name == 'c':
                return LarrowTextBox("C")
            if reg.name == 'd':
                return RarrowTextBox("D")

    bloq = MockBloq()
    power_bloq = Power(bloq, 3)

    assert power_bloq.wire_symbol(None) == Text("Mock**3")

    from qualtran import QUInt, Register

    reg_a = Register('a', QUInt(1))
    reg_b = Register('b', QUInt(1))
    reg_c = Register('c', QUInt(1))
    reg_d = Register('d', QUInt(1))

    assert power_bloq.wire_symbol(reg_a) == Text("A**3")
    assert power_bloq.wire_symbol(reg_b) == TextBox("B**3")
    assert power_bloq.wire_symbol(reg_c) == LarrowTextBox("C**3")
    assert power_bloq.wire_symbol(reg_d) == RarrowTextBox("D**3")
