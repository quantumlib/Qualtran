from typing import Sequence

import cirq

from cirq_qubitization.gate_with_registers import Register, Registers, GateWithRegisters


def test_register():
    r = Register("my_reg", 5)
    assert r.bitsize == 5


def test_registers():
    r1 = Register("r1", 5)
    r2 = Register("r2", 2)
    regs = Registers([r1, r2])

    assert regs.i(0) == r1
    assert regs.i(1) == r2

    assert regs["r1"] == r1
    assert regs["r2"] == r2

    assert list(regs) == [r1, r2]

    qubits = cirq.LineQubit.range(7)
    qregs = regs.split_qubits(qubits)
    assert qregs["r1"] == cirq.LineQubit.range(5)
    assert qregs["r2"] == cirq.LineQubit.range(5, 5 + 2)


class _TestGate(GateWithRegisters):
    @property
    def registers(self) -> Registers:
        r1 = Register("r1", 5)
        r2 = Register("r2", 2)
        regs = Registers([r1, r2])
        return regs

    def decompose_from_registers(
        self, r1: Sequence[cirq.Qid], r2: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield cirq.H.on_each(r1)
        yield cirq.X.on_each(r2)


def test_gate_with_registers():
    tg = _TestGate()
    assert tg._num_qubits_() == 7
    qubits = cirq.LineQubit.range(7)
    circ = cirq.Circuit(tg._decompose_(qubits))
    assert circ.operation_at(cirq.LineQubit(3), 0).gate == cirq.H
