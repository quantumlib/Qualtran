from typing import Sequence

import cirq
import pytest

from cirq_qubitization.cirq_infra.gate_with_registers import (
    GateWithRegisters,
    Register,
    Registers,
    SelectionRegisters,
)
from cirq_qubitization.jupyter_tools import execute_notebook


def test_register():
    r = Register("my_reg", 5)
    assert r.bitsize == 5


def test_registers():
    r1 = Register("r1", 5)
    r2 = Register("r2", 2)
    r3 = Register("r3", 1)
    regs = Registers([r1, r2, r3])
    assert len(regs) == 3

    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3

    assert regs[0:1] == Registers([r1])
    assert regs[0:2] == Registers([r1, r2])
    assert regs[1:3] == Registers([r2, r3])

    assert regs["r1"] == r1
    assert regs["r2"] == r2
    assert regs["r3"] == r3

    assert list(regs) == [r1, r2, r3]

    qubits = cirq.LineQubit.range(8)
    qregs = regs.split_qubits(qubits)
    assert qregs["r1"] == cirq.LineQubit.range(5)
    assert qregs["r2"] == cirq.LineQubit.range(5, 5 + 2)
    assert qregs["r3"] == [cirq.LineQubit(7)]

    qubits = qubits[::-1]
    merged_qregs = regs.merge_qubits(r1=qubits[:5], r2=qubits[5:7], r3=qubits[-1])
    assert merged_qregs == qubits

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }
    assert regs.get_named_qubits() == expected_named_qubits
    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [q for v in Registers(reg_order).get_named_qubits().values() for q in v]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    reg = SelectionRegisters.build(x=(n, N), y=(m, M))
    assert reg.iteration_lengths == (N, M)
    for x in range(N):
        for y in range(M):
            assert reg.to_flat_idx(x, y) == x * M + y

    assert reg.total_iteration_size == N * M


def test_registers_getitem_raises():
    g = Registers.build(a=4, b=3, c=2)
    with pytest.raises(IndexError, match="must be of the type"):
        _ = g[2.5]


def test_registers_build():
    regs1 = Registers([Register("r1", 5), Register("r2", 2)])
    regs2 = Registers.build(r1=5, r2=2)
    assert regs1 == regs2


class _TestGate(GateWithRegisters):
    @property
    def registers(self) -> Registers:
        r1 = Register("r1", 5)
        r2 = Register("r2", 2)
        r3 = Register("r3", 1)
        regs = Registers([r1, r2, r3])
        return regs

    def decompose_from_registers(
        self, r1: Sequence[cirq.Qid], r2: Sequence[cirq.Qid], r3: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield cirq.H.on_each(r1)
        yield cirq.X.on_each(r2)
        yield cirq.X.on_each(r3)


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
