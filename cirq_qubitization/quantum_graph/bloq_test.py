from functools import cached_property
from typing import Sequence

import cirq

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq


class TestBloq(Bloq):
    @cached_property
    def registers(self) -> Registers:
        return Registers.build(control=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("A leaf bloq")

    def on_registers(
        self, control: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.Operation:
        (control,) = control
        (target,) = target
        return cirq.CNOT(control, target)


def test_bloq():
    tb = TestBloq()
    assert len(tb.registers) == 2
    assert tb.registers['control'].bitsize == 1
    assert tb.pretty_name() == 'TestBloq'

    quregs = tb.registers.get_named_qubits()
    circuit = cirq.Circuit(tb.on_registers(**quregs))
    assert circuit == cirq.Circuit(cirq.CNOT(cirq.NamedQubit('control'), cirq.NamedQubit('target')))
