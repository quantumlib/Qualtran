from functools import cached_property
from typing import Sequence

import cirq
import pytest
from attrs import frozen

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq


@frozen
class TestBloq(Bloq):
    @cached_property
    def registers(self) -> Registers:
        return Registers.build(control=1, target=1)

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

    with pytest.raises(NotImplementedError):
        tb.decompose_bloq()
