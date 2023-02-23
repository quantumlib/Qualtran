from functools import cached_property
from typing import Sequence

import cirq
import pytest
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, Side


@frozen
class TestBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(control=1, target=1)

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
    assert tb.registers['control'].side == Side.THRU
    assert tb.pretty_name() == 'TestBloq'

    quregs = tb.registers.get_named_qubits()
    circuit = cirq.Circuit(tb.on_registers(**quregs))
    assert circuit == cirq.Circuit(cirq.CNOT(cirq.NamedQubit('control'), cirq.NamedQubit('target')))

    with pytest.raises(NotImplementedError):
        tb.decompose_bloq()


def test_as_composite_bloq():
    tb = TestBloq()
    cb = tb.as_composite_bloq()
    assert isinstance(cb, CompositeBloq)
    bloqs = list(cb.bloq_instances)
    assert len(bloqs) == 1
    assert bloqs[0].bloq == tb

    cb2 = cb.as_composite_bloq()
    assert cb is cb2
