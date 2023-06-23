from functools import cached_property
from typing import Dict, Tuple

import cirq
import pytest
from attrs import frozen
from cirq_ft import TComplexity

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, Side


@frozen
class TestCNOT(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(control=1, target=1)

    def as_cirq_op(
        self, qubit_manager: cirq.QubitManager, **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        (control,) = cirq_quregs['control']
        (target,) = cirq_quregs['target']
        return cirq.CNOT(control, target), cirq_quregs

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(clifford=1)


def test_bloq():
    tb = TestCNOT()
    assert len(tb.registers) == 2
    assert tb.registers['control'].bitsize == 1
    assert tb.registers['control'].side == Side.THRU
    assert tb.pretty_name() == 'TestCNOT'

    quregs = tb.registers.get_cirq_quregs()
    op, _ = tb.as_cirq_op(cirq.ops.SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.CNOT(cirq.NamedQubit('control'), cirq.NamedQubit('target')))

    with pytest.raises(NotImplementedError):
        tb.decompose_bloq()


def test_as_composite_bloq():
    tb = TestCNOT()
    assert not tb.supports_decompose_bloq()
    cb = tb.as_composite_bloq()
    assert isinstance(cb, CompositeBloq)
    bloqs = list(cb.bloq_instances)
    assert len(bloqs) == 1
    assert bloqs[0].bloq == tb

    cb2 = cb.as_composite_bloq()
    assert cb is cb2


def test_t_complexity():
    assert TestCNOT().t_complexity() == TComplexity(clifford=1)
