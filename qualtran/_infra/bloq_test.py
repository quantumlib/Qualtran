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

from functools import cached_property
from typing import Dict, Tuple

import cirq
import pytest
from attrs import frozen

from qualtran import Bloq, CompositeBloq, Side, Signature
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.cirq_interop import CirqQuregT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.testing import execute_notebook


@frozen
class TestCNOT(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control=1, target=1)

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
    assert len(tb.signature) == 2
    ctrl, trg = tb.signature
    assert ctrl.bitsize == 1
    assert ctrl.side == Side.THRU
    assert tb.pretty_name() == 'TestCNOT'

    quregs = get_named_qubits(tb.signature.lefts())
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


def test_notebook():
    execute_notebook('Bloqs-Tutorial')
