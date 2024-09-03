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
from typing import Set, TYPE_CHECKING

import cirq
import pytest
from attrs import frozen

from qualtran import Bloq, DecomposeNotImplementedError, GateWithRegisters, Signature
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import CHadamard, GlobalPhase
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.cirq_interop.t_complexity_protocol import (
    _from_directly_countable_cirq,
    t_complexity,
    t_complexity_compat,
    TComplexity,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import execute_notebook

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


class DoesNotSupportTComplexity:
    ...


@frozen
class SupportsTComplexityGateWithRegisters(GateWithRegisters):
    @property
    def signature(self) -> Signature:
        return Signature.build(s=1, t=2)

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1, clifford=2)


class SupportTComplexityGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1)


class DoesNotSupportTComplexityGate(cirq.Gate):
    def _num_qubits_(self):
        return 1


class DoesNotSupportTComplexityBloq(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)


@frozen
class SupportsTComplexityBloqViaBuildCallGraph(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(SupportsTComplexityGateWithRegisters(), 5)}


def test_t_complexity_for_bloq_via_build_call_graph():
    bloq = SupportsTComplexityBloqViaBuildCallGraph()
    _, sigma = bloq.call_graph(max_depth=1)
    assert sigma != {}
    assert t_complexity(bloq) == TComplexity(t=5, clifford=10)


def test_t_complexity_for_bloq_does_not_support():
    with pytest.raises(DecomposeNotImplementedError):
        _ = t_complexity(DoesNotSupportTComplexityBloq())


def test_t_complexity_compat():
    with pytest.raises(TypeError):
        _ = t_complexity_compat(DoesNotSupportTComplexity())
    with pytest.raises(TypeError):
        t_complexity_compat([DoesNotSupportTComplexity()])
    with pytest.raises(TypeError):
        _ = t_complexity_compat(DoesNotSupportTComplexityGate())

    assert t_complexity_compat(SupportTComplexityGate().on(cirq.q('t'))) == TComplexity(t=1)

    g = GateHelper(SupportsTComplexityGateWithRegisters())
    assert g.gate._decompose_with_context_(g.operation.qubits) is NotImplemented  # type: ignore[attr-defined]
    assert t_complexity(g.gate) == TComplexity(t=1, clifford=2)
    assert t_complexity_compat(g.operation) == TComplexity(t=1, clifford=2)

    assert t_complexity_compat([cirq.T, cirq.X]) == TComplexity(t=1, clifford=1)

    q = cirq.NamedQubit('q')
    assert t_complexity_compat([cirq.T(q), cirq.X(q)]) == TComplexity(t=1, clifford=1)


def test_gates():
    # T gate and its adjoint
    assert _from_directly_countable_cirq(cirq.T) == TComplexity(t=1)
    assert _from_directly_countable_cirq(cirq.T**-1) == TComplexity(t=1)

    assert _from_directly_countable_cirq(cirq.H) == TComplexity(clifford=1)  # Hadamard
    assert _from_directly_countable_cirq(cirq.CNOT) == TComplexity(clifford=1)  # CNOT
    assert _from_directly_countable_cirq(cirq.S) == TComplexity(clifford=1)  # S
    assert _from_directly_countable_cirq(cirq.S**-1) == TComplexity(clifford=1)  # S†

    # Pauli operators are clifford
    assert _from_directly_countable_cirq(cirq.X) == TComplexity(clifford=1)
    assert _from_directly_countable_cirq(cirq.Y) == TComplexity(clifford=1)
    assert _from_directly_countable_cirq(cirq.Z) == TComplexity(clifford=1)

    # Rotation about X, Y, and Z axes
    assert _from_directly_countable_cirq(cirq.Rx(rads=2)) == TComplexity(rotations=1)
    assert _from_directly_countable_cirq(cirq.Ry(rads=2)) == TComplexity(rotations=1)
    assert _from_directly_countable_cirq(cirq.Rz(rads=2)) == TComplexity(rotations=1)

    # clifford+T
    assert t_complexity(And()) == TComplexity(t=4, clifford=9)
    assert t_complexity(And() ** -1) == TComplexity(clifford=4)

    assert t_complexity_compat(cirq.FREDKIN) == TComplexity(t=7, clifford=14)
    assert t_complexity_compat(cirq.H.controlled()) == TComplexity(clifford=4, rotations=2)
    assert t_complexity(CHadamard()) == TComplexity(clifford=4, rotations=2)

    # Global phase
    assert _from_directly_countable_cirq(cirq.GlobalPhaseGate(1j)) == TComplexity()
    assert t_complexity(GlobalPhase(exponent=1)) == TComplexity()


def test_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity_compat(cirq.T(q)) == TComplexity(t=1)

    gate = And()
    op = gate.on_registers(**get_named_qubits(gate.signature))
    assert t_complexity_compat(op) == TComplexity(t=4, clifford=9)

    gate = And() ** -1
    op = gate.on_registers(**get_named_qubits(gate.signature))
    assert t_complexity_compat(op) == TComplexity(clifford=4)

    assert t_complexity_compat(cirq.global_phase_operation(1j)) == TComplexity()


def test_circuits():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        cirq.Rz(rads=0.6)(q),
        cirq.T(q),
        cirq.X(q) ** 0.5,
        cirq.Rx(rads=0.1)(q),
        cirq.Ry(rads=0.6)(q),
        cirq.measure(q, key='m'),
    )
    assert t_complexity_compat(circuit) == TComplexity(clifford=2, rotations=3, t=1)

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert t_complexity_compat(circuit) == TComplexity(clifford=1, rotations=1, t=1)


def test_circuit_operations():
    q = cirq.NamedQubit('q')
    circuit = cirq.FrozenCircuit(
        cirq.T(q), cirq.X(q) ** 0.5, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m')
    )
    assert t_complexity_compat(cirq.CircuitOperation(circuit)) == TComplexity(
        clifford=2, rotations=1, t=1
    )
    assert t_complexity_compat(cirq.CircuitOperation(circuit, repetitions=10)) == TComplexity(
        clifford=20, rotations=10, t=10
    )

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert t_complexity_compat(cirq.CircuitOperation(circuit)) == TComplexity(
        clifford=1, rotations=1, t=1
    )
    assert t_complexity_compat(cirq.CircuitOperation(circuit, repetitions=3)) == TComplexity(
        clifford=3, rotations=3, t=3
    )


def test_classically_controlled_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity_compat(cirq.X(q).with_classical_controls('c')) == TComplexity(clifford=1)
    assert t_complexity_compat(cirq.Rx(rads=0.1)(q).with_classical_controls('c')) == TComplexity(
        rotations=1
    )
    assert t_complexity_compat(cirq.T(q).with_classical_controls('c')) == TComplexity(t=1)


def test_tagged_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity_compat(cirq.X(q).with_tags('tag1')) == TComplexity(clifford=1)
    assert t_complexity_compat(cirq.T(q).with_tags('tage1')) == TComplexity(t=1)
    assert t_complexity_compat(cirq.Ry(rads=0.1)(q).with_tags('tag1', 'tag2')) == TComplexity(
        rotations=1
    )


@pytest.mark.notebook
def test_notebook():
    execute_notebook('t_complexity')
