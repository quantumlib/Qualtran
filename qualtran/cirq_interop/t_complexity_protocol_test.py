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
from typing import Set

import cirq
import pytest

from qualtran import Bloq, GateWithRegisters, Signature
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import execute_notebook


class SupportTComplexity:
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1)


class DoesNotSupportTComplexity:
    ...


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


class SupportsTComplexityBloqViaBuildCallGraph(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(SupportsTComplexityGateWithRegisters(), 5)}


def test_t_complexity_for_bloq_via_build_call_graph():
    bloq = SupportsTComplexityBloqViaBuildCallGraph()
    assert t_complexity(bloq) == TComplexity(t=5, clifford=10)


def test_t_complexity_for_bloq_does_not_support():
    with pytest.raises(TypeError):
        _ = t_complexity(DoesNotSupportTComplexityBloq())
    assert t_complexity(DoesNotSupportTComplexityBloq(), True) == None


def test_t_complexity():
    with pytest.raises(TypeError):
        _ = t_complexity(DoesNotSupportTComplexity())

    with pytest.raises(TypeError):
        _ = t_complexity(DoesNotSupportTComplexityGate())

    assert t_complexity(DoesNotSupportTComplexity(), fail_quietly=True) is None
    assert t_complexity([DoesNotSupportTComplexity()], fail_quietly=True) is None
    assert t_complexity(DoesNotSupportTComplexityGate(), fail_quietly=True) is None

    assert t_complexity(SupportTComplexity()) == TComplexity(t=1)
    assert t_complexity(SupportTComplexityGate().on(cirq.q('t'))) == TComplexity(t=1)

    g = GateHelper(SupportsTComplexityGateWithRegisters())
    assert g.gate._decompose_with_context_(g.operation.qubits) is NotImplemented
    assert t_complexity(g.gate) == TComplexity(t=1, clifford=2)
    assert t_complexity(g.operation) == TComplexity(t=1, clifford=2)

    assert t_complexity([cirq.T, cirq.X]) == TComplexity(t=1, clifford=1)

    q = cirq.NamedQubit('q')
    assert t_complexity([cirq.T(q), cirq.X(q)]) == TComplexity(t=1, clifford=1)


def test_gates():
    # T gate and its adjoint
    assert t_complexity(cirq.T) == TComplexity(t=1)
    assert t_complexity(cirq.T**-1) == TComplexity(t=1)

    assert t_complexity(cirq.H) == TComplexity(clifford=1)  # Hadamard
    assert t_complexity(cirq.CNOT) == TComplexity(clifford=1)  # CNOT
    assert t_complexity(cirq.S) == TComplexity(clifford=1)  # S
    assert t_complexity(cirq.S**-1) == TComplexity(clifford=1)  # S†

    # Pauli operators are clifford
    assert t_complexity(cirq.X) == TComplexity(clifford=1)
    assert t_complexity(cirq.Y) == TComplexity(clifford=1)
    assert t_complexity(cirq.Z) == TComplexity(clifford=1)

    # Rotation about X, Y, and Z axes
    assert t_complexity(cirq.Rx(rads=2)) == TComplexity(rotations=1)
    assert t_complexity(cirq.Ry(rads=2)) == TComplexity(rotations=1)
    assert t_complexity(cirq.Rz(rads=2)) == TComplexity(rotations=1)

    # clifford+T
    assert t_complexity(And()) == TComplexity(t=4, clifford=9)
    assert t_complexity(And() ** -1) == TComplexity(clifford=4)

    assert t_complexity(cirq.FREDKIN) == TComplexity(t=7, clifford=10)


def test_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity(cirq.T(q)) == TComplexity(t=1)

    gate = And()
    op = gate.on_registers(**get_named_qubits(gate.signature))
    assert t_complexity(op) == TComplexity(t=4, clifford=9)

    gate = And() ** -1
    op = gate.on_registers(**get_named_qubits(gate.signature))
    assert t_complexity(op) == TComplexity(clifford=4)


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
    assert t_complexity(circuit) == TComplexity(clifford=2, rotations=3, t=1)

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert t_complexity(circuit) == TComplexity(clifford=1, rotations=1, t=1)


def test_circuit_operations():
    q = cirq.NamedQubit('q')
    circuit = cirq.FrozenCircuit(
        cirq.T(q), cirq.X(q) ** 0.5, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m')
    )
    assert t_complexity(cirq.CircuitOperation(circuit)) == TComplexity(clifford=2, rotations=1, t=1)
    assert t_complexity(cirq.CircuitOperation(circuit, repetitions=10)) == TComplexity(
        clifford=20, rotations=10, t=10
    )

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert t_complexity(cirq.CircuitOperation(circuit)) == TComplexity(clifford=1, rotations=1, t=1)
    assert t_complexity(cirq.CircuitOperation(circuit, repetitions=3)) == TComplexity(
        clifford=3, rotations=3, t=3
    )


def test_classically_controlled_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity(cirq.X(q).with_classical_controls('c')) == TComplexity(clifford=1)
    assert t_complexity(cirq.Rx(rads=0.1)(q).with_classical_controls('c')) == TComplexity(
        rotations=1
    )
    assert t_complexity(cirq.T(q).with_classical_controls('c')) == TComplexity(t=1)


def test_tagged_operations():
    q = cirq.NamedQubit('q')
    assert t_complexity(cirq.X(q).with_tags('tag1')) == TComplexity(clifford=1)
    assert t_complexity(cirq.T(q).with_tags('tage1')) == TComplexity(t=1)
    assert t_complexity(cirq.Ry(rads=0.1)(q).with_tags('tag1', 'tag2')) == TComplexity(rotations=1)


def test_cache_clear():
    class IsCachable(cirq.Operation):
        def __init__(self) -> None:
            super().__init__()
            self.num_calls = 0
            self._gate = cirq.X

        def _t_complexity_(self) -> TComplexity:
            self.num_calls += 1
            return TComplexity()

        @property
        def qubits(self):
            return [cirq.LineQubit(3)]  # pragma: no cover

        def with_qubits(self, _):
            ...

        @property
        def gate(self):
            return self._gate

    assert t_complexity(cirq.X) == TComplexity(clifford=1)
    # Using a global cache will result in a failure of this test since `cirq.X` has
    # `T-complexity(clifford=1)` but we explicitly return `TComplexity()` for IsCachable
    # operation; for which the hash would be equivalent to the hash of it's subgate i.e. `cirq.X`.
    t_complexity.cache_clear()
    op = IsCachable()
    assert t_complexity([op, op]) == TComplexity()
    assert op.num_calls == 1
    t_complexity.cache_clear()


@pytest.mark.notebook
def test_notebook():
    execute_notebook('t_complexity')
