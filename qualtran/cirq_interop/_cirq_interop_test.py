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
from typing import Dict, Tuple

import cirq
import cirq_ft
import numpy as np
import pytest
import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, CompositeBloq, Side, Signature, Soquet, SoquetT
from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.bloqs.basic_gates import XGate
from qualtran.cirq_interop import (
    cirq_optree_to_cbloq,
    CirqGateAsBloq,
    CirqQuregT,
    decompose_from_cirq_op,
)
from qualtran.testing import execute_notebook


def test_cirq_gate():
    x = CirqGateAsBloq(cirq.X)
    rx = CirqGateAsBloq(cirq.Rx(rads=0.123 * np.pi))
    toffoli = CirqGateAsBloq(cirq.TOFFOLI)

    for b in [x, rx, toffoli]:
        assert len(b.signature) == 1
        assert b.signature[0].side == Side.THRU

    assert x.signature[0].shape == (1,)
    assert toffoli.signature[0].shape == (3,)

    assert str(x) == 'CirqGateAsBloq(gate=cirq.X)'
    assert x.pretty_name() == 'cirq.X'
    assert x.short_name() == 'cirq.X'

    assert rx.pretty_name() == 'cirq.Rx(0.123π)'
    assert rx.short_name() == 'cirq.Rx'

    assert toffoli.pretty_name() == 'cirq.TOFFOLI'
    assert toffoli.short_name() == 'cirq.TOFFOLI'


def test_cirq_circuit_to_cbloq():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)
    cbloq = cirq_optree_to_cbloq(circuit)

    bloq_unitary = cbloq.tensor_contract()
    cirq_unitary = circuit.unitary(qubits)
    np.testing.assert_allclose(cirq_unitary, bloq_unitary, atol=1e-8)


def test_cbloq_to_cirq_circuit():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)
    cbloq = cirq_optree_to_cbloq(circuit)

    # important! we lose moment structure
    circuit = cirq.Circuit(circuit.all_operations())

    # Note: a 1d `shape` bloq register is actually two-dimensional in cirq-world
    # because of the implicit `bitsize` dimension (which must be explicit in cirq-world).
    # CirqGate has registers of bitsize=1 and shape=(n,); hence the list transpose below.
    circuit2, _ = cbloq.to_cirq_circuit(
        **{'qubits': [[q] for q in qubits]}, qubit_manager=cirq.ops.SimpleQubitManager()
    )

    assert circuit == circuit2


@frozen
class SwapTwoBitsTest(Bloq):
    @property
    def signature(self):
        return Signature.build(x=1, y=1)

    def as_cirq_op(
        self, qubit_manager: cirq.QubitManager, x: CirqQuregT, y: CirqQuregT
    ) -> Tuple[cirq.Operation, Dict[str, CirqQuregT]]:
        (x,) = x
        (y,) = y
        return cirq.SWAP(x, y), {'x': np.array([x]), 'y': np.array([y])}


def test_swap_two_bits_to_cirq():
    circuit, out_quregs = (
        SwapTwoBitsTest()
        .as_composite_bloq()
        .to_cirq_circuit(
            x=[cirq.NamedQubit('q1')],
            y=[cirq.NamedQubit('q2')],
            qubit_manager=cirq.ops.SimpleQubitManager(),
        )
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """\
q1: ───×───
       │
q2: ───×───""",
    )


@frozen
class SwapTest(Bloq):
    n: int

    @property
    def signature(self):
        return Signature.build(x=self.n, y=self.n)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: Soquet, y: Soquet
    ) -> Dict[str, SoquetT]:
        xs = bb.split(x)
        ys = bb.split(y)
        for i in range(self.n):
            xs[i], ys[i] = bb.add(SwapTwoBitsTest(), x=xs[i], y=ys[i])
        return {'x': bb.join(xs), 'y': bb.join(ys)}


def test_swap():
    swap_circuit, _ = (
        SwapTest(n=5)
        .as_composite_bloq()
        .to_cirq_circuit(
            x=cirq.LineQubit.range(5),
            y=cirq.LineQubit.range(100, 105),
            qubit_manager=cirq.ops.SimpleQubitManager(),
        )
    )
    op = next(swap_circuit.all_operations())
    swap_decomp_circuit = cirq.Circuit(cirq.decompose_once(op))

    should_be = cirq.Circuit(
        [
            cirq.Moment(
                cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(100)),
                cirq.SWAP(cirq.LineQubit(1), cirq.LineQubit(101)),
                cirq.SWAP(cirq.LineQubit(2), cirq.LineQubit(102)),
                cirq.SWAP(cirq.LineQubit(3), cirq.LineQubit(103)),
                cirq.SWAP(cirq.LineQubit(4), cirq.LineQubit(104)),
            )
        ]
    )
    assert swap_decomp_circuit == should_be


def test_multi_and_allocates():
    multi_and = MultiAnd(cvs=(1, 1, 1, 1))
    cirq_quregs = multi_and.signature.get_cirq_quregs()
    assert sorted(cirq_quregs.keys()) == ['ctrl']
    multi_and_circuit, out_quregs = multi_and.decompose_bloq().to_cirq_circuit(
        **cirq_quregs, qubit_manager=cirq.ops.SimpleQubitManager()
    )
    assert sorted(out_quregs.keys()) == ['ctrl', 'junk', 'target']


def test_bloq_as_cirq_gate_left_register():
    bb = BloqBuilder()
    q = bb.allocate(1)
    q = bb.add(XGate(), q=q)
    bb.free(q)
    cbloq = bb.finalize()
    circuit, _ = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, """_c(0): ───alloc───X───free───""")


@frozen
class TestCNOT(Bloq):
    @property
    def signature(self) -> Signature:
        return Signature.build(control=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self)

    def as_cirq_op(
        self, qubit_manager: cirq.QubitManager, **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        (control,) = cirq_quregs['control']
        (target,) = cirq_quregs['target']
        return cirq.CNOT(control, target), cirq_quregs


@frozen
class TestCNOTSymbolic(TestCNOT):
    @property
    def signature(self) -> Signature:
        c, t = sympy.Symbol('c'), sympy.Symbol('t')
        return Signature.build(control=c, target=t)


def test_bloq_decompose_from_cirq_op():
    tb = TestCNOT()
    assert len(tb.signature) == 2
    ctrl, trg = tb.signature
    assert ctrl.bitsize == 1
    assert ctrl.side == Side.THRU
    assert tb.pretty_name() == 'TestCNOT'

    cirq_quregs = tb.signature.get_cirq_quregs()
    circuit, _ = tb.decompose_bloq().to_cirq_circuit(**cirq_quregs)
    assert circuit == cirq.Circuit(cirq.CNOT(*cirq_quregs['control'], *cirq_quregs['target']))
    assert tb.t_complexity() == cirq_ft.TComplexity(clifford=1)

    with pytest.raises(NotImplementedError):
        TestCNOTSymbolic().decompose_bloq()


def test_notebook():
    execute_notebook('cirq_interop')
