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
import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.cirq_interop import BloqAsCirqGate, CirqQuregT
from qualtran.testing import execute_notebook


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
    cirq.testing.assert_has_diagram(circuit, """_c(0): ───Allocate───X───Free───""")


def test_bloq_as_cirq_gate_multi_dimensional_signature():
    bloq = SwapWithZero(2, 3, 4)
    cirq_quregs = bloq.signature.get_cirq_quregs()
    op = BloqAsCirqGate(bloq).on_registers(**cirq_quregs)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(op),
        '''
selection0: ──────SwapWithZero───
                  │
selection1: ──────selection──────
                  │
targets[0][0]: ───targets────────
                  │
targets[0][1]: ───targets────────
                  │
targets[0][2]: ───targets────────
                  │
targets[1][0]: ───targets────────
                  │
targets[1][1]: ───targets────────
                  │
targets[1][2]: ───targets────────
                  │
targets[2][0]: ───targets────────
                  │
targets[2][1]: ───targets────────
                  │
targets[2][2]: ───targets────────
                  │
targets[3][0]: ───targets────────
                  │
targets[3][1]: ───targets────────
                  │
targets[3][2]: ───targets────────
''',
    )
    cbloq = bloq.decompose_bloq()
    cirq.testing.assert_has_diagram(
        cbloq.to_cirq_circuit(**cirq_quregs)[0],
        '''
selection0: ──────────────────────────────@(approx)───
                                          │
selection1: ──────@(approx)───@(approx)───┼───────────
                  │           │           │
targets[0][0]: ───×(x)────────┼───────────×(x)────────
                  │           │           │
targets[0][1]: ───×(x)────────┼───────────×(x)────────
                  │           │           │
targets[0][2]: ───×(x)────────┼───────────×(x)────────
                  │           │           │
targets[1][0]: ───×(y)────────┼───────────┼───────────
                  │           │           │
targets[1][1]: ───×(y)────────┼───────────┼───────────
                  │           │           │
targets[1][2]: ───×(y)────────┼───────────┼───────────
                              │           │
targets[2][0]: ───────────────×(x)────────×(y)────────
                              │           │
targets[2][1]: ───────────────×(x)────────×(y)────────
                              │           │
targets[2][2]: ───────────────×(x)────────×(y)────────
                              │
targets[3][0]: ───────────────×(y)────────────────────
                              │
targets[3][1]: ───────────────×(y)────────────────────
                              │
targets[3][2]: ───────────────×(y)────────────────────
''',
    )


def test_notebook():
    execute_notebook('cirq_interop')
