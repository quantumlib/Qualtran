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
from typing import Dict, List, Tuple, TYPE_CHECKING

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, ConnectionT, Signature, Soquet, SoquetT
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import Toffoli, XGate, YGate
from qualtran.bloqs.factoring import ModExp
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate, CirqQuregT
from qualtran.cirq_interop.t_complexity_protocol import t_complexity_compat
from qualtran.testing import execute_notebook

if TYPE_CHECKING:
    import quimb.tensor as qtn


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

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        from qualtran.bloqs.basic_gates import TwoBitSwap

        return TwoBitSwap().my_tensors(incoming=incoming, outgoing=outgoing)


def test_swap_two_bits_to_cirq():
    circuit = SwapTwoBitsTest().as_composite_bloq().to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        """\
x: ───×───
      │
y: ───×───""",
    )


@frozen
class SwapTest(Bloq):
    n: int

    @property
    def signature(self):
        return Signature.build(x=self.n, y=self.n)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: Soquet, y: Soquet, **kwargs
    ) -> Dict[str, SoquetT]:
        xs = bb.split(x)
        ys = bb.split(y)
        for i in range(self.n):
            xs[i], ys[i] = bb.add(SwapTwoBitsTest(), x=xs[i], y=ys[i])
        return {'x': bb.join(xs), 'y': bb.join(ys)}


@frozen
class SwapTestWithOnlyTensorData(Bloq):
    n: int

    @property
    def signature(self):
        return Signature.build(x=self.n, y=self.n)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        from qualtran.simulation.tensor._dense import _order_incoming_outgoing_indices

        inds = _order_incoming_outgoing_indices(
            self.signature, incoming=incoming, outgoing=outgoing
        )
        data = SwapTest(self.n).tensor_contract().reshape((2,) * len(inds))

        return [qtn.Tensor(data=data, inds=inds)]


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_bloq_as_cirq_gate_uses_tensor_data_for_unitary(n: int):
    unitary_one = cirq.unitary(BloqAsCirqGate(SwapTest(n)))
    unitary_two = cirq.unitary(BloqAsCirqGate(SwapTestWithOnlyTensorData(n)))
    np.testing.assert_allclose(unitary_one, unitary_two)


def test_swap():
    swap_circuit = (
        SwapTest(n=5)
        .as_composite_bloq()
        .to_cirq_circuit(
            cirq_quregs={'x': cirq.LineQubit.range(5), 'y': cirq.LineQubit.range(100, 105)},
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
    cirq_quregs = get_named_qubits(multi_and.signature.lefts())
    assert sorted(cirq_quregs.keys()) == ['ctrl']
    multi_and_circuit, out_quregs = multi_and.decompose_bloq().to_cirq_circuit_and_quregs(
        **cirq_quregs, qubit_manager=cirq.ops.SimpleQubitManager()
    )
    assert sorted(out_quregs.keys()) == ['ctrl', 'junk', 'target']


def test_flat_cbloq_to_cirq_circuit_minimizes_qubit_allocation():
    bloq = PrepareUniformSuperposition(n=3, cvs=(1,))
    qm = cirq.GreedyQubitManager(prefix='anc', maximize_reuse=True)
    cbloq = bloq.as_composite_bloq()
    assert len(cbloq.to_cirq_circuit(qubit_manager=qm).all_qubits()) == 3
    cbloq = bloq.decompose_bloq()
    assert len(cbloq.to_cirq_circuit(qubit_manager=qm).all_qubits()) == 5
    cbloq = bloq.decompose_bloq().flatten_once()
    assert len(cbloq.to_cirq_circuit(qubit_manager=qm).all_qubits()) == 7
    qm = cirq.GreedyQubitManager(prefix='anc', maximize_reuse=True)
    # Note: This should also be 7 but to work correctly, it relies on
    # `greedy_topological_sort` iterating on allocation nodes in insertion order.
    # `cbloq.flatten()` preserves this now because cbloq.iter_bloqnections is also
    #  updated to use `greedy_topological_sort` instead of `nx.topological_sort`.
    #  In general, we should have a more stable way to preserve this property,
    #  potentially by maintaing a sorted order in `binst.i`;
    #  xref: https://github.com/quantumlib/Qualtran/issues/1098
    cbloq = bloq.decompose_bloq().flatten()
    assert len(cbloq.to_cirq_circuit(qubit_manager=qm).all_qubits()) == 7


def test_contruct_op_from_gate():
    and_gate = And()
    in_quregs = {'ctrl': np.array([*cirq.LineQubit.range(2)]).reshape(2, 1)}
    qm = cirq.ops.SimpleQubitManager()
    # Allocates new qubits for RIGHT only registers.
    op, out_quregs = and_gate.as_cirq_op(qm, **in_quregs)
    assert len(out_quregs['target']) == 1
    assert op == and_gate.on_registers(**out_quregs)
    # Deallocates qubits for LEFT only registers.
    and_inv = And().adjoint()
    op, inv_out_quregs = and_inv.as_cirq_op(qm, **out_quregs)
    assert inv_out_quregs == in_quregs
    assert op == and_inv.on_registers(**out_quregs)


def test_construct_op_from_gate_raises():
    and_gate = And()
    qm = cirq.ops.SimpleQubitManager()
    q = [*cirq.LineQubit.range(2)]
    in_quregs: dict[str, np.ndarray] = {}
    with pytest.raises(ValueError, match='Compatible reg.*must exist'):
        _ = and_gate.as_cirq_op(qm, **in_quregs)

    in_quregs = {'ctrl': np.array(q)}
    with pytest.raises(ValueError, match='Compatible reg.*must exist'):
        _ = and_gate.as_cirq_op(qm, **in_quregs)

    in_quregs = {'ctrl': np.array(q).reshape(2, 1), 'target': np.array([cirq.q('t')])}
    with pytest.raises(ValueError, match='RIGHT register.*shouldn\'t exist in'):
        _ = and_gate.as_cirq_op(qm, **in_quregs)


def test_bloq_as_cirq_gate_left_register():
    bb = BloqBuilder()
    q = bb.allocate(1)
    q = bb.add(XGate(), q=q)
    bb.free(q)
    q = bb.allocate(1, dirty=True)
    q = bb.add(YGate(), q=q)
    bb.free(q, dirty=True)
    cbloq = bb.finalize()
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        """
_b(0): ───Y───

_c(0): ───X───
""",
    )


def test_bloq_as_cirq_gate_for_mod_exp():
    # ModExp is a good test because, similar to And gate, it has a RIGHT only register.
    # but also has a decomposition specified.
    mod_exp = ModExp.make_for_shor(4, 3)
    gate = BloqAsCirqGate(mod_exp)
    # Use Cirq's infrastructure to construct an operation and corresponding decomposition.
    quregs = get_named_qubits(gate.signature)
    op = gate.on_registers(**quregs)
    # cirq.decompose_once(op) delegates to underlying Bloq's decomposition specified in
    # `bloq.decompose_bloq()` and wraps resulting composite bloq in a Cirq op-tree. Note
    # how `BloqAsCirqGate.decompose_with_registers()` automatically takes care of mapping
    # newly allocated RIGHT registers in the decomposition to the one's specified by the user
    # when constructing the original operation (in this case, register `x`).
    circuit = cirq.Circuit(op, cirq.decompose_once(op))
    assert t_complexity_compat(circuit) == 2 * mod_exp.t_complexity()
    cirq.testing.assert_has_diagram(
        circuit,
        '''
exponent0: ───exponent─────────────────────────@─────
              │                                │
exponent1: ───exponent───────────────────@─────┼─────
              │                          │     │
exponent2: ───exponent─────────────@─────┼─────┼─────
              │                    │     │     │
exponent3: ───exponent───────@─────┼─────┼─────┼─────
              │              │     │     │     │
x0: ──────────x──────────1───*=3───*=1───*=1───*=1───
              │          │   │     │     │     │
x1: ──────────x──────────1───*=3───*=1───*=1───*=1───
''',
    )
    # Alternatively, decompose the Bloq and then convert the composite Bloq to a Cirq circuit.
    cbloq = mod_exp.decompose_bloq()
    # When converting a composite Bloq to a Cirq circuit, we only need to specify the input
    # registers.
    decomposed_circuit, out_regs = cbloq.to_cirq_circuit_and_quregs(exponent=quregs['exponent'])
    # Whereas when directly applying a cirq gate on qubits to get an operations, we need to
    # specify both input and output registers.
    circuit = cirq.Circuit(gate.on_registers(**out_regs), decomposed_circuit)
    assert t_complexity_compat(circuit) == 2 * mod_exp.t_complexity()
    # Notice the newly allocated qubits _C(0) and _C(1) for output register x.
    cirq.testing.assert_has_diagram(
        circuit,
        '''
_c(0): ───────x──────────1───*=3───*=1───*=1───*=1───
              │          │   │     │     │     │
_c(1): ───────x──────────1───*=3───*=1───*=1───*=1───
              │              │     │     │     │
exponent0: ───exponent───────┼─────┼─────┼─────@─────
              │              │     │     │
exponent1: ───exponent───────┼─────┼─────@───────────
              │              │     │
exponent2: ───exponent───────┼─────@─────────────────
              │              │
exponent3: ───exponent───────@───────────────────────''',
    )


def test_toffoli_circuit_diagram():
    q = cirq.LineQubit.range(3)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(Toffoli().on(*q)),
        """
0: ───@───
      │
1: ───@───
      │
2: ───X───
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(Toffoli().on(*q)),
        """
0: ---@---
      |
1: ---@---
      |
2: ---X---
""",
        use_unicode_characters=False,
    )


@pytest.mark.notebook
def test_notebook():
    execute_notebook('cirq_interop')
