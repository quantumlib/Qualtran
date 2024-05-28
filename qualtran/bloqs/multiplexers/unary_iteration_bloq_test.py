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

import itertools
from functools import cached_property
from typing import Iterator, List, Sequence, Set, Tuple, TYPE_CHECKING

import cirq
import pytest

from qualtran import BoundedQUInt, QAny, Register, Signature
from qualtran._infra.gate_with_registers import get_named_qubits, total_bits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.bookkeeping import Join, Split
from qualtran.bloqs.multiplexers.unary_iteration_bloq import unary_iteration, UnaryIterationGate
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting.generalizers import cirq_to_bloqs
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.resource_counting import BloqCountT


class ApplyXToLthQubit(UnaryIterationGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int, control_bitsize: int = 1):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (Register('control', QAny(self._control_bitsize)),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', BoundedQUInt(self._selection_bitsize, self._target_bitsize)),)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self._target_bitsize)),)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        selection: int,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(selection + 1)])

    def nth_operation_callgraph(self, **selection_regs_name_to_val) -> Set['BloqCountT']:
        return {(CNOT(), 1)}


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, control_bitsize",
    [(3, 5, 1), (2, 4, 2), (1, 2, 3), (3, 4, 0)],
)
def test_unary_iteration_gate(selection_bitsize, target_bitsize, control_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyXToLthQubit(selection_bitsize, target_bitsize, control_bitsize)
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= 2 * (selection_bitsize + control_bitsize) + target_bitsize - 1

    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.operation.qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        qubit_vals[g.quregs['target'][-(n + 1)]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        assert_circuit_inp_out_cirqsim(g.circuit, g.operation.qubits, initial_state, final_state)


class ApplyXToIJKthQubit(UnaryIterationGate):
    def __init__(self, target_shape: Tuple[int, int, int]):
        self._target_shape = target_shape

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Register(
                'ijk'[i],
                BoundedQUInt((self._target_shape[i] - 1).bit_length(), self._target_shape[i]),
            )
            for i in range(3)
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Signature.build(
                t1=self._target_shape[0], t2=self._target_shape[1], t3=self._target_shape[2]
            )
        )

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        i: int,
        j: int,
        k: int,
        t1: Sequence[cirq.Qid],
        t2: Sequence[cirq.Qid],
        t3: Sequence[cirq.Qid],
    ) -> Iterator[cirq.OP_TREE]:
        yield [cirq.CNOT(control, t1[i]), cirq.CNOT(control, t2[j]), cirq.CNOT(control, t3[k])]

    def nth_operation_callgraph(self, **selection_regs_name_to_val) -> Set['BloqCountT']:
        return {(CNOT(), 3)}


@pytest.mark.slow
@pytest.mark.parametrize("target_shape", [(2, 3, 2), (2, 2, 2)])
def test_multi_dimensional_unary_iteration_gate(target_shape: Tuple[int, int, int]):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyXToIJKthQubit(target_shape)
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert (
        len(g.all_qubits) <= total_bits(gate.signature) + total_bits(gate.selection_registers) - 1
    )

    max_i, max_j, max_k = target_shape
    i_len, j_len, k_len = tuple(reg.total_bits() for reg in gate.selection_registers)
    for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
        qubit_vals = {x: 0 for x in g.operation.qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update(zip(g.quregs['i'], iter_bits(i, i_len)))
        qubit_vals.update(zip(g.quregs['j'], iter_bits(j, j_len)))
        qubit_vals.update(zip(g.quregs['k'], iter_bits(k, k_len)))
        # Construct initial state
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        # Build correct statevector with selection_integer bit flipped in the target register:
        for reg_name, idx in zip(['t1', 't2', 't3'], [i, j, k]):
            qubit_vals[g.quregs[reg_name][idx]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        assert_circuit_inp_out_cirqsim(g.circuit, g.operation.qubits, initial_state, final_state)


def test_unary_iteration_loop():
    n_range, m_range = (3, 5), (6, 8)
    selection_registers = [Register('n', BoundedQUInt(3, 5)), Register('m', BoundedQUInt(3, 8))]
    selection = get_named_qubits(selection_registers)
    target = {(n, m): cirq.q(f't({n}, {m})') for n in range(*n_range) for m in range(*m_range)}
    qm = cirq.GreedyQubitManager("ancilla", maximize_reuse=True)
    circuit = cirq.Circuit()
    i_ops: List[cirq.Operation] = []
    # Build the unary iteration circuit
    for i_optree, i_ctrl, i in unary_iteration(
        n_range[0], n_range[1], i_ops, [], list(selection['n']), qm
    ):
        circuit.append(i_optree)
        j_ops: List[cirq.Operation] = []
        for j_optree, j_ctrl, j in unary_iteration(
            m_range[0], m_range[1], j_ops, [i_ctrl], list(selection['m']), qm
        ):
            circuit.append(j_optree)
            # Conditionally perform operations on target register using `j_ctrl`, `i` & `j`.
            circuit.append(cirq.CNOT(j_ctrl, target[(i, j)]))
        circuit.append(j_ops)
    circuit.append(i_ops)
    all_qubits = sorted(circuit.all_qubits())

    i_len, j_len = 3, 3
    for i, j in itertools.product(range(*n_range), range(*m_range)):
        qubit_vals = {x: 0 for x in all_qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update(zip(selection['n'], iter_bits(i, i_len)))
        qubit_vals.update(zip(selection['m'], iter_bits(j, j_len)))
        # Construct initial state
        initial_state = [qubit_vals[x] for x in all_qubits]
        # Build correct statevector with selection_integer bit flipped in the target register:
        qubit_vals[target[(i, j)]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_unary_iteration_loop_empty_range():
    qm = cirq.SimpleQubitManager()
    assert list(unary_iteration(4, 4, [], [], [cirq.q('s')], qm)) == []
    assert list(unary_iteration(4, 3, [], [], [cirq.q('s')], qm)) == []


def verify_bloq_has_consistent_build_callgraph(bloq: 'Bloq'):
    _, sigma_call = bloq.call_graph()
    _, sigma_decomp_call = bloq.decompose_bloq().call_graph(generalizer=cirq_to_bloqs)
    for key in sigma_decomp_call.keys():
        if key in sigma_call:
            assert sigma_call[key] == sigma_decomp_call[key]
        else:
            assert isinstance(key, (Split, Join))


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, control_bitsize",
    [(3, 5, 1), (2, 4, 2), (1, 2, 3), (3, 4, 0)],
)
def test_bloq_has_consistent_decomposition(selection_bitsize, target_bitsize, control_bitsize):
    bloq = ApplyXToLthQubit(selection_bitsize, target_bitsize, control_bitsize)
    assert_valid_bloq_decomposition(bloq)
    assert bloq.t_complexity().t == 4 * (target_bitsize - 2 + control_bitsize)
    verify_bloq_has_consistent_build_callgraph(bloq)


@pytest.mark.parametrize("target_shape", [(2, 3, 2), (2, 2, 2)])
def test_multi_dimensional_bloq_has_consistent_decomposition(target_shape: Tuple[int, int, int]):
    bloq = ApplyXToIJKthQubit(target_shape)
    assert_valid_bloq_decomposition(bloq)
    verify_bloq_has_consistent_build_callgraph(bloq)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('unary_iteration')
