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

import cirq
import numpy as np
import pytest

from qualtran._infra.gate_with_registers import split_qubits, total_bits
from qualtran.bloqs.basic_gates import CNOT, TGate
from qualtran.bloqs.qrom import QROM
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import t_complexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting.generalizers import cirq_to_bloqs
from qualtran.testing import (
    assert_valid_bloq_decomposition,
    assert_wire_symbols_match_expected,
    execute_notebook,
)


@pytest.mark.parametrize(
    "data,num_controls",
    [
        pytest.param(
            data,
            num_controls,
            id=f"{num_controls}-data{idx}",
            marks=pytest.mark.slow if num_controls == 2 and idx == 2 else (),
        )
        for idx, data in enumerate(
            [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
        )
        for num_controls in [0, 1, 2]
    ],
)
def test_qrom_1d(data, num_controls):
    qrom = QROM.build(*data, num_controls=num_controls)
    assert_valid_bloq_decomposition(qrom)

    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))

    assert (
        len(inverse.all_qubits())
        <= total_bits(g.r) + total_bits(qrom.selection_registers) + num_controls
    )
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()

    for selection_integer in range(len(data[0])):
        for cval in range(2):
            qubit_vals = {x: 0 for x in g.all_qubits}
            qubit_vals.update(
                zip(
                    g.quregs.get('selection', ()),
                    iter_bits(selection_integer, total_bits(qrom.selection_registers)),
                )
            )
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))

            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}_"]
                    qubit_vals.update(zip(target, iter_bits(d[selection_integer], len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]

            assert_circuit_inp_out_cirqsim(
                decomposed_circuit, g.all_qubits, initial_state, final_state
            )
            assert_circuit_inp_out_cirqsim(
                decomposed_circuit + inverse, g.all_qubits, initial_state, initial_state
            )
            assert_circuit_inp_out_cirqsim(
                decomposed_circuit + inverse, g.all_qubits, final_state, final_state
            )


def test_qrom_diagram():
    d0 = np.array([1, 2, 3])
    d1 = np.array([4, 5, 6])
    qrom = QROM.build(d0, d1)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**split_qubits(qrom.signature, q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In───────
      │
1: ───In───────
      │
2: ───QROM_0───
      │
3: ───QROM_0───
      │
4: ───QROM_1───
      │
5: ───QROM_1───
      │
6: ───QROM_1───""",
    )


def test_notebook():
    execute_notebook('qrom')


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
def test_t_complexity(data):
    qrom = QROM.build(*data)
    g = GateHelper(qrom)
    n = np.prod(qrom.data[0].shape)
    assert t_complexity(g.gate) == t_complexity(g.operation)
    assert t_complexity(g.gate).t == max(0, 4 * n - 8), n


def _assert_qrom_has_diagram(qrom: QROM, expected_diagram: str):
    gh = GateHelper(qrom)
    op = gh.operation
    context = cirq.DecompositionContext(qubit_manager=cirq.GreedyQubitManager(prefix="anc"))
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    selection = [
        *itertools.chain.from_iterable(gh.quregs[reg.name] for reg in qrom.selection_registers)
    ]
    selection = [q for q in selection if q in circuit.all_qubits()]
    anc = sorted(set(circuit.all_qubits()) - set(op.qubits))
    selection_and_anc = (selection[0],) + sum(zip(selection[1:], anc), ())
    qubit_order = cirq.QubitOrder.explicit(selection_and_anc, fallback=cirq.QubitOrder.DEFAULT)
    cirq.testing.assert_has_diagram(circuit, expected_diagram, qubit_order=qubit_order)


def test_qrom_variable_spacing():
    # Tests for variable spacing optimization applied from https://arxiv.org/abs/2007.07391
    data = [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]  # Figure 3a.
    assert t_complexity(QROM.build(data)).t == (8 - 2) * 4
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]  # Figure 3b.
    assert t_complexity(QROM.build(data)).t == (5 - 2) * 4
    data = [1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]  # Negative test: t count is not (g-2)*4
    assert t_complexity(QROM.build(data)).t == (8 - 2) * 4
    # Works as expected when multiple data arrays are to be loaded.
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    # (a) Both data sequences are identical
    assert t_complexity(QROM.build(data, data)).t == (5 - 2) * 4
    # (b) Both data sequences have identical structure, even though the elements are not same.
    assert t_complexity(QROM.build(data, 2 * np.array(data))).t == (5 - 2) * 4
    # Works as expected when multidimensional input data is to be loaded
    qrom = QROM.build(
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2],
            ]
        )
    )
    # Value to be loaded depends only the on the first bit of outer loop.
    _assert_qrom_has_diagram(
        qrom,
        r'''
selection00: ───X───@───X───@───
                    │       │
target0_0: ─────────┼───────X───
                    │
target0_1: ─────────X───────────
    ''',
    )
    # When inner loop range is not a power of 2, the inner segment tree cannot be skipped.
    qrom = QROM.build(
        np.array(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
            dtype=int,
        )
    )
    _assert_qrom_has_diagram(
        qrom,
        r'''
selection00: ───X───@─────────@───────@──────X───@─────────@───────@──────
                    │         │       │          │         │       │
selection10: ───────(0)───────┼───────@──────────(0)───────┼───────@──────
                    │         │       │          │         │       │
anc_1: ─────────────And───@───X───@───And†───────And───@───X───@───And†───
                          │       │                    │       │
target0_0: ───────────────┼───────┼────────────────────X───────X──────────
                          │       │
target0_1: ───────────────X───────X───────────────────────────────────────
        ''',
    )
    # No T-gates needed if all elements to load are identical.
    assert t_complexity(QROM.build([3, 3, 3, 3])).t == 0


def test_qrom_wire_symbols():
    qrom = QROM.build([3, 3, 3, 3])
    assert_wire_symbols_match_expected(qrom, ['In', 'data_a'])

    qrom = QROM.build([3, 3, 3, 3], [2, 2, 2, 2])
    assert_wire_symbols_match_expected(qrom, ['In', 'data_a', 'data_b'])

    qrom = QROM.build([[3, 3], [3, 3]], [[2, 2], [2, 2]], [[1, 1], [2, 2]])
    assert_wire_symbols_match_expected(qrom, ['In_i', 'In_j', 'data_a', 'data_b', 'data_c'])

    qrom = QROM.build(np.arange(27).reshape(3, 3, 3))
    assert_wire_symbols_match_expected(qrom, ['In_i', 'In_j', 'In_k', 'data_a'])


@pytest.mark.parametrize(
    "data,num_controls",
    [
        pytest.param(
            data,
            num_controls,
            id=f"{num_controls}-data{idx}",
            marks=pytest.mark.slow if num_controls == 2 and idx == 0 else (),
        )
        for idx, data in enumerate(
            [
                [np.arange(6).reshape(2, 3), 4 * np.arange(6).reshape(2, 3)],
                [np.arange(8).reshape(2, 2, 2)],
            ]
        )
        for num_controls in [0, 1, 2]
    ],
)
def test_qrom_multi_dim(data, num_controls):
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = QROM(
        data,
        selection_bitsizes=selection_bitsizes,
        target_bitsizes=target_bitsizes,
        num_controls=num_controls,
    )
    assert_valid_bloq_decomposition(qrom)

    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))

    assert (
        len(inverse.all_qubits())
        <= total_bits(g.r) + total_bits(qrom.selection_registers) + num_controls
    )
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()

    lens = tuple(reg.total_bits() for reg in qrom.selection_registers)
    for idxs in itertools.product(*[range(dim) for dim in data[0].shape]):
        qubit_vals = {x: 0 for x in g.all_qubits}
        for cval in range(2):
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))
            for isel in range(len(idxs)):
                qubit_vals.update(
                    zip(g.quregs[f'selection{isel}'], iter_bits(idxs[isel], lens[isel]))
                )
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}_"]
                    qubit_vals.update(zip(target, iter_bits(int(d[idxs]), len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]
            qubit_vals = {x: 0 for x in g.all_qubits}
            assert_circuit_inp_out_cirqsim(
                decomposed_circuit, g.all_qubits, initial_state, final_state
            )


@pytest.mark.parametrize(
    "data",
    [
        [np.arange(6, dtype=int).reshape(2, 3), 4 * np.arange(6, dtype=int).reshape(2, 3)],
        [np.arange(8, dtype=int).reshape(2, 2, 2)],
    ],
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_ndim_t_complexity(data, num_controls):
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = QROM(data, selection_bitsizes, target_bitsizes, num_controls=num_controls)
    g = GateHelper(qrom)
    n = data[0].size
    assert t_complexity(g.gate) == t_complexity(g.operation) == qrom.t_complexity()
    assert t_complexity(g.gate).t == max(0, 4 * n - 8 + 4 * num_controls)


@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_call_graph_matches_decomposition(num_controls):
    # Base case
    arr = np.arange(50)
    qrom = QROM.build(arr, num_controls=num_controls)
    _, sigma_call = qrom.call_graph(generalizer=cirq_to_bloqs)
    _, sigma_dcmp = qrom.decompose_bloq().call_graph(generalizer=cirq_to_bloqs)
    assert sigma_call[TGate()] == sigma_dcmp[TGate()]
    assert sigma_call[CNOT()] == sigma_dcmp[CNOT()]

    # Multiple Multi dimensional arrays
    arr_a = np.arange(64).reshape(8, 8)
    arr_b = 10 * np.arange(64).reshape(8, 8)
    qrom = QROM.build(arr_a, arr_b, num_controls=num_controls)
    _, sigma_call = qrom.call_graph(generalizer=cirq_to_bloqs)
    _, sigma_dcmp = qrom.decompose_bloq().call_graph(generalizer=cirq_to_bloqs)
    assert sigma_call[TGate()] == sigma_dcmp[TGate()]
    assert sigma_call[CNOT()] == sigma_dcmp[CNOT()]

    # Variable QROM case.
    arr_a = np.array([1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5])
    arr_b = 10 * arr_a
    qrom = QROM.build(arr_a, arr_b, num_controls=num_controls)
    _, sigma_call = qrom.call_graph(generalizer=cirq_to_bloqs)
    _, sigma_dcmp = qrom.decompose_bloq().call_graph(generalizer=cirq_to_bloqs)
    assert sigma_call[TGate()] == sigma_dcmp[TGate()]
    assert sigma_call[CNOT()] == sigma_dcmp[CNOT()]
