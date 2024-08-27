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
import sympy

import qualtran.testing as qlt_testing
from qualtran import QUInt
from qualtran._infra.gate_with_registers import split_qubits, total_bits
from qualtran.bloqs.data_loading.qrom import _qrom_multi_data, _qrom_multi_dim, _qrom_small, QROM
from qualtran.cirq_interop.t_complexity_protocol import t_complexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_qrom_small(bloq_autotester):
    bloq_autotester(_qrom_small)


def test_qrom_multi_data(bloq_autotester):
    bloq_autotester(_qrom_multi_data)


def test_qrom_multi_dim(bloq_autotester):
    bloq_autotester(_qrom_multi_dim)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data,num_controls",
    [
        pytest.param(data, num_controls, id=f"{num_controls}-data{idx}")
        for idx, data in enumerate(
            [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
        )
        for num_controls in [0, 1, 2]
    ],
)
def test_qrom_1d_full(data, num_controls: int):
    qrom = QROM.build_from_data(*data, num_controls=num_controls)
    qlt_testing.assert_valid_bloq_decomposition(qrom)

    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    # missing cirq stubs
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))  # type: ignore[operator]

    assert (
        len(inverse.all_qubits())
        <= total_bits(g.r) + total_bits(qrom.selection_registers) + num_controls
    )
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()

    controls = {'control': 2**num_controls - 1} if num_controls else {}
    zero_targets = {f'target{i}_': 0 for i in range(len(data))}
    for selection_integer in range(len(data[0])):

        out = qrom.call_classically(**controls, selection=selection_integer, **zero_targets)
        for i in range(len(data)):
            assert out[-i - 1] == data[-i - 1][selection_integer]

        for cval in range(2):
            qubit_vals = {x: 0 for x in g.all_qubits}
            qubit_vals.update(
                zip(
                    g.quregs.get('selection', ()),
                    QUInt(total_bits(qrom.selection_registers)).to_bits(selection_integer),
                )
            )
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))

            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}_"]
                    qubit_vals.update(zip(target, QUInt(len(target)).to_bits(d[selection_integer])))
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


def test_qrom_1d_classical():
    rs = np.random.RandomState()
    data = rs.randint(0, 2**3, size=10)
    sel_size = int(np.ceil(np.log2(10)))
    qrom = QROM([data], (sel_size,), target_bitsizes=(3,))
    cbloq = qrom.decompose_bloq()
    for i in range(len(data)):
        i_out, data_out = qrom.call_classically(selection=i, target0_=0)
        assert i_out == i
        assert data_out == data[i]

        decomp_ret = cbloq.call_classically(selection=i, target0_=0)
        assert decomp_ret == (i_out, data_out)


def test_qrom_1d_classical_nonzero_target():
    rs = np.random.RandomState()
    data = rs.randint(0, 2**3, size=10)
    sel_size = int(np.ceil(np.log2(10)))
    qrom = QROM([data], (sel_size,), target_bitsizes=(3,))
    cbloq = qrom.decompose_bloq()
    for i in range(len(data)):
        target_in = int('111', 2)
        i_out, data_out = qrom.call_classically(selection=i, target0_=target_in)
        assert i_out == i
        assert data_out == data[i] ^ target_in

        decomp_ret = cbloq.call_classically(selection=i, target0_=target_in)
        assert decomp_ret == (i_out, data_out)


def test_qrom_1d_multitarget_classical():
    rs = np.random.RandomState()
    n = 10
    data_sets = [rs.randint(0, 2**3, size=n) for _ in range(3)]
    sel_size = int(np.ceil(np.log2(10)))
    qrom = QROM(data_sets, (sel_size,), target_bitsizes=(3, 3, 3))
    cbloq = qrom.decompose_bloq()
    for i in range(n):
        init = {f'target{i}_': 0 for i in range(3)}
        i_out, *data_out = qrom.call_classically(selection=i, **init)
        assert i_out == i
        assert data_out == [data[i] for data in data_sets]

        decomp_i_out, *decomp_data_out = cbloq.call_classically(selection=i, **init)
        assert decomp_i_out == i_out
        assert len(data_out) == len(decomp_data_out)
        for do, decomp_do in zip(data_out, decomp_data_out):
            np.testing.assert_array_equal(do, decomp_do)


def test_qrom_3d_classical():
    rs = np.random.RandomState()
    data = rs.randint(0, 2**3, size=(3, 2, 4))
    sel_sizes = (2, 1, 2)
    qrom = QROM([data], sel_sizes, target_bitsizes=(3,))
    cbloq = qrom.decompose_bloq()
    for i in range(3):
        for j in range(2):
            for k in range(4):
                *i_out, data_out = qrom.call_classically(
                    selection0=i, selection1=j, selection2=k, target0_=0
                )
                assert i_out == [i, j, k]
                assert data_out == data[i, j, k]

                *decomp_i_out, decomp_data_out = cbloq.call_classically(
                    selection0=i, selection1=j, selection2=k, target0_=0
                )
                assert decomp_i_out == i_out
                assert decomp_data_out == data_out


def test_qrom_diagram():
    d0 = np.array([1, 2, 3])
    d1 = np.array([4, 5, 6])
    qrom = QROM.build_from_data(d0, d1)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**split_qubits(qrom.signature, q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In───────
      │
1: ───In───────
      │
2: ───QROM_a───
      │
3: ───QROM_a───
      │
4: ───QROM_b───
      │
5: ───QROM_b───
      │
6: ───QROM_b───""",
    )


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('qrom')


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
def test_t_complexity(data):
    qrom = QROM.build_from_data(*data)
    n = np.prod(qrom.data[0].shape)
    assert t_complexity(qrom).t == max(0, 4 * n - 8), n


def test_t_complexity_symbolic():
    N, M = sympy.symbols('N M')
    b1, b2 = sympy.symbols('b1 b2')
    t1, t2 = sympy.symbols('t1 t2')
    c = sympy.Symbol('c')
    qrom_symb = QROM.build_from_bitsize(
        (N, M), (b1, b2), target_shapes=((t1,), (t2,)), num_controls=c
    )
    t_counts = qrom_symb.t_complexity()
    n_and = N * M - 2 + c
    assert t_counts.t == 4 * n_and
    from qualtran.bloqs.mcmt import And

    assert (
        t_counts.clifford
        == N * M * b1 * b2 * t1 * t2
        + (And().t_complexity().clifford + And().adjoint().t_complexity().clifford) * n_and
    )


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
    assert t_complexity(QROM.build_from_data(data)).t == (8 - 2) * 4
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]  # Figure 3b.
    assert t_complexity(QROM.build_from_data(data)).t == (5 - 2) * 4
    data = [1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]  # Negative test: t count is not (g-2)*4
    assert t_complexity(QROM.build_from_data(data)).t == (8 - 2) * 4
    # Works as expected when multiple data arrays are to be loaded.
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    # (a) Both data sequences are identical
    assert t_complexity(QROM.build_from_data(data, data)).t == (5 - 2) * 4
    # (b) Both data sequences have identical structure, even though the elements are not same.
    assert t_complexity(QROM.build_from_data(data, 2 * np.array(data))).t == (5 - 2) * 4
    # Works as expected when multidimensional input data is to be loaded
    qrom = QROM.build_from_data(
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
selection00: ───X───@────X───@────
                    │        │
target0_0: ─────────⊕1───────⊕2───
                    │        │
target0_1: ─────────⊕1───────⊕2───
    ''',
    )
    # When inner loop range is not a power of 2, the inner segment tree cannot be skipped.
    qrom = QROM.build_from_data(
        np.array(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
            dtype=int,
        )
    )
    _assert_qrom_has_diagram(
        qrom,
        r'''
selection00: ───X───@──────────@────────@──────X───@──────────@────────@──────
                    │          │        │          │          │        │
selection10: ───────(0)────────┼────────@──────────(0)────────┼────────@──────
                    │          │        │          │          │        │
anc_1: ─────────────And───@────X───@────And†───────And───@────X───@────And†───
                          │        │                     │        │
target0_0: ───────────────⊕1───────⊕1────────────────────⊕2───────⊕2──────────
                          │        │                     │        │
target0_1: ───────────────⊕1───────⊕1────────────────────⊕2───────⊕2──────────
''',
    )
    # No T-gates needed if all elements to load are identical.
    assert t_complexity(QROM.build_from_data([3, 3, 3, 3])).t == 0


def test_qrom_wire_symbols():
    qrom = QROM.build_from_data([3, 3, 3, 3])
    qlt_testing.assert_wire_symbols_match_expected(qrom, ['In', 'QROM_a'])

    qrom = QROM.build_from_data([3, 3, 3, 3], [2, 2, 2, 2])
    qlt_testing.assert_wire_symbols_match_expected(qrom, ['In', 'QROM_a', 'QROM_b'])

    qrom = QROM.build_from_data([[3, 3], [3, 3]], [[2, 2], [2, 2]], [[1, 1], [2, 2]])
    qlt_testing.assert_wire_symbols_match_expected(
        qrom, ['In_i', 'In_j', 'QROM_a', 'QROM_b', 'QROM_c']
    )

    qrom = QROM.build_from_data(np.arange(27).reshape(3, 3, 3))
    qlt_testing.assert_wire_symbols_match_expected(qrom, ['In_i', 'In_j', 'In_k', 'QROM_a'])


@pytest.mark.slow
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
def test_qrom_multi_dim_full(data, num_controls):
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = QROM(
        data,
        selection_bitsizes=selection_bitsizes,
        target_bitsizes=target_bitsizes,
        num_controls=num_controls,
    )
    qlt_testing.assert_valid_bloq_decomposition(qrom)

    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    # Missing cirq stubs
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))  # type: ignore[operator]

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
                    zip(g.quregs[f'selection{isel}'], QUInt(lens[isel]).to_bits(idxs[isel]))
                )
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}_"]
                    qubit_vals.update(zip(target, QUInt(len(target)).to_bits(int(d[idxs]))))
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
    n = data[0].size
    assert t_complexity(qrom).t == max(0, 4 * n - 8 + 4 * num_controls)


@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_call_graph_matches_decomposition(num_controls):
    # Base case
    arr = np.arange(50)
    qrom = QROM.build_from_data(arr, num_controls=num_controls)
    cost_call = get_cost_value(qrom, QECGatesCost())
    cost_dcmp = get_cost_value(qrom.decompose_bloq(), QECGatesCost())
    assert cost_call.total_t_and_ccz_count() == cost_dcmp.total_t_and_ccz_count()

    # Multiple Multi dimensional arrays
    arr_a = np.arange(64).reshape(8, 8)
    arr_b = 10 * np.arange(64).reshape(8, 8)
    qrom = QROM.build_from_data(arr_a, arr_b, num_controls=num_controls)
    cost_call = get_cost_value(qrom, QECGatesCost())
    cost_dcmp = get_cost_value(qrom.decompose_bloq(), QECGatesCost())
    assert cost_call.total_t_and_ccz_count() == cost_dcmp.total_t_and_ccz_count()

    # Variable QROM case.
    arr_a = np.array([1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5])
    arr_b = 10 * arr_a
    qrom = QROM.build_from_data(arr_a, arr_b, num_controls=num_controls)
    cost_call = get_cost_value(qrom, QECGatesCost())
    cost_dcmp = get_cost_value(qrom.decompose_bloq(), QECGatesCost())
    assert cost_call.total_t_and_ccz_count() == cost_dcmp.total_t_and_ccz_count()
