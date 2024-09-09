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

import cirq
import numpy as np
import pytest

from qualtran._infra.data_types import QUInt
from qualtran._infra.gate_with_registers import get_named_qubits, split_qubits
from qualtran.bloqs.data_loading import QROM
from qualtran.bloqs.data_loading.select_swap_qrom import (
    _qroam_multi_data,
    _qroam_multi_dim,
    find_optimal_log_block_size,
    SelectSwapQROM,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize(
    "data,block_size",
    [
        pytest.param(
            data,
            block_size,
            id=f"{block_size}-data{didx}",
            marks=pytest.mark.slow if block_size == 1 and didx == 1 else (),
        )
        for didx, data in enumerate([[[1, 2, 3, 4, 5]], [[1, 2, 3], [3, 2, 1]], [[1], [2], [3]]])
        for block_size in [None, 0, 1]
        if block_size is None or 2**block_size <= len(data[0])
    ],
)
def test_select_swap_qrom(data, block_size):
    qrom = SelectSwapQROM.build_from_data(*data, log_block_sizes=block_size)
    assert_valid_bloq_decomposition(qrom)

    qubit_regs = get_named_qubits(qrom.signature)
    selection = qubit_regs.get("selection", ())
    q_len = qrom.batched_qrom_selection_bitsizes[0]
    assert isinstance(q_len, int)
    selection_q, selection_r = (selection[:q_len], selection[q_len:])
    targets = [qubit_regs[f"target{i}_"] for i in range(len(data))]

    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    qrom_circuit = cirq.Circuit(
        cirq.decompose_once(qrom.on_registers(**qubit_regs), context=context)
    )
    dirty_target_ancilla = sorted(
        qrom_circuit.all_qubits() - set(q for qs in qubit_regs.values() for q in qs.flatten())
    )
    assert greedy_mm._size == len(dirty_target_ancilla) == len(greedy_mm._free_qubits)

    circuit = cirq.Circuit(
        # Prepare dirty ancillas in an arbitrary state.
        cirq.H.on_each(*dirty_target_ancilla),
        cirq.T.on_each(*dirty_target_ancilla),
        # The dirty ancillas should remain unaffected by qroam.
        *qrom_circuit,
        # Bring back the dirty ancillas to their original state.
        (cirq.T**-1).on_each(*dirty_target_ancilla),
        cirq.H.on_each(*dirty_target_ancilla),
    )
    all_qubits = sorted(circuit.all_qubits())
    for selection_integer in range(len(data[0])):
        svals_q = QUInt(len(selection_q)).to_bits(selection_integer // qrom.block_sizes[0])
        svals_r = QUInt(len(selection_r)).to_bits(selection_integer % qrom.block_sizes[0])
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection_q, svals_q)})
        qubit_vals.update({s: sval for s, sval in zip(selection_r, svals_r)})

        dvals = np.random.randint(2, size=len(dirty_target_ancilla))
        qubit_vals.update({d: dval for d, dval in zip(dirty_target_ancilla, dvals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, data):
            for q, b in zip(target, QUInt(len(target)).to_bits(d[selection_integer])):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]
        assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_select_swap_qrom_classical_sim():
    rng = np.random.default_rng(42)
    # 1D data, 1 dataset
    N, max_N, log_block_sizes = 25, 2**10, 3
    data = rng.integers(max_N, size=N)
    bloq = SelectSwapQROM.build_from_data(data, log_block_sizes=log_block_sizes)
    cbloq = bloq.decompose_bloq()
    for x in range(N):
        vals = bloq.call_classically(selection=x, target0_=0)
        cvals = cbloq.call_classically(selection=x, target0_=0)
        assert vals == cvals == (x, data[x])

    # 2D data, 1 datasets
    N, M, max_N, log_block_sizes = 7, 11, 2**5, (2, 3)
    data = rng.integers(max_N, size=N * M).reshape(N, M)
    bloq = SelectSwapQROM.build_from_data(data, log_block_sizes=log_block_sizes)
    cbloq = bloq.decompose_bloq()
    for x in range(N):
        for y in range(M):
            vals = bloq.call_classically(selection0=x, selection1=y, target0_=0)
            cvals = cbloq.call_classically(selection0=x, selection1=y, target0_=0)
            assert vals == cvals == (x, y, data[x][y])


@pytest.mark.slow
def test_select_swap_qrom_classical_sim_multi_dataset():
    rng = np.random.default_rng(42)
    # 1D data, 2 datasets
    N, max_N, log_block_sizes = 25, 2**20, 3
    data = [rng.integers(max_N, size=N), rng.integers(max_N, size=N)]
    bloq = SelectSwapQROM.build_from_data(*data, log_block_sizes=log_block_sizes)
    cbloq = bloq.decompose_bloq()
    for x in range(N):
        vals = bloq.call_classically(selection=x, target0_=0, target1_=0)
        cvals = cbloq.call_classically(selection=x, target0_=0, target1_=0)
        assert vals == cvals == (x, data[0][x], data[1][x])

    # 2D data, 2 datasets
    N, M, max_N, log_block_sizes = 7, 11, 2**5, (2, 3)
    data = [
        rng.integers(max_N, size=N * M).reshape(N, M),
        rng.integers(max_N, size=N * M).reshape(N, M),
    ]
    bloq = SelectSwapQROM.build_from_data(*data, log_block_sizes=log_block_sizes)
    cbloq = bloq.decompose_bloq()
    for x in range(N):
        for y in range(M):
            vals = bloq.call_classically(selection0=x, selection1=y, target0_=0, target1_=0)
            cvals = cbloq.call_classically(selection0=x, selection1=y, target0_=0, target1_=0)
            assert vals == cvals == (x, y, data[0][x][y], data[1][x][y])


def test_qroam_diagram():
    data = [[1, 2, 3], [4, 5, 6]]
    qrom = SelectSwapQROM.build_from_data(*data, log_block_sizes=(1,))
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**split_qubits(qrom.signature, q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In────────
      │
1: ───In────────
      │
2: ───QROAM_a───
      │
3: ───QROAM_a───
      │
4: ───QROAM_b───
      │
5: ───QROAM_b───
      │
6: ───QROAM_b───
""",
    )


def test_qroam_raises():
    with pytest.raises(ValueError, match="must have same shape"):
        _ = SelectSwapQROM.build_from_data([1, 2], [1, 2, 3])


def test_qroam_hashable():
    qrom = SelectSwapQROM.build_from_data([1, 2, 5, 6, 7, 8])
    assert hash(qrom) is not None
    assert t_complexity(qrom) == TComplexity(32, 160, 0)


def test_qroam_t_complexity():
    qroam = SelectSwapQROM.build_from_data(
        [1, 2, 3, 4, 5, 6, 7, 8], target_bitsizes=(4,), log_block_sizes=(2,)
    )
    gate_counts = get_cost_value(qroam, QECGatesCost())
    assert gate_counts == GateCounts(t=192, clifford=1082)
    assert qroam.t_complexity() == TComplexity(t=192, clifford=1082)


def test_qroam_many_registers():
    # Test > 10 registers which resulted in https://github.com/quantumlib/Qualtran/issues/556
    target_bitsizes = (3,) * 10 + (1,) * 2 + (3,)
    log_block_size = find_optimal_log_block_size(10, sum(target_bitsizes))
    qrom = SelectSwapQROM.build_from_data(
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (1,) * 10,
        (0,) * 10,
        (1,) * 10,
        (3,) * 10,
        log_block_sizes=(log_block_size,),
    )
    qrom.call_graph()


def test_qroam_multi_data_autotest(bloq_autotester):
    bloq_autotester(_qroam_multi_data)


def test_qroam_multi_dim_autotest(bloq_autotester):
    bloq_autotester(_qroam_multi_dim)


@pytest.mark.parametrize('use_dirty_ancilla', [pytest.param(True, marks=pytest.mark.slow), False])
def test_tensor_contraction(use_dirty_ancilla: bool):
    data = np.array([[0, 1, 0, 1]] * 8)
    log_block_sizes = (2, 1)
    qroam = SelectSwapQROM.build_from_data(
        data, use_dirty_ancilla=use_dirty_ancilla, log_block_sizes=log_block_sizes
    )
    qrom = QROM.build_from_data(data)
    np.testing.assert_allclose(qrom.tensor_contract(), qroam.tensor_contract(), atol=1e-8)


def test_select_swap_block_sizes():
    data = [*range(1600)]
    qroam_opt = SelectSwapQROM.build_from_data(data)
    qroam_subopt = SelectSwapQROM.build_from_data(data, log_block_sizes=(8,))
    assert qroam_opt.block_sizes == (16,)
    assert qroam_opt.t_complexity().t < qroam_subopt.t_complexity().t

    qroam = SelectSwapQROM.build_from_data(data, use_dirty_ancilla=False)
    assert qroam.block_sizes == (8,)
