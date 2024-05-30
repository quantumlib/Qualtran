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

from qualtran._infra.data_types import BoundedQUInt
from qualtran._infra.gate_with_registers import get_named_qubits, split_qubits
from qualtran.bloqs.data_loading.select_swap_qrom import find_optimal_log_block_size, SelectSwapQROM
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.slow
@pytest.mark.parametrize(
    "data,block_size",
    [
        pytest.param(
            data,
            block_size,
            id=f"{block_size}-data{didx}",
            marks=pytest.mark.slow if block_size == 2 and didx == 0 else (),
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

    dirty_target_ancilla = [
        q for q in qrom_circuit.all_qubits() if isinstance(q, cirq.ops.BorrowableQubit)
    ]

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
    dtype = qrom.selection_registers[0].dtype
    assert isinstance(dtype, BoundedQUInt)
    for selection_integer in range(int(dtype.iteration_length)):
        svals_q = list(iter_bits(selection_integer // qrom.block_sizes[0], len(selection_q)))
        svals_r = list(iter_bits(selection_integer % qrom.block_sizes[0], len(selection_r)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection_q, svals_q)})
        qubit_vals.update({s: sval for s, sval in zip(selection_r, svals_r)})

        dvals = np.random.randint(2, size=len(dirty_target_ancilla))
        qubit_vals.update({d: dval for d, dval in zip(dirty_target_ancilla, dvals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, data):
            for q, b in zip(target, iter_bits(d[selection_integer], len(target))):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]
        assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


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


def test_qrom_t_complexity():
    qrom = SelectSwapQROM.build_from_data(
        [1, 2, 3, 4, 5, 6, 7, 8], target_bitsizes=(4,), log_block_sizes=(2,)
    )
    _, sigma = qrom.call_graph()
    assert t_counts_from_sigma(sigma) == qrom.t_complexity().t == 192


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
