import cirq
import numpy as np
import pytest

import cirq_qubitization
from cirq_qubitization import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [3, 2, 1]]])
@pytest.mark.parametrize("block_size", [None, 1, 2, 3])
def test_select_swap_qrom(data, block_size):
    qrom = cirq_qubitization.SelectSwapQROM(*data, block_size=block_size)
    qubit_regs = qrom.registers.get_named_qubits()
    selection = qubit_regs["selection"]
    selection_q, selection_r = selection[: qrom.selection_q], selection[qrom.selection_q :]
    targets = [qubit_regs[f"target{i}"] for i in range(len(data))]
    qrom_circuit = cirq.Circuit(cirq.decompose(qrom.on_registers(**qubit_regs)))
    dirty_target_ancilla = [
        q for q in qrom_circuit.all_qubits() if isinstance(q, cirq_qubitization.BorrowableQubit)
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

    all_qubits = circuit.all_qubits()
    for selection_integer in range(qrom.iteration_length):
        svals_q = list(iter_bits(selection_integer // qrom.block_size, len(selection_q)))
        svals_r = list(iter_bits(selection_integer % qrom.block_size, len(selection_r)))
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
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_qrom_repr():
    qrom = cirq_qubitization.SelectSwapQROM([1, 2], [3, 5])
    cirq.testing.assert_equivalent_repr(qrom, setup_code="import cirq_qubitization\n")


def test_qroam_diagram():
    cirq_qubitization.qalloc_reset()
    data = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]
    blocksize = 2
    qrom = cirq_qubitization.SelectSwapQROM(*data, block_size=blocksize)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    op = qrom.on_registers(**qrom.registers.split_qubits(q))
    circuit = cirq.Circuit(op, cirq.decompose_once(op))
    cirq.testing.assert_has_diagram(
        circuit,
        """
                                    ┌─────┐                                 ┌─────┐
_b0: ─────────────QROM_0───swap_0────@────────swap_0──────QROM_0───swap_0────@────────swap_0──────
                  │        │         │        │           │        │         │        │
_b1: ─────────────QROM_0───swap_0────┼@───────swap_0──────QROM_0───swap_0────┼@───────swap_0──────
                  │        │         ││       │           │        │         ││       │
_b2: ─────────────QROM_1───swap_0────┼┼@──────swap_0──────QROM_1───swap_0────┼┼@──────swap_0──────
                  │        │         │││      │           │        │         │││      │
_b3: ─────────────QROM_1───swap_0────┼┼┼@─────swap_0──────QROM_1───swap_0────┼┼┼@─────swap_0──────
                  │        │         ││││     │           │        │         ││││     │
_b4: ─────────────QROM_1───swap_0────┼┼┼┼@────swap_0──────QROM_1───swap_0────┼┼┼┼@────swap_0──────
                  │        │         │││││    │           │        │         │││││    │
_b5: ─────────────QROM_2───swap_1────┼┼┼┼┼────swap_1──────QROM_2───swap_1────┼┼┼┼┼────swap_1──────
                  │        │         │││││    │           │        │         │││││    │
_b6: ─────────────QROM_2───swap_1────┼┼┼┼┼────swap_1──────QROM_2───swap_1────┼┼┼┼┼────swap_1──────
                  │        │         │││││    │           │        │         │││││    │
_b7: ─────────────QROM_3───swap_1────┼┼┼┼┼────swap_1──────QROM_3───swap_1────┼┼┼┼┼────swap_1──────
                  │        │         │││││    │           │        │         │││││    │
_b8: ─────────────QROM_3───swap_1────┼┼┼┼┼────swap_1──────QROM_3───swap_1────┼┼┼┼┼────swap_1──────
                  │        │         │││││    │           │        │         │││││    │
_b9: ─────────────QROM_3───swap_1────┼┼┼┼┼────swap_1──────QROM_3───swap_1────┼┼┼┼┼────swap_1──────
                  │        │         │││││    │           │        │         │││││    │
_c0: ─────────────Anc──────┼─────────┼┼┼┼┼────┼───────────Anc──────┼─────────┼┼┼┼┼────┼───────────
                  │        │         │││││    │           │        │         │││││    │
0: ─────In_q──────In───────┼─────────┼┼┼┼┼────┼───────────In───────┼─────────┼┼┼┼┼────┼───────────
        │         │        │         │││││    │           │        │         │││││    │
1: ─────In_q──────In───────┼─────────┼┼┼┼┼────┼───────────In^-1────┼─────────┼┼┼┼┼────┼───────────
        │                  │         │││││    │                    │         │││││    │
2: ─────In_r───────────────@(r⇋0)────┼┼┼┼┼────@(r⇋0)^-1────────────@(r⇋0)────┼┼┼┼┼────@(r⇋0)^-1───
        │                            │││││                                   │││││
3: ─────QROAM_0──────────────────────X┼┼┼┼───────────────────────────────────X┼┼┼┼────────────────
        │                             ││││                                    ││││
4: ─────QROAM_0───────────────────────X┼┼┼────────────────────────────────────X┼┼┼────────────────
        │                              │││                                     │││
5: ─────QROAM_1────────────────────────X┼┼─────────────────────────────────────X┼┼────────────────
        │                               ││                                      ││
6: ─────QROAM_1─────────────────────────X┼──────────────────────────────────────X┼────────────────
        │                                │                                       │
7: ─────QROAM_1──────────────────────────X───────────────────────────────────────X────────────────
                                    └─────┘                                 └─────┘
""",
    )


def test_qroam_raises():
    with pytest.raises(ValueError, match="must be of equal length"):
        _ = cirq_qubitization.SelectSwapQROM([1, 2], [1, 2, 3])


def test_qroam_make_on():
    cirq_qubitization.qalloc_reset()
    data = [[1, 0, 1] * 33, [2, 3, 2] * 33]
    selection = [cirq.q(f'selection_{i}') for i in range(7)]
    targets = [[cirq.q(f'target_{i}_{j}') for j in range(1 + i)] for i in range(2)]
    qrom = cirq_qubitization.SelectSwapQROM(*data)
    op = qrom.on_registers(selection=selection, target0=targets[0], target1=targets[1])
    circuit = cirq.Circuit(op, cirq.decompose_once(op))
    cirq.testing.assert_has_diagram(
        circuit,
        """
                                             ┌───┐                                  ┌───┐
_b0: ─────────────────────QROM_0────swap_0────@──────swap_0──────QROM_0────swap_0────@──────swap_0──────
                          │         │         │      │           │         │         │      │
_b1: ─────────────────────QROM_1────swap_0────┼@─────swap_0──────QROM_1────swap_0────┼@─────swap_0──────
                          │         │         ││     │           │         │         ││     │
_b2: ─────────────────────QROM_1────swap_0────┼┼@────swap_0──────QROM_1────swap_0────┼┼@────swap_0──────
                          │         │         │││    │           │         │         │││    │
_b3: ─────────────────────QROM_2────swap_1────┼┼┼────swap_1──────QROM_2────swap_1────┼┼┼────swap_1──────
                          │         │         │││    │           │         │         │││    │
_b4: ─────────────────────QROM_3────swap_1────┼┼┼────swap_1──────QROM_3────swap_1────┼┼┼────swap_1──────
                          │         │         │││    │           │         │         │││    │
_b5: ─────────────────────QROM_3────swap_1────┼┼┼────swap_1──────QROM_3────swap_1────┼┼┼────swap_1──────
                          │         │         │││    │           │         │         │││    │
_b6: ─────────────────────QROM_4────swap_2────┼┼┼────swap_2──────QROM_4────swap_2────┼┼┼────swap_2──────
                          │         │         │││    │           │         │         │││    │
_b7: ─────────────────────QROM_5────swap_2────┼┼┼────swap_2──────QROM_5────swap_2────┼┼┼────swap_2──────
                          │         │         │││    │           │         │         │││    │
_b8: ─────────────────────QROM_5────swap_2────┼┼┼────swap_2──────QROM_5────swap_2────┼┼┼────swap_2──────
                          │         │         │││    │           │         │         │││    │
_b9: ─────────────────────QROM_6────swap_3────┼┼┼────swap_3──────QROM_6────swap_3────┼┼┼────swap_3──────
                          │         │         │││    │           │         │         │││    │
_b10: ────────────────────QROM_7────swap_3────┼┼┼────swap_3──────QROM_7────swap_3────┼┼┼────swap_3──────
                          │         │         │││    │           │         │         │││    │
_b11: ────────────────────QROM_7────swap_3────┼┼┼────swap_3──────QROM_7────swap_3────┼┼┼────swap_3──────
                          │         │         │││    │           │         │         │││    │
_b12: ────────────────────QROM_8────swap_4────┼┼┼────swap_4──────QROM_8────swap_4────┼┼┼────swap_4──────
                          │         │         │││    │           │         │         │││    │
_b13: ────────────────────QROM_9────swap_4────┼┼┼────swap_4──────QROM_9────swap_4────┼┼┼────swap_4──────
                          │         │         │││    │           │         │         │││    │
_b14: ────────────────────QROM_9────swap_4────┼┼┼────swap_4──────QROM_9────swap_4────┼┼┼────swap_4──────
                          │         │         │││    │           │         │         │││    │
_b15: ────────────────────QROM_10───swap_5────┼┼┼────swap_5──────QROM_10───swap_5────┼┼┼────swap_5──────
                          │         │         │││    │           │         │         │││    │
_b16: ────────────────────QROM_11───swap_5────┼┼┼────swap_5──────QROM_11───swap_5────┼┼┼────swap_5──────
                          │         │         │││    │           │         │         │││    │
_b17: ────────────────────QROM_11───swap_5────┼┼┼────swap_5──────QROM_11───swap_5────┼┼┼────swap_5──────
                          │         │         │││    │           │         │         │││    │
_b18: ────────────────────QROM_12───swap_6────┼┼┼────swap_6──────QROM_12───swap_6────┼┼┼────swap_6──────
                          │         │         │││    │           │         │         │││    │
_b19: ────────────────────QROM_13───swap_6────┼┼┼────swap_6──────QROM_13───swap_6────┼┼┼────swap_6──────
                          │         │         │││    │           │         │         │││    │
_b20: ────────────────────QROM_13───swap_6────┼┼┼────swap_6──────QROM_13───swap_6────┼┼┼────swap_6──────
                          │         │         │││    │           │         │         │││    │
_b21: ────────────────────QROM_14───swap_7────┼┼┼────swap_7──────QROM_14───swap_7────┼┼┼────swap_7──────
                          │         │         │││    │           │         │         │││    │
_b22: ────────────────────QROM_15───swap_7────┼┼┼────swap_7──────QROM_15───swap_7────┼┼┼────swap_7──────
                          │         │         │││    │           │         │         │││    │
_b23: ────────────────────QROM_15───swap_7────┼┼┼────swap_7──────QROM_15───swap_7────┼┼┼────swap_7──────
                          │         │         │││    │           │         │         │││    │
_c0: ─────────────────────Anc───────┼─────────┼┼┼────┼───────────Anc───────┼─────────┼┼┼────┼───────────
                          │         │         │││    │           │         │         │││    │
_c1: ─────────────────────Anc───────┼─────────┼┼┼────┼───────────Anc───────┼─────────┼┼┼────┼───────────
                          │         │         │││    │           │         │         │││    │
_c2: ─────────────────────Anc───────┼─────────┼┼┼────┼───────────Anc───────┼─────────┼┼┼────┼───────────
                          │         │         │││    │           │         │         │││    │
selection_0: ───In_q──────In────────┼─────────┼┼┼────┼───────────In────────┼─────────┼┼┼────┼───────────
                │         │         │         │││    │           │         │         │││    │
selection_1: ───In_q──────In────────┼─────────┼┼┼────┼───────────In────────┼─────────┼┼┼────┼───────────
                │         │         │         │││    │           │         │         │││    │
selection_2: ───In_q──────In────────┼─────────┼┼┼────┼───────────In────────┼─────────┼┼┼────┼───────────
                │         │         │         │││    │           │         │         │││    │
selection_3: ───In_q──────In────────┼─────────┼┼┼────┼───────────In^-1─────┼─────────┼┼┼────┼───────────
                │                   │         │││    │                     │         │││    │
selection_4: ───In_r────────────────@(r⇋0)────┼┼┼────@(r⇋0)────────────────@(r⇋0)────┼┼┼────@(r⇋0)──────
                │                   │         │││    │                     │         │││    │
selection_5: ───In_r────────────────@(r⇋0)────┼┼┼────@(r⇋0)────────────────@(r⇋0)────┼┼┼────@(r⇋0)──────
                │                   │         │││    │                     │         │││    │
selection_6: ───In_r────────────────@(r⇋0)────┼┼┼────@(r⇋0)^-1─────────────@(r⇋0)────┼┼┼────@(r⇋0)^-1───
                │                             │││                                    │││
target_0_0: ────QROAM_0───────────────────────X┼┼────────────────────────────────────X┼┼────────────────
                │                              ││                                     ││
target_1_0: ────QROAM_1────────────────────────X┼─────────────────────────────────────X┼────────────────
                │                               │                                      │
target_1_1: ────QROAM_1─────────────────────────X──────────────────────────────────────X────────────────
                                             └───┘                                  └───┘
""",
    )
