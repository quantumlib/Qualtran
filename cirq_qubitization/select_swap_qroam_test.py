import numpy as np
import pytest
import cirq

import cirq_qubitization
from cirq_qubitization import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [3, 2, 1]]])
@pytest.mark.parametrize("block_size", [None, 1, 2, 3])
def test_select_swap_qrom(data, block_size):
    qrom = cirq_qubitization.SelectSwapQROM(*data, block_size=block_size)
    qubit_regs = qrom.registers.get_named_qubits()
    all_qubits = qrom.registers.merge_qubits(**qubit_regs)
    selection_q = qubit_regs["selection_q"]
    selection_r = qubit_regs["selection_r"]
    targets = [qubit_regs[f"target{i}"] for i in range(len(data))]
    dirty_target_ancilla = qrom.target_dirty_ancilla.merge_qubits(**qubit_regs)

    circuit = cirq.Circuit(
        # Prepare dirty ancillas in an arbitrary state.
        cirq.H.on_each(*dirty_target_ancilla),
        cirq.T.on_each(*dirty_target_ancilla),
        # The dirty ancillas should remain unaffected by qroam.
        qrom.on_registers(**qubit_regs),
        # Bring back the dirty ancillas to their original state.
        (cirq.T**-1).on_each(*dirty_target_ancilla),
        cirq.H.on_each(*dirty_target_ancilla),
    )

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
    data = [[1, 2, 3], [4, 5, 6]]
    blocksize = 2
    qrom = cirq_qubitization.SelectSwapQROM(*data, block_size=blocksize)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**qrom.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ────In_q───────
       │
1: ────In_r───────
       │
2: ────QROAM_0────
       │
3: ────QROAM_0────
       │
4: ────QROAM_1────
       │
5: ────QROAM_1────
       │
6: ────QROAM_1────
       │
7: ────TAnc_0_0───
       │
8: ────TAnc_0_0───
       │
9: ────TAnc_0_1───
       │
10: ───TAnc_0_1───
       │
11: ───TAnc_0_1───
       │
12: ───TAnc_1_0───
       │
13: ───TAnc_1_0───
       │
14: ───TAnc_1_1───
       │
15: ───TAnc_1_1───
       │
16: ───TAnc_1_1───
""",
    )


def test_qroam_raises():
    with pytest.raises(ValueError, match="must be of equal length"):
        _ = cirq_qubitization.SelectSwapQROM([1, 2], [1, 2, 3])


def test_qroam_make_on():
    data = [[1, 0, 1] * 33, [2, 3, 2] * 33]
    selection = [cirq.q(f'selection_{i}') for i in range(7)]
    targets = [[cirq.q(f'target_{i}_{j}') for j in range(1 + i)] for i in range(2)]
    ancilla = cirq_qubitization.GreedyQubitManager(prefix="ancilla")
    qrom_op = cirq_qubitization.SelectSwapQROM.make_on(
        *data, selection=selection, target0=targets[0], target1=targets[1], ancilla=ancilla
    )
    circuit = cirq.Circuit(qrom_op)
    print(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        """
ancilla_0: ─────Anc────────
                │
ancilla_1: ─────Anc────────
                │
ancilla_2: ─────Anc────────
                │
ancilla_3: ─────TAnc_0_0───
                │
ancilla_4: ─────TAnc_0_1───
                │
ancilla_5: ─────TAnc_0_1───
                │
ancilla_6: ─────TAnc_1_0───
                │
ancilla_7: ─────TAnc_1_1───
                │
ancilla_8: ─────TAnc_1_1───
                │
ancilla_9: ─────TAnc_2_0───
                │
ancilla_10: ────TAnc_2_1───
                │
ancilla_11: ────TAnc_2_1───
                │
ancilla_12: ────TAnc_3_0───
                │
ancilla_13: ────TAnc_3_1───
                │
ancilla_14: ────TAnc_3_1───
                │
ancilla_15: ────TAnc_4_0───
                │
ancilla_16: ────TAnc_4_1───
                │
ancilla_17: ────TAnc_4_1───
                │
ancilla_18: ────TAnc_5_0───
                │
ancilla_19: ────TAnc_5_1───
                │
ancilla_20: ────TAnc_5_1───
                │
ancilla_21: ────TAnc_6_0───
                │
ancilla_22: ────TAnc_6_1───
                │
ancilla_23: ────TAnc_6_1───
                │
ancilla_24: ────TAnc_7_0───
                │
ancilla_25: ────TAnc_7_1───
                │
ancilla_26: ────TAnc_7_1───
                │
selection_0: ───In_q───────
                │
selection_1: ───In_q───────
                │
selection_2: ───In_q───────
                │
selection_3: ───In_q───────
                │
selection_4: ───In_r───────
                │
selection_5: ───In_r───────
                │
selection_6: ───In_r───────
                │
target_0_0: ────QROAM_0────
                │
target_1_0: ────QROAM_1────
                │
target_1_1: ────QROAM_1────
""",
    )
