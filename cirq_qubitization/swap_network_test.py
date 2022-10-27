import random

import cirq
import numpy as np
import pytest

import cirq_qubitization
import cirq_qubitization.testing as cq_testing

random.seed(12345)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers",
    [[3, 5, 1], [2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]],
)
def test_swap_with_zero_gate(selection_bitsize, target_bitsize, n_target_registers):
    # Construct the gate.
    gate = cirq_qubitization.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    # Allocate selection and target qubits.
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    selection = all_qubits[:selection_bitsize]
    targets = {
        f'target{i}': all_qubits[st : st + target_bitsize]
        for i, st in enumerate(range(selection_bitsize, len(all_qubits), target_bitsize))
    }
    # Create a circuit.
    circuit = cirq.Circuit(gate.on_registers(selection=selection, **targets))

    # Load data[i] in i'th target register; where each register is of size target_bitsize
    data = [random.randint(0, 2**target_bitsize - 1) for _ in range(n_target_registers)]
    target_state = [int(x) for d in data for x in format(d, f"0{target_bitsize}b")]

    sim = cirq.Simulator(dtype=np.complex128)
    expected_state_vector = np.zeros(2**target_bitsize)
    # Iterate on every selection integer.
    for selection_integer in range(len(data)):
        # Load `selection_integer` in the selection register and construct initial state.
        selection_state = [int(x) for x in format(selection_integer, f"0{selection_bitsize}b")]
        initial_state = selection_state + target_state
        # Simulate the circuit with the initial state.
        result = sim.simulate(circuit, initial_state=initial_state)
        # Get the sub_state_vector corresponding to qubit register `target[0]`.
        result_state_vector = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0


def test_swap_with_zero_gate_diagram():
    gate = cirq_qubitization.SwapWithZeroGate(3, 2, 4)
    q = cirq.LineQubit.range(cirq.num_qubits(gate))
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ────@(r⇋0)───
       │
1: ────@(r⇋0)───
       │
2: ────@(r⇋0)───
       │
3: ────swap_0───
       │
4: ────swap_0───
       │
5: ────swap_1───
       │
6: ────swap_1───
       │
7: ────swap_2───
       │
8: ────swap_2───
       │
9: ────swap_3───
       │
10: ───swap_3───
""",
    )


def test_multi_target_cswap():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = qubits[0], qubits[1:3], qubits[3:]
    cswap = cirq_qubitization.MultiTargetCSwap(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    cswap_approx = cirq_qubitization.MultiTargetCSwapApprox(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    setup_code = "import cirq\nimport cirq_qubitization"
    cirq.testing.assert_implements_consistent_protocols(cswap, setup_code=setup_code)
    cirq.testing.assert_implements_consistent_protocols(cswap_approx, setup_code=setup_code)
    circuit = cirq.Circuit(cswap, cswap_approx)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───@──────@(approx)───
      │      │
1: ───×(x)───×(x)────────
      │      │
2: ───×(x)───×(x)────────
      │      │
3: ───×(y)───×(y)────────
      │      │
4: ───×(y)───×(y)────────
    """,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ---@--------@(approx)---
      |        |
1: ---swap_x---swap_x------
      |        |
2: ---swap_x---swap_x------
      |        |
3: ---swap_y---swap_y------
      |        |
4: ---swap_y---swap_y------
    """,
        use_unicode_characters=False,
    )


def test_notebook():
    cq_testing.execute_notebook('qrom')
