import numpy as np
import pytest
import cirq
import cirq_qubitization
import random

random.seed(12345)


@pytest.mark.parametrize(
    "selection_register, target_register_bit_size, target_register_length",
    [[2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]],
)
def test_swap_with_zero_gate(
    selection_register, target_register_bit_size, target_register_length
):
    # Construct the gate.
    gate = cirq_qubitization.SwapWithZeroGate(
        selection_register, target_register_bit_size, target_register_length
    )
    # Allocate selection and target qubits.
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    selection = all_qubits[:selection_register]
    target = [
        all_qubits[st : st + target_register_bit_size]
        for st in range(selection_register, len(all_qubits), target_register_bit_size)
    ]
    # Create a circuit.
    circuit = cirq.Circuit(gate.on(selection=selection, target=target))

    # Load data[i] in i'th target register; where each register is of size target_register_bit_size
    data = [
        random.randint(0, 2**target_register_bit_size - 1)
        for _ in range(target_register_length)
    ]
    target_state = [
        int(x) for d in data for x in format(d, f"0{target_register_bit_size}b")
    ]

    sim = cirq.Simulator(dtype=np.complex128)
    expected_state_vector = np.zeros(2**target_register_bit_size)
    # Iterate on every selection integer.
    for selection_integer in range(len(data)):
        # Load `selection_integer` in the selection register and construct initial state.
        selection_state = [
            int(x) for x in format(selection_integer, f"0{selection_register}b")
        ]
        initial_state = selection_state + target_state
        # Simulate the circuit with the initial state.
        result = sim.simulate(circuit, initial_state=initial_state)
        # Get the sub_state_vector corresponding to qubit register `target[0]`.
        result_state_vector = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=list(
                range(selection_register, selection_register + target_register_bit_size)
            ),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0
