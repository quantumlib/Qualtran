import pytest
import cirq_qubitization
import cirq


@pytest.mark.parametrize(
    "selection_length,target_length",
    [(2, 4), (3, 8), (4, 9)],
)
def test_selected_majorana_fermion_gate(selection_length, target_length):
    all_qubits = cirq.LineQubit.range(2 * selection_length + target_length + 2)
    control, selection, ancilla, accumulator, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_length : 2],
        all_qubits[2 : 2 * selection_length + 1 : 2],
        all_qubits[2 * selection_length + 1],
        all_qubits[2 * selection_length + 2 :],
    )
    gate = cirq_qubitization.SelectedMajoranaFermionGate(
        selection_length, target_length
    )
    circuit = cirq.Circuit(
        gate.on(
            control_register=control,
            selection_register=selection,
            selection_ancilla=ancilla,
            accumulator=accumulator,
            target_register=target,
        )
    )

    sim = cirq.Simulator()
    for selection_integer in range(target_length):
        svals = [int(x) for x in format(selection_integer, f"0{selection_length}b")]
        qubit_vals = {
            x: int(x == control) for x in all_qubits
        }  # turn on control bit to activate circuit
        qubit_vals.update(
            {s: sval for s, sval in zip(selection, svals)}
        )  # Initialize selection bits appropriately

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)

        final_target_state = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=list(range(len(all_qubits) - len(target), len(all_qubits))),
        )

        expected_target_state = cirq.Circuit(
            [cirq.Z(q) for q in target[: selection_integer - 1]],
            cirq.Y(target[selection_integer]),
            [cirq.I(q) for q in target[selection_integer + 1 :]],
        ).final_state_vector(qubit_order=target)

        cirq.testing.assert_allclose_up_to_global_phase(
            expected_target_state, final_target_state, atol=1e-6
        )
