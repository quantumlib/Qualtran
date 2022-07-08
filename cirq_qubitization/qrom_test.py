import cirq
import cirq_qubitization


def test_qrom():
    data = [1, 2, 3, 4, 5]
    qrom = cirq_qubitization.QROM(data)
    all_qubits = cirq.LineQubit.range(qrom.num_qubits())
    control, selection, ancilla, target = (
        all_qubits[0],
        all_qubits[1 : 2 * qrom.selection_register : 2],
        all_qubits[2 : 2 * qrom.selection_register + 1 : 2],
        all_qubits[2 * qrom.selection_register + 1 :],
    )
    circuit = cirq.Circuit(qrom.on(control, *selection, *ancilla, *target))

    sim = cirq.Simulator()
    for selection_integer in range(qrom.iteration_length):
        svals = [
            int(x) for x in format(selection_integer, f"0{qrom.selection_register}b")
        ]
        qubit_vals = {x: int(x == control) for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)
        initial_state[2 * qrom.selection_register + 1 :] = [
            int(x)
            for x in format(data[selection_integer], f"0{qrom.selection_register}b")
        ]
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output
