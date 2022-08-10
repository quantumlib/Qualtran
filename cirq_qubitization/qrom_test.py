import pytest
import cirq
import cirq_qubitization
import itertools


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]]])
def test_qrom(data):
    qrom = cirq_qubitization.QROM(*data)
    qubit_regs = qrom.registers.get_named_qubits()
    all_qubits = qrom.registers.merge_qubits(**qubit_regs)
    selection, ancilla = qubit_regs["selection"], qubit_regs["ancilla"]
    targets = [qubit_regs[f"target{i}"] for i in range(len(data))]
    circuit = cirq.Circuit(qrom.on_registers(**qubit_regs))

    sim = cirq.Simulator()
    for selection_integer in range(qrom.iteration_length):
        svals = [int(x) for x in format(selection_integer, f"0{len(selection)}b")]
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state, qubit_order=all_qubits)

        for target, d in zip(targets, data):
            for q, b in zip(target, f"{d[selection_integer]:0{len(target)}b}"):
                qubit_vals[q] = int(b)
        final_state = [qubit_vals[x] for x in all_qubits]
        expected_output = "".join(str(x) for x in final_state)
        assert result.dirac_notation()[1:-1] == expected_output
