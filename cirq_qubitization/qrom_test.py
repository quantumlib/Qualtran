import pytest
import cirq
import cirq_qubitization
import itertools


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]]])
def test_qrom(data):
    qrom = cirq_qubitization.QROM(*data)
    all_qubits = cirq.LineQubit.range(qrom.num_qubits())
    selection, ancilla, flat_target = (
        all_qubits[: 2 * qrom.selection_bitsize - 1 : 2],
        all_qubits[1 : 2 * qrom.selection_bitsize : 2],
        all_qubits[2 * qrom.selection_bitsize :],
    )
    target_lengths = [max(d).bit_length() for d in data]
    target = [
        flat_target[y - x : y] for x, y in zip(target_lengths, itertools.accumulate(target_lengths))
    ]
    circuit = cirq.Circuit(
        qrom.on_registers(
            selection_register=selection,
            selection_ancilla=ancilla,
            target_register=target if len(target) > 1 else flat_target,
        )
    )

    sim = cirq.Simulator()
    for selection_integer in range(qrom.iteration_length):
        svals = [int(x) for x in format(selection_integer, f"0{qrom.selection_bitsize}b")]
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)

        start = 2 * qrom.selection_bitsize
        for d, d_bits in zip(data, target_lengths):
            end = start + d_bits
            initial_state[start:end] = [
                int(x) for x in format(d[selection_integer], f"0{end - start}b")
            ]
            start = end
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output
