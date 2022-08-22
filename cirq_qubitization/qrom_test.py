import cirq
import pytest

import cirq_qubitization
from cirq_qubitization import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]]])
def test_qrom(data):
    qrom = cirq_qubitization.QROM(*data)
    qubit_regs = qrom.registers.get_named_qubits()
    all_qubits = qrom.registers.merge_qubits(**qubit_regs)
    selection = qubit_regs["selection"]
    targets = [qubit_regs[f"target{i}"] for i in range(len(data))]
    circuit = cirq.Circuit(qrom.on_registers(**qubit_regs))

    for selection_integer in range(qrom.iteration_length):
        svals = list(iter_bits(selection_integer, len(selection)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, data):
            for q, b in zip(target, iter_bits(d[selection_integer], len(target))):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_qrom_repr():
    qrom = cirq_qubitization.QROM([1, 2], [3, 5])
    cirq.testing.assert_equivalent_repr(qrom, setup_code="import cirq_qubitization\n")
