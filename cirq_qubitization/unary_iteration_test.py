from typing import Tuple
import pytest
import cirq
import cirq_qubitization


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    input_states = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    output_states = [inp[:2] + (1 if inp[:2] == cv else 0,) for inp in input_states]

    circuit = cirq.Circuit(cirq_qubitization.And(cv).on(c1, c2, t))
    for input, output in zip(input_states, output_states):
        result = cirq.Simulator().simulate(circuit, initial_state=input)
        assert result.dirac_notation()[1:-1] == "".join(str(x) for x in output)


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate_adjoint(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    all_cvs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    input_states = [inp + (1 if inp == cv else 0,) for inp in all_cvs]
    output_states = [inp + (0,) for inp in all_cvs]

    circuit = cirq.Circuit(cirq_qubitization.And(cv, adjoint=True).on(c1, c2, t))
    for input, output in zip(input_states, output_states):
        result = cirq.Simulator().simulate(circuit, initial_state=input)
        assert result.dirac_notation()[1:-1] == "".join(str(x) for x in output)


def test_unary_iteration():
    def get_qubits(selection_length: int, target_length: int):
        buffer = cirq.LineQubit.range(2 * selection_length + target_length + 1)
        return (
            buffer[0],
            buffer[1 : 2 * selection_length : 2],
            buffer[2 : 2 * selection_length + 1 : 2],
            buffer[2 * selection_length + 1 :],
        )

    control, selection, ancilla, target = get_qubits(3, 5)
    circuit = cirq_qubitization.unary_iteration(
        control, selection, ancilla, cirq.X.on_each(*target[::-1])
    ).circuit
    sim = cirq.Simulator()
    for selection_integer in range(len(target)):
        svals = [int(x) for x in format(selection_integer, f"0{len(selection)}b")]
        qubit_vals = {s: sval for s, sval in zip(selection, svals)}
        qubit_vals.update({control: 1})
        qubit_vals.update({x: 0 for x in ancilla + target})
        initial_state = [qubit_vals[x] for x in sorted(circuit.all_qubits())]
        result = sim.simulate(circuit, initial_state=initial_state)
        initial_state[-(selection_integer + 1)] = 1
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output
