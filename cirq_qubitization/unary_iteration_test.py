from typing import Sequence
import cirq
import cirq_qubitization


class ApplyXToLthQubit(cirq_qubitization.UnaryIterationGate):
    def __init__(self, selection_length: int, target_length: int):
        self.selection_length = selection_length
        self.target_length = target_length

    @property
    def control_register(self) -> int:
        return 1

    @property
    def selection_register(self) -> int:
        return self.selection_length

    @property
    def target_register(self) -> int:
        return self.target_length

    @property
    def iteration_length(self) -> int:
        return self.target_length

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(n + 1)])


def test_unary_iteration():
    selection_length = 3
    target_length = 5
    all_qubits = cirq.LineQubit.range(2 * selection_length + target_length + 1)
    control, selection, ancilla, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_length : 2],
        all_qubits[2 : 2 * selection_length + 1 : 2],
        all_qubits[2 * selection_length + 1 :],
    )

    circuit = cirq.Circuit(
        ApplyXToLthQubit(3, 5).on(control, *selection, *ancilla, *target)
    )
    sim = cirq.Simulator()
    for selection_integer in range(target_length):
        svals = [int(x) for x in format(selection_integer, f"0{selection_length}b")]
        qubit_vals = {x: int(x == control) for x in all_qubits}  # turn on control bit to activate circuit
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})  # Initialize selection bits appropriately

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)
        initial_state[-(selection_integer + 1)] = 1  # Build correct statevector with selection_integer bit flipped in the target register
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output
