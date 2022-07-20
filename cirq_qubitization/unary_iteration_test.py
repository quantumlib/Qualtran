from typing import Sequence
import cirq
import cirq_qubitization


class ApplyXToLthQubit(cirq_qubitization.UnaryIterationGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize

    @property
    def control_bitsize(self) -> int:
        return 1

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        return self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return self._target_bitsize

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(n + 1)])


def test_unary_iteration():
    selection_bitsize = 3
    target_bitsize = 5
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, ancilla, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_bitsize : 2],
        all_qubits[2 : 2 * selection_bitsize + 1 : 2],
        all_qubits[2 * selection_bitsize + 1 :],
    )

    circuit = cirq.Circuit(
        ApplyXToLthQubit(3, 5).on(control, *selection, *ancilla, *target)
    )
    sim = cirq.Simulator()
    for n in range(len(target)):
        svals = [int(x) for x in format(n, f"0{len(selection)}b")]
        # turn on control bit to activate circuit:
        qubit_vals = {x: int(x == control) for x in all_qubits}
        # Initialize selection bits appropriately:

        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)
        # Build correct statevector with selection_integer bit flipped in the target register:
        initial_state[-(n + 1)] = 1
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output
