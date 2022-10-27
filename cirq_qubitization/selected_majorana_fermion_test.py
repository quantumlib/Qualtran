import cirq
import pytest

import cirq_qubitization


@pytest.mark.parametrize("selection_bitsize, target_bitsize", [(2, 4), (3, 8), (4, 9)])
@pytest.mark.parametrize("target_gate", [cirq.X, cirq.Y])
def test_selected_majorana_fermion_gate(selection_bitsize, target_bitsize, target_gate):
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 2)
    control, selection, ancilla, accumulator, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_bitsize : 2],
        all_qubits[2 : 2 * selection_bitsize + 1 : 2],
        all_qubits[2 * selection_bitsize + 1],
        all_qubits[2 * selection_bitsize + 2 :],
    )
    gate = cirq_qubitization.SelectedMajoranaFermionGate(
        selection_bitsize, target_bitsize, target_gate=target_gate
    )
    circuit = cirq.Circuit(
        gate.on_registers(
            control=control,
            selection=selection,
            ancilla=ancilla,
            accumulator=accumulator,
            target=target,
        )
    )

    sim = cirq.Simulator()
    for selection_integer in range(target_bitsize):
        svals = [int(x) for x in format(selection_integer, f"0{selection_bitsize}b")]
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
            target_gate(target[selection_integer]),
            [cirq.I(q) for q in target[selection_integer + 1 :]],
        ).final_state_vector(qubit_order=target)

        cirq.testing.assert_allclose_up_to_global_phase(
            expected_target_state, final_target_state, atol=1e-6
        )


def test_selected_majorana_fermion_gate_diagram():
    selection_bitsize, target_bitsize = 3, 5
    gate = cirq_qubitization.SelectedMajoranaFermionGate(
        selection_bitsize, target_bitsize, target_gate=cirq.X
    )
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.get_named_qubits()))
    qubits = list(q for v in gate.registers.get_named_qubits().values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control: ───────@─────
                │
selection0: ────In────
                │
selection1: ────In────
                │
selection2: ────In────
                │
ancilla0: ──────Anc───
                │
ancilla1: ──────Anc───
                │
ancilla2: ──────Anc───
                │
target0: ───────ZX────
                │
target1: ───────ZX────
                │
target2: ───────ZX────
                │
target3: ───────ZX────
                │
target4: ───────ZX────
                │
accumulator: ───Acc───
    """,
        qubit_order=qubits,
    )
