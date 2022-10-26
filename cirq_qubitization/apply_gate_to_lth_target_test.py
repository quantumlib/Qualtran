import cirq
import pytest

import cirq_qubitization
from cirq_qubitization import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits


@pytest.mark.parametrize("selection_bitsize,target_bitsize", [[3, 5], [3, 7], [4, 5]])
def test_apply_gate_to_lth_qubit(selection_bitsize, target_bitsize):
    gate = cirq_qubitization.ApplyGateToLthQubit(
        selection_bitsize, target_bitsize, lambda _: cirq.X
    )
    q = gate.registers.get_named_qubits()
    circuit = cirq.Circuit(gate.on_registers(**q))
    all_qubits = sorted(circuit.all_qubits())
    (control,), selection, ancilla, target = (
        q["control"],
        q["selection"],
        q["ancilla"],
        q["target"],
    )
    for n in range(len(target)):
        svals = iter_bits(n, len(selection))
        # turn on control bit to activate circuit:
        qubit_vals = {x: int(x == control) for x in all_qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        final_state = initial_state.copy()
        final_state[-(n + 1)] = 1
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_apply_gate_to_lth_qubit_diagram():
    # Apply Z gate to all odd targets and Identity to even targets.
    gate = cirq_qubitization.ApplyGateToLthQubit(
        3, 5, lambda n: cirq.Z if n & 1 else cirq.I, control_bitsize=2
    )
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.get_named_qubits()))
    qubits = list(q for v in gate.registers.get_named_qubits().values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control0: ─────@─────
               │
control1: ─────@─────
               │
selection0: ───In────
               │
selection1: ───In────
               │
selection2: ───In────
               │
ancilla0: ─────Anc───
               │
ancilla1: ─────Anc───
               │
ancilla2: ─────Anc───
               │
ancilla3: ─────Anc───
               │
target0: ──────I─────
               │
target1: ──────Z─────
               │
target2: ──────I─────
               │
target3: ──────Z─────
               │
target4: ──────I─────
    """,
        qubit_order=qubits,
    )


def test_notebook():
    cq_testing.execute_notebook('apply_gate_to_lth_target', subdir='gates')
