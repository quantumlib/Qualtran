import cirq
import numpy as np
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra as cirq_infra
import cirq_qubitization.cirq_infra.testing as cq_testing


@pytest.mark.parametrize("num_targets", [3, 4, 6, 8, 10])
def test_multi_target_cnot(num_targets):
    qubits = cirq.LineQubit.range(num_targets + 1)
    naive_circuit = cirq.Circuit(cirq.CNOT(qubits[0], q) for q in qubits[1:])
    op = cirq_qubitization.MultiTargetCNOT(num_targets).on(*qubits)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(op), naive_circuit, atol=1e-6
    )
    optimal_circuit = cirq.Circuit(cirq.decompose_once(op))
    assert len(optimal_circuit) == 2 * np.ceil(np.log2(num_targets)) + 1


def test_multi_controlled_not_diagram():
    gate = cirq_qubitization.MultiControlNOT(5)
    with cirq_infra.memory_management_context():
        g = cq_testing.GateHelper(gate)
        circuit = cirq.Circuit(cirq.decompose_once(g.operation))
        ancilla = sorted(circuit.all_qubits() - set(g.operation.qubits))
    qubit_order = (
        g.quregs['controls'][:2]
        + [q for kv in zip(ancilla, g.quregs['controls'][2:]) for q in kv]
        + g.quregs['target']
    )
    cirq.testing.assert_has_diagram(
        circuit,
        '''
controls0: ───────────────@───────────────────────@───────────
                          │                       │
controls1: ───────────────@───────────────────────@───────────
                          │                       │
_b0: ─────────────────@───X───@───────────────@───X───@───────
                      │       │               │       │
controls2: ───────────@───────@───────────────@───────@───────
                      │       │               │       │
_b1: ─────────────@───X───────X───@───────@───X───────X───@───
                  │               │       │               │
controls3: ───────@───────────────@───────@───────────────@───
                  │               │       │               │
_b2: ─────────@───X───────────────X───@───X───────────────X───
              │                       │
controls4: ───@───────────────────────@───────────────────────
              │                       │
target: ──────X───────────────────────X───────────────────────
''',
        qubit_order=qubit_order,
    )
