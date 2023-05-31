import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra as cirq_infra
import cirq_qubitization.cirq_infra.testing as cq_testing


@pytest.mark.parametrize("num_targets", [3, 4, 6, 8, 10])
def test_multi_target_cnot(num_targets):
    qubits = cirq.LineQubit.range(num_targets + 1)
    naive_circuit = cirq.Circuit(cirq.CNOT(qubits[0], q) for q in qubits[1:])
    op = cq.MultiTargetCNOT(num_targets).on(*qubits)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(op), naive_circuit, atol=1e-6
    )
    optimal_circuit = cirq.Circuit(cirq.decompose_once(op))
    assert len(optimal_circuit) == 2 * np.ceil(np.log2(num_targets)) + 1


@pytest.mark.parametrize('pauli, c', [(cirq.Z, '@'), (cirq.X, 'X'), (cirq.Y, 'Y')])
def test_multi_controlled_not_diagram(pauli, c):
    gate = cq.MultiControlPauli(
        [1, 0, 1, 0, 1], target_gate=pauli, mode=cq.MultiControlPauli.DecomposeMode.DIRTY_ANCILLA
    )
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
        f'''
controls0: ───────────────@───────────────────────@───────────────
                          │                       │
controls1: ───X───────────@───────────────────────@───X───────────
                          │                       │
_b0: ─────────────────@───X───@───────────────@───X───@───────────
                      │       │               │       │
controls2: ───────────@───────@───────────────@───────@───────────
                      │       │               │       │
_b1: ─────────────@───X───────X───@───────@───X───────X───@───────
                  │               │       │               │
controls3: ───X───@───────────────@───────@───────────────@───X───
                  │               │       │               │
_b2: ─────────@───X───────────────X───@───X───────────────X───────
              │                       │
controls4: ───@───────────────────────@───────────────────────────
              │                       │
target: ──────{c}───────────────────────{c}───────────────────────────
''',
        qubit_order=qubit_order,
    )


@pytest.mark.parametrize("num_controls", [*range(7, 17)])
@pytest.mark.parametrize("pauli", [cirq.X, cirq.Y, cirq.Z])
@pytest.mark.parametrize('cv', [0, 1])
@pytest.mark.parametrize(
    'mode',
    [
        cq.MultiControlPauli.DecomposeMode.DIRTY_ANCILLA,
        cq.MultiControlPauli.DecomposeMode.CLEAN_ANCILLA,
    ],
)
def test_t_complexity_mcp(
    num_controls: int, pauli: cirq.Pauli, cv: int, mode: cq.MultiControlPauli.DecomposeMode
):
    gate = cq.MultiControlPauli([cv] * num_controls, target_gate=pauli, mode=mode)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(gate)
