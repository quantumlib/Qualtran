import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.cirq_infra import testing as cq_testing


@pytest.mark.parametrize("selection_bitsize, target_bitsize", [(2, 4), (3, 8), (4, 9)])
@pytest.mark.parametrize("target_gate", [cirq.X, cirq.Y])
def test_selected_majorana_fermion_gate(selection_bitsize, target_bitsize, target_gate):
    greedy_mm = cq.cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = cq.SelectedMajoranaFermionGate(
        cq.SelectionRegisters.build(selection=(selection_bitsize, target_bitsize)),
        target_gate=target_gate,
    )
    g = cq_testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= gate.registers.bitsize + selection_bitsize + 1

    sim = cirq.Simulator(dtype=np.complex128)
    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # All controls 'on' to activate circuit
        qubit_vals |= {c: 1 for c in g.quregs['control']}
        # Set selection according to `n`
        qubit_vals |= zip(g.quregs['selection'], iter_bits(n, selection_bitsize))

        initial_state = [qubit_vals[x] for x in g.all_qubits]

        result = sim.simulate(
            g.decomposed_circuit, initial_state=initial_state, qubit_order=g.all_qubits
        )

        final_target_state = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=[g.all_qubits.index(q) for q in g.quregs['target']],
        )

        expected_target_state = cirq.Circuit(
            [cirq.Z(q) for q in g.quregs['target'][:n]],
            target_gate(g.quregs['target'][n]),
            [cirq.I(q) for q in g.quregs['target'][n + 1 :]],
        ).final_state_vector(qubit_order=g.quregs['target'])

        cirq.testing.assert_allclose_up_to_global_phase(
            expected_target_state, final_target_state, atol=1e-6
        )


def test_selected_majorana_fermion_gate_diagram():
    selection_bitsize, target_bitsize = 3, 5
    gate = cq.SelectedMajoranaFermionGate(
        cq.SelectionRegisters.build(selection=(selection_bitsize, target_bitsize)),
        target_gate=cirq.X,
    )
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.get_named_qubits()))
    qubits = list(q for v in gate.registers.get_named_qubits().values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control: ──────@────
               │
selection0: ───In───
               │
selection1: ───In───
               │
selection2: ───In───
               │
target0: ──────ZX───
               │
target1: ──────ZX───
               │
target2: ──────ZX───
               │
target3: ──────ZX───
               │
target4: ──────ZX───
""",
        qubit_order=qubits,
    )


def test_selected_majorana_fermion_gate_decomposed_diagram():
    selection_bitsize, target_bitsize = 2, 3
    gate = cq.SelectedMajoranaFermionGate(
        cq.SelectionRegisters.build(selection=(selection_bitsize, target_bitsize)),
        target_gate=cirq.X,
    )
    greedy_mm = cq.cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    g = cq_testing.GateHelper(gate)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation, context=context))
    ancillas = sorted(set(circuit.all_qubits()) - set(g.operation.qubits))
    qubits = (
        g.quregs['control']
        + [q for qs in zip(g.quregs['selection'], ancillas[1:]) for q in qs]
        + ancillas[0:1]
        + g.quregs['target']
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
control: ──────@───@──────────────────────────────────────@───────────@──────
               │   │                                      │           │
selection0: ───┼───(0)────────────────────────────────────┼───────────@──────
               │   │                                      │           │
_a_1: ─────────┼───And───@─────────────@───────────@──────X───@───@───And†───
               │         │             │           │          │   │
selection1: ───┼─────────(0)───────────┼───────────@──────────┼───┼──────────
               │         │             │           │          │   │
_a_2: ─────────┼─────────And───@───@───X───@───@───And†───────┼───┼──────────
               │               │   │       │   │              │   │
_a_0: ─────────X───────────────X───┼───@───X───┼───@──────────X───┼───@──────
                                   │   │       │   │              │   │
target0: ──────────────────────────X───@───────┼───┼──────────────┼───┼──────
                                               │   │              │   │
target1: ──────────────────────────────────────X───@──────────────┼───┼──────
                                                                  │   │
target2: ─────────────────────────────────────────────────────────X───@──────    """,
        qubit_order=qubits,
    )


def test_selected_majorana_fermion_gate_make_on():
    selection_bitsize, target_bitsize = 3, 5
    gate = cq.SelectedMajoranaFermionGate(
        cq.SelectionRegisters.build(selection=(selection_bitsize, target_bitsize)),
        target_gate=cirq.X,
    )
    op = gate.on_registers(**gate.registers.get_named_qubits())
    op2 = cq.SelectedMajoranaFermionGate.make_on(
        target_gate=cirq.X, **gate.registers.get_named_qubits()
    )
    assert op == op2
