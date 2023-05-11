import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.reflection_using_prepare_test import (
    greedily_allocate_ancilla,
    keep,
)
from cirq_qubitization.generic_select_test import get_1d_ising_hamiltonian


def walk_operator_for_pauli_hamiltonian(
    ham: cirq.PauliSum, eps: float
) -> cq.QubitizationWalkOperator:
    q = sorted(ham.qubits)
    ham_dps = [ps.dense(q) for ps in ham]
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    prepare = cq.StatePreparationAliasSampling(ham_coeff, probability_epsilon=eps)
    select = cq.GenericSelect(
        prepare.selection_registers.bitsize, select_unitaries=ham_dps, target_bitsize=len(q)
    )
    return cq.QubitizationWalkOperator(select=select, prepare=prepare)


def get_walk_operator_for_1d_ising_model(num_sites: int, eps: float) -> cq.QubitizationWalkOperator:
    ham = get_1d_ising_hamiltonian(cirq.LineQubit.range(num_sites))
    return walk_operator_for_pauli_hamiltonian(ham, eps)


@pytest.mark.parametrize('num_sites,eps', [(4, 2e-1), (3, 1e-1)])
def test_qubitization_walk_operator(num_sites: int, eps: float):
    ham = get_1d_ising_hamiltonian(cirq.LineQubit.range(num_sites))
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    qubitization_lambda = np.sum(ham_coeff)

    walk = walk_operator_for_pauli_hamiltonian(ham, eps)

    g = cq_testing.GateHelper(walk)
    with cq.memory_management_context():
        walk_circuit = cirq.Circuit(cirq.decompose(g.operation, keep=keep, on_stuck_raise=None))

    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[: len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)

    greedy_mm = cq.GreedyQubitManager('ancilla', maximize_reuse=True)
    walk_circuit = cq.map_clean_and_borrowable_qubits(walk_circuit, qm=greedy_mm)
    assert len(walk_circuit.all_qubits()) < 23
    qubit_order = cirq.QubitOrder.explicit(
        [*g.quregs['selection'], *g.quregs['target']], fallback=cirq.QubitOrder.DEFAULT
    )

    sim = cirq.Simulator(dtype=np.complex128)

    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())
    for eig_idx, eig_val in enumerate(eigen_values):
        # Applying the walk operator W on an initial state |L>|k>
        K_state = eigen_vectors[:, eig_idx].flatten()
        prep_L_K = cirq.Circuit(
            cirq.StatePreparationChannel(L_state, name="PREP_L").on(*g.quregs['selection']),
            cirq.StatePreparationChannel(K_state, name="PREP_K").on(*g.quregs['target']),
        )
        # Initial state: |L>|k>
        L_K = sim.simulate(prep_L_K, qubit_order=qubit_order).final_state_vector

        prep_walk_circuit = prep_L_K + walk_circuit
        # Final state: W|L>|k>|temp> with |temp> register traced out.
        final_state = sim.simulate(prep_walk_circuit, qubit_order=qubit_order).final_state_vector
        final_state = final_state.reshape(len(L_K), -1).sum(axis=1)

        # Overlap: <L|k|W|k|L> = E_{k} / lambda
        overlap = np.vdot(L_K, final_state)
        cirq.testing.assert_allclose_up_to_global_phase(
            overlap, eig_val / qubitization_lambda, atol=1e-6
        )


def test_qubitization_walk_operator_diagrams():
    num_sites, eps = 4, 1e-1
    walk = get_walk_operator_for_1d_ising_model(num_sites, eps)
    # 1. Diagram for $W = SELECT.R_{L}$
    qu_regs = walk.registers.get_named_qubits()
    walk_op = walk.on_registers(**qu_regs)
    circuit = cirq.Circuit(cirq.decompose_once(walk_op))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ───In──────────────R_L───
               │               │
selection1: ───In──────────────R_L───
               │               │
selection2: ───In──────────────R_L───
               │
target0: ──────GenericSelect─────────
               │
target1: ──────GenericSelect─────────
               │
target2: ──────GenericSelect─────────
               │
target3: ──────GenericSelect─────────
''',
    )
    # 2. Diagram for $W^{2} = SELECT.R_{L}.SELCT.R_{L}$
    walk_squared_op = walk.with_power(2).on_registers(**qu_regs)
    circuit = cirq.Circuit(cirq.decompose_once(walk_squared_op))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ───In──────────────R_L───In──────────────R_L───
               │               │     │               │
selection1: ───In──────────────R_L───In──────────────R_L───
               │               │     │               │
selection2: ───In──────────────R_L───In──────────────R_L───
               │                     │
target0: ──────GenericSelect─────────GenericSelect─────────
               │                     │
target1: ──────GenericSelect─────────GenericSelect─────────
               │                     │
target2: ──────GenericSelect─────────GenericSelect─────────
               │                     │
target3: ──────GenericSelect─────────GenericSelect─────────
''',
    )
    # 3. Diagram for $Ctrl-W = Ctrl-SELECT.Ctrl-R_{L}$
    controlled_walk_op = walk.controlled().on_registers(**qu_regs, control=cirq.q('control'))
    circuit = cirq.Circuit(cirq.decompose_once(controlled_walk_op))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
control: ──────@───────────────@─────
               │               │
selection0: ───In──────────────R_L───
               │               │
selection1: ───In──────────────R_L───
               │               │
selection2: ───In──────────────R_L───
               │
target0: ──────GenericSelect─────────
               │
target1: ──────GenericSelect─────────
               │
target2: ──────GenericSelect─────────
               │
target3: ──────GenericSelect─────────
''',
    )
    # 4. Diagram for $Ctrl-W = Ctrl-SELECT.Ctrl-R_{L}$ in terms of $Ctrl-SELECT$ and $PREPARE$.
    gateset_to_keep = cirq.Gateset(
        cq.GenericSelect, cq.StatePreparationAliasSampling, cq.MultiControlPauli, cirq.X
    )

    def keep(op):
        ret = op in gateset_to_keep
        if op.gate is not None and isinstance(op.gate, cirq.ops.raw_types._InverseCompositeGate):
            ret |= op.gate._original in gateset_to_keep
        return ret

    circuit = cirq.Circuit(cirq.decompose(controlled_walk_op, keep=keep, on_stuck_raise=None))
    circuit = greedily_allocate_ancilla(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        '''
ancilla_0: ────────────────────sigma_mu───────────────────────────────sigma_mu────────────────────────
                               │                                      │
ancilla_1: ────────────────────alt────────────────────────────────────alt─────────────────────────────
                               │                                      │
ancilla_2: ────────────────────alt────────────────────────────────────alt─────────────────────────────
                               │                                      │
ancilla_3: ────────────────────alt────────────────────────────────────alt─────────────────────────────
                               │                                      │
ancilla_4: ────────────────────keep───────────────────────────────────keep────────────────────────────
                               │                                      │
ancilla_5: ────────────────────less_than_equal────────────────────────less_than_equal─────────────────
                               │                                      │
control: ──────@───────────────┼───────────────────────────────Z──────┼───────────────────────────────
               │               │                               │      │
selection0: ───In──────────────StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───
               │               │                               │      │
selection1: ───In──────────────selection───────────────────────@(0)───selection───────────────────────
               │               │                               │      │
selection2: ───In──────────────selection^-1────────────────────@(0)───selection───────────────────────
               │
target0: ──────GenericSelect──────────────────────────────────────────────────────────────────────────
               │
target1: ──────GenericSelect──────────────────────────────────────────────────────────────────────────
               │
target2: ──────GenericSelect──────────────────────────────────────────────────────────────────────────
               │
target3: ──────GenericSelect──────────────────────────────────────────────────────────────────────────''',
    )
