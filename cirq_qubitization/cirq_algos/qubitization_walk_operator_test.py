import cirq

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.reflection_using_prepare_test import greedily_allocate_ancilla
from cirq_qubitization.generic_select_test import get_1d_ising_hamiltonian


def get_walk_operator_for_1d_ising_model(num_sites: int, eps: float):
    q = cirq.LineQubit.range(num_sites)
    ham = get_1d_ising_hamiltonian(q)
    ham_dps = [ps.dense(q) for ps in ham]
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    prepare = cq.StatePreparationAliasSampling(ham_coeff, probability_epsilon=eps)
    select = cq.GenericSelect(
        prepare.selection_registers.bitsize, select_unitaries=ham_dps, target_bitsize=num_sites
    )
    walk = cq.QubitizationWalkOperator(select=select, prepare=prepare)
    return walk


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
ancilla_0: ────────────────────sigma_mu────────────────────────────────────sigma_mu────────────────────────
                               │                                           │
ancilla_1: ────────────────────alt─────────────────────────────────────────alt─────────────────────────────
                               │                                           │
ancilla_2: ────────────────────alt─────────────────────────────────────────alt─────────────────────────────
                               │                                           │
ancilla_3: ────────────────────alt─────────────────────────────────────────alt─────────────────────────────
                               │                                           │
ancilla_4: ────────────────────keep────────────────────────────────────────keep────────────────────────────
                               │                                           │
ancilla_5: ────────────────────less_than_equal─────────────────────────────less_than_equal─────────────────
                               │                                           │
control: ──────@───────────────┼───────────────────────────────────Z───────┼───────────────────────────────
               │               │                                   │       │
selection0: ───In──────────────StatePreparationAliasSampling───X───@───X───StatePreparationAliasSampling───
               │               │                                   │       │
selection1: ───In──────────────selection───────────────────────X───@───X───selection───────────────────────
               │               │                                   │       │
selection2: ───In──────────────selection^-1────────────────────X───@───X───selection───────────────────────
               │
target0: ──────GenericSelect───────────────────────────────────────────────────────────────────────────────
               │
target1: ──────GenericSelect───────────────────────────────────────────────────────────────────────────────
               │
target2: ──────GenericSelect───────────────────────────────────────────────────────────────────────────────
               │
target3: ──────GenericSelect───────────────────────────────────────────────────────────────────────────────
''',
    )
