#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import cirq
import numpy as np
import pytest

from qualtran import Adjoint
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.chemistry.ising.walk_operator import get_walk_operator_for_1d_ising_model
from qualtran.bloqs.mcmt import MultiControlPauli
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    _thc_walk_op,
    _walk_op,
    _walk_op_chem_sparse,
)
from qualtran.bloqs.reflections.reflection_using_prepare_test import (
    construct_gate_helper_and_qubit_order,
)
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def test_qubitization_walk_operator_autotest(bloq_autotester):
    bloq_autotester(_walk_op)


def test_qubitization_walk_operator_chem_thc_autotest(bloq_autotester):
    bloq_autotester(_thc_walk_op)


def test_qubitization_walk_operator_chem_sparse_autotest(bloq_autotester):
    bloq_autotester(_walk_op_chem_sparse)


@pytest.mark.slow
@pytest.mark.parametrize('num_sites,eps', [(3, 0.5), (4, 0.5)])
def test_qubitization_walk_operator(num_sites: int, eps: float):
    walk, ham = get_walk_operator_for_1d_ising_model(num_sites, eps)
    assert_valid_bloq_decomposition(walk)

    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    assert walk.sum_of_lcu_coefficients is not None
    qubitization_lambda = float(walk.sum_of_lcu_coefficients)
    np.testing.assert_allclose(qubitization_lambda, sum(ham_coeff))

    g, qubit_order, walk_circuit = construct_gate_helper_and_qubit_order(walk)

    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[: len(ham_coeff)] = np.sqrt(np.array(ham_coeff) / qubitization_lambda)

    assert len(walk_circuit.all_qubits()) < 23

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


@pytest.mark.slow
def test_qubitization_walk_operator_adjoint():
    num_sites, eps = 3, 0.5
    walk, _ = get_walk_operator_for_1d_ising_model(num_sites, eps)
    walk_inv_tensor = walk.adjoint().tensor_contract()
    walk_adj_tensor = Adjoint(walk).tensor_contract()
    np.testing.assert_allclose(walk_inv_tensor, walk_adj_tensor, atol=1e-12)


def test_t_complexity_for_controlled_and_adjoint():
    num_sites, eps = 4, 2e-1
    walk, _ = get_walk_operator_for_1d_ising_model(num_sites, eps)
    assert walk.controlled().adjoint().t_complexity() == walk.adjoint().controlled().t_complexity()


def test_qubitization_walk_operator_diagrams():
    num_sites, eps = 4, 1e-1
    walk, _ = get_walk_operator_for_1d_ising_model(num_sites, eps)
    # 1. Diagram for $W = SELECT.R_{L}$
    g, qubit_order, walk_circuit = construct_gate_helper_and_qubit_order(walk, decompose_once=True)
    cirq.testing.assert_has_diagram(
        walk_circuit,
        '''
selection0: ───B[H]───R_L───
               │      │
selection1: ───B[H]───R_L───
               │      │
selection2: ───B[H]───R_L───
               │
target0: ──────B[H]─────────
               │
target1: ──────B[H]─────────
               │
target2: ──────B[H]─────────
               │
target3: ──────B[H]─────────
''',
    )

    # 2. Diagram for $W^{2} = B[H].R_{L}.B[H].R_{L}$
    def decompose_twice(op):
        ops = []
        for sub_op in cirq.decompose_once(op):
            ops += cirq.decompose_once(sub_op)
        return ops

    walk_squared_op = (walk**2).on_registers(**g.quregs)
    circuit = cirq.Circuit(decompose_twice(walk_squared_op))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ───B[H]───R_L───B[H]───R_L───
               │      │     │      │
selection1: ───B[H]───R_L───B[H]───R_L───
               │      │     │      │
selection2: ───B[H]───R_L───B[H]───R_L───
               │            │
target0: ──────B[H]─────────B[H]─────────
               │            │
target1: ──────B[H]─────────B[H]─────────
               │            │
target2: ──────B[H]─────────B[H]─────────
               │            │
target3: ──────B[H]─────────B[H]─────────
''',
    )
    # 3. Diagram for $Ctrl-W = Ctrl-B[H].Ctrl-R_{L}$
    controlled_walk_op = walk.controlled().on_registers(**g.quregs, control=cirq.q('control'))
    circuit = cirq.Circuit(cirq.decompose_once(controlled_walk_op))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
control: ──────@──────@─────
               │      │
selection0: ───B[H]───R_L───
               │      │
selection1: ───B[H]───R_L───
               │      │
selection2: ───B[H]───R_L───
               │
target0: ──────B[H]─────────
               │
target1: ──────B[H]─────────
               │
target2: ──────B[H]─────────
               │
target3: ──────B[H]─────────
''',
    )
    # 4. Diagram for $Ctrl-W = Ctrl-SELECT.Ctrl-R_{L}$ in terms of $Ctrl-SELECT$ and $PREPARE$.
    gateset_to_keep = cirq.Gateset(
        SelectPauliLCU, StatePreparationAliasSampling, MultiControlPauli, cirq.X
    )

    def keep(op):
        ret = op in gateset_to_keep
        if op.gate is not None and isinstance(op.gate, Adjoint):
            ret |= op.gate.subbloq in gateset_to_keep
        return ret

    greedy_mm = cirq.GreedyQubitManager(prefix="ancilla", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(
        cirq.decompose(controlled_walk_op, keep=keep, on_stuck_raise=None, context=context)
    )
    # pylint: disable=line-too-long
    cirq.testing.assert_has_diagram(
        circuit,
        '''
                                                                       ┌──────────────────────────────┐
ancilla_0: ─────────────────────sigma_mu─────────────────────────────────sigma_mu─────────────────────────
                                │                                        │
ancilla_1: ─────────────────────sigma_mu─────────────────────────────────sigma_mu─────────────────────────
                                │                                        │
ancilla_2: ─────────────────────sigma_mu─────────────────────────────────sigma_mu─────────────────────────
                                │                                        │
ancilla_3: ─────────────────────sigma_mu─────────────────────────────────sigma_mu─────────────────────────
                                │                                        │
ancilla_4: ─────────────────────sigma_mu─────────────────────────────────sigma_mu─────────────────────────
                                │                                        │
ancilla_5: ─────────────────────alt──────────────────────────────────────alt──────────────────────────────
                                │                                        │
ancilla_6: ─────────────────────alt──────────────────────────────────────alt──────────────────────────────
                                │                                        │
ancilla_7: ─────────────────────alt──────────────────────────────────────alt──────────────────────────────
                                │                                        │
ancilla_8: ─────────────────────keep─────────────────────────────────────keep─────────────────────────────
                                │                                        │
ancilla_9: ─────────────────────keep─────────────────────────────────────keep─────────────────────────────
                                │                                        │
ancilla_10: ────────────────────keep─────────────────────────────────────keep─────────────────────────────
                                │                                        │
ancilla_11: ────────────────────keep─────────────────────────────────────keep─────────────────────────────
                                │                                        │
ancilla_12: ────────────────────keep─────────────────────────────────────keep─────────────────────────────
                                │                                        │
ancilla_13: ────────────────────less_than_equal──────────────────────────less_than_equal──────────────────
                                │                                        │
control: ──────@────────────────┼───────────────────────────────Z───────Z┼────────────────────────────────
               │                │                               │        │
selection0: ───In───────────────StatePreparationAliasSampling───@(0)─────StatePreparationAliasSampling────
               │                │                               │        │
selection1: ───In───────────────selection───────────────────────@(0)─────selection────────────────────────
               │                │                               │        │
selection2: ───In───────────────selection^-1────────────────────@(0)─────selection────────────────────────
               │
target0: ──────SelectPauliLCU─────────────────────────────────────────────────────────────────────────────
               │
target1: ──────SelectPauliLCU─────────────────────────────────────────────────────────────────────────────
               │
target2: ──────SelectPauliLCU─────────────────────────────────────────────────────────────────────────────
               │
target3: ──────SelectPauliLCU─────────────────────────────────────────────────────────────────────────────
                                                                       └──────────────────────────────┘    
''',
    )
    # pylint: enable=line-too-long


def test_qubitization_walk_operator_consistent_protocols_and_controlled():
    gate, _ = get_walk_operator_for_1d_ising_model(4, 1e-1)
    op = gate.on_registers(**get_named_qubits(gate.signature))
    # Build controlled gate
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        gate.controlled(),
        gate.controlled(num_controls=1),
        gate.controlled(control_values=(1,)),
        op.controlled_by(cirq.q("control")).gate,
    )
    equals_tester.add_equality_group(
        gate.controlled(control_values=(0,)),
        gate.controlled(num_controls=1, control_values=(0,)),
        op.controlled_by(cirq.q("control"), control_values=(0,)).gate,
    )


@pytest.mark.notebook
def test_notebook():
    execute_notebook('qubitization_walk_operator')
