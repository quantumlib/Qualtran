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

import itertools
from typing import Optional

import cirq
import numpy as np
import pytest
from numpy.typing import NDArray

from qualtran import Adjoint, Bloq
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.arithmetic import LessThanConstant, LessThanEqual
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.mcmt import MultiControlPauli, MultiTargetCNOT
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.reflections.reflection_using_prepare import (
    _refl_around_zero,
    _refl_using_prep,
    ReflectionUsingPrepare,
)
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.cirq_interop import BloqAsCirqGate
from qualtran.cirq_interop.testing import GateHelper
from qualtran.resource_counting.generalizers import (
    ignore_alloc_free,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

gateset_to_keep = cirq.Gateset(
    And,
    LessThanConstant,
    LessThanEqual,
    CSwap,
    MultiTargetCNOT,
    MultiControlPauli,
    cirq.H,
    cirq.CCNOT,
    cirq.CNOT,
    cirq.FREDKIN,
    cirq.ControlledGate,
    cirq.AnyUnitaryGateFamily(1),
)


def test_reflection_using_prepare_examples(bloq_autotester):
    bloq_autotester(_refl_using_prep)
    bloq_autotester(_refl_around_zero)


def keep(op: cirq.Operation):
    ret = op in gateset_to_keep
    if op.gate is not None and isinstance(op.gate, cirq.ops.raw_types._InverseCompositeGate):
        ret |= op.gate._original in gateset_to_keep
    if op.gate is not None and isinstance(op.gate, Adjoint):
        subgate = (
            op.gate.subbloq
            if isinstance(op.gate.subbloq, cirq.Gate)
            else BloqAsCirqGate(op.gate.subbloq)
        )
        ret |= subgate in gateset_to_keep
    return ret


def construct_gate_helper_and_qubit_order(gate, decompose_once: bool = False):
    g = GateHelper(gate)
    greedy_mm = cirq.GreedyQubitManager(prefix="ancilla", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    if decompose_once:
        circuit = cirq.Circuit(cirq.decompose_once(g.operation, context=context))
    else:
        circuit = cirq.Circuit(
            cirq.decompose(g.operation, keep=keep, on_stuck_raise=None, context=context)
        )
    ordered_input = list(itertools.chain(*g.quregs.values()))
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    assert len(circuit.all_qubits()) < 24
    return g, qubit_order, circuit


def get_3q_uniform_dirac_notation(signs, global_phase: complex = 1):
    coeff = str(0.35 * global_phase * np.sign(global_phase))
    terms = [
        f'{coeff}|000⟩',
        f'{coeff}|001⟩',
        f'{coeff}|010⟩',
        f'{coeff}|011⟩',
        f'{coeff}|100⟩',
        f'{coeff}|101⟩',
        f'{coeff}|110⟩',
        f'{coeff}|111⟩',
    ]
    ret = terms[0] if signs[0] == '+' else f'-{terms[0]}'
    for c, term in zip(signs[1:], terms[1:]):
        ret = ret + f' {c} {term}'
    return ret


@pytest.mark.slow
@pytest.mark.parametrize('num_ones', [5])
@pytest.mark.parametrize('eps', [0.05])
@pytest.mark.parametrize('global_phase', [+1, -1j])
def test_reflection_using_prepare(num_ones, eps, global_phase):
    data = [1] * num_ones
    prepare_gate = StatePreparationAliasSampling.from_probabilities(data, precision=eps)

    gate = ReflectionUsingPrepare(prepare_gate, global_phase=global_phase)
    assert_valid_bloq_decomposition(gate)

    g, qubit_order, decomposed_circuit = construct_gate_helper_and_qubit_order(gate)

    initial_state_prep = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    initial_state = cirq.dirac_notation(initial_state_prep.final_state_vector())
    assert initial_state == get_3q_uniform_dirac_notation('++++++++')
    result = cirq.Simulator(dtype=np.complex64).simulate(
        initial_state_prep + decomposed_circuit, qubit_order=qubit_order
    )
    selection = g.quregs['selection']
    prepared_state = result.final_state_vector.reshape(2 ** len(selection), -1).sum(axis=1)
    if np.sign(global_phase) == 1:
        signs = '-' * num_ones + '+' * (9 - num_ones)
    elif np.sign(global_phase) == -1:
        signs = '+' * num_ones + '-' * (9 - num_ones)
    assert cirq.dirac_notation(prepared_state) == get_3q_uniform_dirac_notation(signs, global_phase)


def test_reflection_using_prepare_diagram():
    data = [1, 2, 3, 4, 5, 6]
    eps = 2.1
    prepare_gate = StatePreparationAliasSampling.from_probabilities(data, precision=eps)
    # No control
    gate = ReflectionUsingPrepare(prepare_gate, control_val=None)
    # op = gate.on_registers(**get_named_qubits(gate.signature))
    g, qubit_order, circuit = construct_gate_helper_and_qubit_order(gate, decompose_once=True)
    cirq.testing.assert_has_diagram(
        circuit,
        '''
ancilla_0: ────X───────────────────────────────Z──────X───────────────────────────────
                                               │
ancilla_1: ────sigma_mu────────────────────────┼──────sigma_mu────────────────────────
               │                               │      │
ancilla_2: ────alt─────────────────────────────┼──────alt─────────────────────────────
               │                               │      │
ancilla_3: ────alt─────────────────────────────┼──────alt─────────────────────────────
               │                               │      │
ancilla_4: ────alt─────────────────────────────┼──────alt─────────────────────────────
               │                               │      │
ancilla_5: ────keep────────────────────────────┼──────keep────────────────────────────
               │                               │      │
ancilla_6: ────less_than_equal─────────────────┼──────less_than_equal─────────────────
               │                               │      │
selection0: ───StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───
               │                               │      │
selection1: ───selection───────────────────────@(0)───selection───────────────────────
               │                               │      │
selection2: ───selection^-1────────────────────@(0)───selection───────────────────────
''',
    )

    # Control on `|1>` state
    gate = ReflectionUsingPrepare(prepare_gate, control_val=1)
    g, qubit_order, circuit = construct_gate_helper_and_qubit_order(gate, decompose_once=True)
    cirq.testing.assert_has_diagram(
        circuit,
        '''
ancilla_0: ────sigma_mu───────────────────────────────sigma_mu────────────────────────
               │                                      │
ancilla_1: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_2: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_3: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_4: ────keep───────────────────────────────────keep────────────────────────────
               │                                      │
ancilla_5: ────less_than_equal────────────────────────less_than_equal─────────────────
               │                                      │
control: ──────┼───────────────────────────────Z──────┼───────────────────────────────
               │                               │      │
selection0: ───StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───
               │                               │      │
selection1: ───selection───────────────────────@(0)───selection───────────────────────
               │                               │      │
selection2: ───selection^-1────────────────────@(0)───selection───────────────────────
''',
    )

    # Control on `|0>` state
    gate = ReflectionUsingPrepare(prepare_gate, control_val=0)
    g, qubit_order, circuit = construct_gate_helper_and_qubit_order(gate, decompose_once=True)
    cirq.testing.assert_has_diagram(
        circuit,
        '''
               ┌──────────────────────────────┐          ┌──────────────────────────────┐
ancilla_0: ─────sigma_mu───────────────────────────────────sigma_mu─────────────────────────
                │                                          │
ancilla_1: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_2: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_3: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_4: ─────keep───────────────────────────────────────keep─────────────────────────────
                │                                          │
ancilla_5: ─────less_than_equal────────────────────────────less_than_equal──────────────────
                │                                          │
control: ───────┼────────────────────────────X────Z───────X┼────────────────────────────────
                │                                 │        │
selection0: ────StatePreparationAliasSampling─────@(0)─────StatePreparationAliasSampling────
                │                                 │        │
selection1: ────selection─────────────────────────@(0)─────selection────────────────────────
                │                                 │        │
selection2: ────selection^-1──────────────────────@(0)─────selection────────────────────────
               └──────────────────────────────┘          └──────────────────────────────┘
''',
    )


def test_reflection_using_prepare_consistent_protocols_and_controlled():
    prepare_gate = StatePreparationAliasSampling.from_probabilities([1, 2, 3, 4], precision=0.1)
    # No control
    gate = ReflectionUsingPrepare(prepare_gate, control_val=None)
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


def test_reflection_around_zero():
    def ref_state(nqubits: int) -> NDArray:
        zero = np.zeros(shape=(2**nqubits, 2**nqubits))
        zero[0, 0] = 2.0
        zero -= np.eye(2**nqubits)
        return zero

    # Check the tensor is 2|0><0| - 1
    bitsizes = (1,)
    zero_prep = PrepareIdentity.from_bitsizes(bitsizes)
    bloq = ReflectionUsingPrepare(zero_prep, global_phase=-1)
    assert np.allclose(bloq.tensor_contract(), ref_state(1))
    bitsizes = (1, 2)
    zero_prep = PrepareIdentity.from_bitsizes(bitsizes)
    bloq = ReflectionUsingPrepare(zero_prep, global_phase=-1)
    assert np.allclose(bloq.tensor_contract(), ref_state(3))


@pytest.mark.parametrize('global_phase', [+1, -1j])
@pytest.mark.parametrize('control_val', [None, 0, 1])
def test_call_graph_matches_decomp(global_phase, control_val):
    data = [1] * 5
    eps = 1e-11
    prepare_gate = StatePreparationAliasSampling.from_probabilities(data, precision=0.01)

    def catch_zpow_bloq_s_gate_inv(bloq) -> Optional[Bloq]:
        # Hack to catch the fact that cirq special cases some ZPowGates
        if isinstance(bloq, ZPowGate) and np.any(np.isclose(float(bloq.exponent), [0.5, -0.5])):
            # we're already ignoring cliffords
            return None
        return bloq

    gate = ReflectionUsingPrepare(
        prepare_gate, global_phase=global_phase, eps=eps, control_val=control_val
    )
    _, cost_decomp = gate.decompose_bloq().call_graph(
        generalizer=[ignore_split_join, ignore_alloc_free, ignore_cliffords]
    )
    _, cost_call = gate.call_graph(
        generalizer=[
            ignore_split_join,
            ignore_alloc_free,
            ignore_cliffords,
            catch_zpow_bloq_s_gate_inv,
        ]
    )
    assert cost_decomp == cost_call


@pytest.mark.notebook
def test_notebook():
    execute_notebook('reflections')
