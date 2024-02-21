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

from qualtran import BoundedQUInt, Register
from qualtran._infra.gate_with_registers import get_named_qubits, total_bits
from qualtran.bloqs.selected_majorana_fermion import SelectedMajoranaFermion
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.slow
@pytest.mark.parametrize("selection_bitsize, target_bitsize", [(2, 4), (3, 8), (4, 9)])
@pytest.mark.parametrize("target_gate", [cirq.X, cirq.Y])
def test_selected_majorana_fermion_gate(selection_bitsize, target_bitsize, target_gate):
    gate = SelectedMajoranaFermion(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)),
        target_gate=target_gate,
    )
    assert_valid_bloq_decomposition(gate)

    g = GateHelper(gate)
    assert len(g.all_qubits) <= total_bits(gate.signature) + selection_bitsize + 1

    sim = cirq.Simulator(dtype=np.complex128)
    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.operation.qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in g.operation.qubits]

        result = sim.simulate(
            g.circuit, initial_state=initial_state, qubit_order=g.operation.qubits
        )

        final_target_state = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=[g.operation.qubits.index(q) for q in g.quregs['target']],
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
    gate = SelectedMajoranaFermion(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)), target_gate=cirq.X
    )
    circuit = cirq.Circuit(gate.on_registers(**get_named_qubits(gate.signature)))
    qubits = list(q for v in get_named_qubits(gate.signature).values() for q in v)
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
    gate = SelectedMajoranaFermion(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)), target_gate=cirq.X
    )
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    g = GateHelper(gate)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation, context=context))
    ancillas = sorted(set(circuit.all_qubits()) - set(g.operation.qubits))
    qubits = np.concatenate(
        [
            g.quregs['control'],
            [q for qs in zip(g.quregs['selection'], ancillas[1:]) for q in qs],
            ancillas[0:1],
            g.quregs['target'],
        ]
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
    gate = SelectedMajoranaFermion(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)), target_gate=cirq.X
    )
    op = gate.on_registers(**get_named_qubits(gate.signature))
    op2 = SelectedMajoranaFermion.make_on(target_gate=cirq.X, **get_named_qubits(gate.signature))
    assert op == op2
