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
import pytest

import qualtran.testing as qlt_testing
from qualtran import BoundedQUInt, QUInt, Register, Signature
from qualtran._infra.gate_with_registers import get_named_qubits, total_bits
from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import (
    _apply_z_to_odd,
    ApplyGateToLthQubit,
)
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper


def test_apply_z_to_odd(bloq_autotester):
    bloq_autotester(_apply_z_to_odd)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('apply_gate_to_lth_target')


@pytest.mark.parametrize("selection_bitsize,target_bitsize", [[3, 5], [3, 7], [4, 5]])
def test_apply_gate_to_lth_qubit(selection_bitsize, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyGateToLthQubit(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)), lambda _: cirq.X
    )
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    # Upper bounded because not all ancillas may be used as part of unary iteration.
    assert (
        len(g.all_qubits)
        <= target_bitsize + 2 * (selection_bitsize + total_bits(gate.control_registers)) - 1
    )

    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], QUInt(selection_bitsize).to_bits(n)))

        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals[g.quregs['target'][n]] = 1
        final_state = [qubit_vals[x] for x in g.all_qubits]
        assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )


def test_apply_gate_to_lth_qubit_diagram():
    # Apply Z gate to all odd targets and Identity to even targets.
    gate = ApplyGateToLthQubit(
        Register('selection', BoundedQUInt(3, 5)),
        lambda n: cirq.Z if n & 1 else cirq.I,
        control_regs=Signature.build(control=2),
    )
    circuit = cirq.Circuit(gate.on_registers(**get_named_qubits(gate.signature)))
    qubits = list(q for v in get_named_qubits(gate.signature).values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control0: ─────@────
               │
control1: ─────@────
               │
selection0: ───In───
               │
selection1: ───In───
               │
selection2: ───In───
               │
target0: ──────I────
               │
target1: ──────Z────
               │
target2: ──────I────
               │
target3: ──────Z────
               │
target4: ──────I────
""",
        qubit_order=qubits,
    )


def test_apply_gate_to_lth_qubit_make_on():
    gate = ApplyGateToLthQubit(
        Register('selection', BoundedQUInt(3, 5)),
        lambda n: cirq.Z if n & 1 else cirq.I,
        control_regs=Signature.build(control=2),
    )
    op = gate.on_registers(**get_named_qubits(gate.signature))
    op2 = ApplyGateToLthQubit.make_on(
        nth_gate=lambda n: cirq.Z if n & 1 else cirq.I,
        **get_named_qubits(gate.signature),  # type: ignore[arg-type]
    )
    # Note: ApplyGateToLthQubit doesn't support value equality.
    assert op.qubits == op2.qubits
    assert isinstance(op.gate, ApplyGateToLthQubit)
    assert isinstance(op2.gate, ApplyGateToLthQubit)
    assert op.gate.selection_regs == op2.gate.selection_regs
    assert op.gate.control_regs == op2.gate.control_regs


@pytest.mark.parametrize("selection_bitsize,target_bitsize", [[3, 5], [3, 7], [4, 5]])
def test_bloq_has_consistent_decomposition(selection_bitsize, target_bitsize):
    bloq = ApplyGateToLthQubit(
        Register('selection', BoundedQUInt(selection_bitsize, target_bitsize)),
        lambda n: cirq.Z if n & 1 else cirq.I,
        control_regs=Signature.build(control=2),
    )
    qlt_testing.assert_valid_bloq_decomposition(bloq)
