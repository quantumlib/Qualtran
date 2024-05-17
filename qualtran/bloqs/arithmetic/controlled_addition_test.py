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
from qualtran import QUInt
from qualtran.bloqs.arithmetic.controlled_addition import ControlledAdd
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.resource_counting.generalizers import ignore_split_join


@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize('b', [1, 2, 3])
@pytest.mark.parametrize('num_bits_a', [2, 3])
@pytest.mark.parametrize('num_bits_b', [5])
@pytest.mark.parametrize('controlled_on', [0, 1])
@pytest.mark.parametrize('control', [0, 1])
def test_controlled_addition(a, b, num_bits_a, num_bits_b, controlled_on, control):
    num_anc = num_bits_b - 1
    gate = ControlledAdd(QUInt(num_bits_a), QUInt(num_bits_b), cv=controlled_on)
    qubits = cirq.LineQubit.range(num_bits_a + num_bits_b + 1)
    op = gate.on_registers(ctrl=qubits[0], a=qubits[1 : num_bits_a + 1], b=qubits[num_bits_a + 1 :])
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    circuit0 = cirq.Circuit(op)
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (num_bits_a + num_bits_b + num_anc + 1)
    initial_state[0] = control
    initial_state[1 : num_bits_a + 1] = list(iter_bits(a, num_bits_a))
    initial_state[num_bits_a + 1 : num_bits_a + num_bits_b + 1] = list(iter_bits(b, num_bits_b))
    final_state = [0] * (num_bits_a + num_bits_b + num_anc + 1)
    final_state[0] = control
    final_state[1 : num_bits_a + 1] = list(iter_bits(a, num_bits_a))
    if control == controlled_on:
        final_state[num_bits_a + 1 : num_bits_a + num_bits_b + 1] = list(
            iter_bits(a + b, num_bits_b)
        )
    else:
        final_state[num_bits_a + 1 : num_bits_a + num_bits_b + 1] = list(iter_bits(b, num_bits_b))
    assert_circuit_inp_out_cirqsim(circuit, qubits + ancillas, initial_state, final_state)
    assert_circuit_inp_out_cirqsim(
        circuit0, qubits, initial_state[:-num_anc], final_state[:-num_anc]
    )


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_counts_controlled(n: int):
    add = ControlledAdd(QUInt(n), cv=1)
    num_and = 2 * n - 1
    t_count = 4 * num_and

    qlt_testing.assert_valid_bloq_decomposition(add)
    assert add.t_complexity() == add.decompose_bloq().t_complexity()
    assert add.bloq_counts() == add.decompose_bloq().bloq_counts(generalizer=ignore_split_join)
    assert add.t_complexity().t == t_count
