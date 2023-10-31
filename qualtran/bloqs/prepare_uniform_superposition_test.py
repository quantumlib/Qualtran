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

import qualtran.testing as qlt_testing
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.prepare_uniform_superposition import (
    _c_prep_uniform,
    _prep_uniform,
    PrepareUniformSuperposition,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity


def test_prep_uniform(bloq_autotester):
    bloq_autotester(_prep_uniform)


def test_c_prep_uniform(bloq_autotester):
    bloq_autotester(_c_prep_uniform)


def test_notebook():
    qlt_testing.execute_notebook('prepare_uniform_superposition')


@pytest.mark.parametrize("n", [*range(3, 20), 25, 41])
@pytest.mark.parametrize("num_controls", [0, 1])
def test_prepare_uniform_superposition(n, num_controls):
    gate = PrepareUniformSuperposition(n, cvs=[1] * num_controls)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    control, target = (all_qubits[:num_controls], all_qubits[num_controls:])
    turn_on_controls = [cirq.X(c) for c in control]
    prepare_uniform_op = gate.on(*control, *target)
    circuit = cirq.Circuit(turn_on_controls, prepare_uniform_op)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=all_qubits)
    final_target_state = cirq.sub_state_vector(
        result.final_state_vector,
        keep_indices=list(range(num_controls, num_controls + len(target))),
    )
    expected_target_state = np.asarray([np.sqrt(1.0 / n)] * n + [0] * (2 ** len(target) - n))
    cirq.testing.assert_allclose_up_to_global_phase(
        expected_target_state, final_target_state, atol=1e-6
    )
    qlt_testing.assert_valid_bloq_decomposition(gate)


@pytest.mark.parametrize("n", [*range(3, 41, 3)])
def test_prepare_uniform_superposition_t_complexity(n: int):
    gate = PrepareUniformSuperposition(n)
    result = t_complexity(gate)
    assert result.rotations <= 2
    # TODO(#235): Uncomputing `LessThanGate` should take 0 T-gates instead of 4 * n
    # and therefore the total complexity should come down to `8 * logN`
    assert result.t <= 12 * (n - 1).bit_length()

    gate = PrepareUniformSuperposition(n, cvs=(1,))
    result = t_complexity(gate)
    # TODO(#233): Controlled-H is currently counted as a separate rotation, but it can be
    # implemented using 2 T-gates.
    assert result.rotations <= 2 + 2 * total_bits(gate.signature)
    assert result.t <= 12 * (n - 1).bit_length()


def test_prepare_uniform_superposition_consistent_protocols():
    gate = PrepareUniformSuperposition(5, cvs=(1, 0))
    # Diagrams
    expected_symbols = ('@', '@(0)', 'UNIFORM(5)', 'target', 'target')
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_symbols
    # Equality
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        PrepareUniformSuperposition(5, cvs=(1, 0)), PrepareUniformSuperposition(5, cvs=[1, 0])
    )
    equals_tester.add_equality_group(
        PrepareUniformSuperposition(5, cvs=(0, 1)), PrepareUniformSuperposition(5, cvs=[0, 1])
    )
    equals_tester.add_equality_group(
        PrepareUniformSuperposition(5),
        PrepareUniformSuperposition(5, cvs=()),
        PrepareUniformSuperposition(5, cvs=[]),
    )
