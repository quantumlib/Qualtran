#  Copyright 2024 Google LLC
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
from qualtran.bloqs.mcmt.multi_target_cnot import _c_multi_not, _c_multi_not_symb, MultiTargetCNOT


def test_examples(bloq_autotester):
    bloq_autotester(_c_multi_not)


def test_symbolic_examples(bloq_autotester):
    bloq_autotester(_c_multi_not_symb)


@pytest.mark.parametrize("num_targets", [3, 4, 6, 8, 10])
def test_multi_target_cnot(num_targets):
    qubits = cirq.LineQubit.range(num_targets + 1)
    naive_circuit = cirq.Circuit(cirq.CNOT(qubits[0], q) for q in qubits[1:])
    bloq = MultiTargetCNOT(num_targets)
    op = bloq.on(*qubits)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(op), naive_circuit, atol=1e-6
    )
    optimal_circuit = cirq.Circuit(cirq.decompose_once(op))
    assert len(optimal_circuit) == 2 * np.ceil(np.log2(num_targets)) + 1
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize', range(1, 5))
def test_multitargetcnot_classical_action(bitsize):
    b = MultiTargetCNOT(bitsize)
    qlt_testing.assert_consistent_classical_action(b, targets=range(2**bitsize), control=range(2))
