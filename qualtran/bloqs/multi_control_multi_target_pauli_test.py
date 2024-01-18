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

from qualtran.bloqs.multi_control_multi_target_pauli import (
    MultiControlPauli,
    MultiControlX,
    MultiTargetCNOT,
)
from qualtran.cirq_interop.testing import assert_decompose_is_consistent_with_t_complexity
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize("num_targets", [3, 4, 6, 8, 10])
def test_multi_target_cnot(num_targets):
    qubits = cirq.LineQubit.range(num_targets + 1)
    naive_circuit = cirq.Circuit(cirq.CNOT(qubits[0], q) for q in qubits[1:])
    op = MultiTargetCNOT(num_targets).on(*qubits)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(op), naive_circuit, atol=1e-6
    )
    optimal_circuit = cirq.Circuit(cirq.decompose_once(op))
    assert len(optimal_circuit) == 2 * np.ceil(np.log2(num_targets)) + 1
    assert_valid_bloq_decomposition(op.gate)


@pytest.mark.parametrize("num_controls", [*range(7, 17)])
@pytest.mark.parametrize("pauli", [cirq.X, cirq.Y, cirq.Z])
@pytest.mark.parametrize('cv', [0, 1])
def test_t_complexity_mcp(num_controls: int, pauli: cirq.Pauli, cv: int):
    gate = MultiControlPauli([cv] * num_controls, target_gate=pauli)
    assert_valid_bloq_decomposition(gate)
    assert_decompose_is_consistent_with_t_complexity(gate)


@pytest.mark.parametrize("cvs", [(), (0), (1, 0), (1, 1, 1), (1, 0, 1, 0)])
def test_multi_control_x(cvs):
    bloq = MultiControlX(cvs=cvs)
    assert_valid_bloq_decomposition(bloq=bloq)


@pytest.mark.parametrize(
    "cvs,x,ctrl",
    [((), 0, 0), ((0), 1, 0), ((1, 0), 0, 2), ((1, 1, 1), 1, 7), ((1, 0, 1, 0), 1, 10)],
)
def test_classical_multi_control_x(x, cvs, ctrl):
    bloq = MultiControlX(cvs=cvs)
    cbloq = bloq.decompose_bloq()
    ret1 = bloq.call_classically(x=x, ctrl=ctrl)
    ret2 = cbloq.call_classically(x=x, ctrl=ctrl)
    assert ret1 == ret2


@pytest.mark.parametrize(
    "cvs,x,ctrl,result",
    [
        ((), 0, 0, 1),
        ((0), 1, 0, 0),
        ((1, 0), 0, 2, 1),
        ((1, 1, 1), 1, 7, 0),
        ((1, 0, 1, 0), 1, 10, 0),
        ((1), 0, 0, 0),
    ],
)
def test_classical_multi_control_x_correctness(x, cvs, ctrl, result):
    bloq = MultiControlX(cvs=cvs)
    cbloq = bloq.decompose_bloq()
    ret = bloq.call_classically(x=x, ctrl=ctrl)
    assert ret[-1] == result
