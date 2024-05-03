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
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import (
    MultiControlPauli,
    MultiControlX,
    MultiTargetCNOT,
)


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


@pytest.mark.parametrize("num_controls", [0, 1, 2, *range(7, 17)])
@pytest.mark.parametrize("pauli", [cirq.X, cirq.Y, cirq.Z])
@pytest.mark.parametrize('cv', [0, 1])
def test_t_complexity_mcp(num_controls: int, pauli: cirq.Pauli, cv: int):
    gate = MultiControlPauli([cv] * num_controls, target_gate=pauli)
    qlt_testing.assert_valid_bloq_decomposition(gate)
    qlt_testing.assert_equivalent_bloq_counts(gate)


@pytest.mark.parametrize("num_controls", [*range(10)])
@pytest.mark.parametrize("pauli", [cirq.X, cirq.Y, cirq.Z])
@pytest.mark.parametrize('cv', [0, 1])
def test_mcp_unitary(num_controls: int, pauli: cirq.Pauli, cv: int):
    cvs = (cv,) * num_controls
    gate = MultiControlPauli(cvs, target_gate=pauli)
    cpauli = pauli.controlled(control_values=cvs) if num_controls else pauli
    np.testing.assert_allclose(gate.tensor_contract(), cirq.unitary(cpauli))


@pytest.mark.parametrize("cvs", [(0,), (1, 0), (1, 1, 1), (1, 0, 1, 0)])
def test_multi_control_x(cvs):
    bloq = MultiControlX(cvs=cvs)
    qlt_testing.assert_valid_bloq_decomposition(bloq=bloq)


@pytest.mark.parametrize(
    "cvs,x,ctrls,result",
    [
        ((0,), 1, (0,), 0),
        ((1, 0), 0, (1, 0), 1),
        ((1, 1, 1), 1, (1, 1, 1), 0),
        ((1, 0, 1, 0), 1, (1, 0, 1, 0), 0),
        ((1,), 0, (0,), 0),
        ((), 0, (), 1),
    ],
)
def test_classical_multi_control_pauli_target_x(cvs, x, ctrls, result):
    bloq = MultiControlPauli(cvs=cvs, target_gate=cirq.X)
    cbloq = bloq.decompose_bloq()
    kwargs = {'target': x} | ({'controls': ctrls} if ctrls else {})
    bloq_classical = bloq.call_classically(**kwargs)
    cbloq_classical = cbloq.call_classically(**kwargs)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.parametrize(
    "cvs,x,ctrls,result",
    [
        ((0,), 1, (0,), 0),
        ((1, 0), 0, (1, 0), 1),
        ((1, 1, 1), 1, (1, 1, 1), 0),
        ((1, 0, 1, 0), 1, (1, 0, 1, 0), 0),
        ((1,), 0, (0,), 0),
    ],
)
def test_classical_multi_control_x(cvs, x, ctrls, result):
    bloq = MultiControlX(cvs=cvs)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=x, ctrls=ctrls)
    cbloq_classical = cbloq.call_classically(x=x, ctrls=ctrls)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result
