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

from qualtran.bloqs.rotations.phase_gradient import PhaseGradientSchoolBook, PhaseGradientState
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('n', [6, 7, 8])
def test_phase_gradient_state(n: int):
    gate = PhaseGradientState(n)
    assert_valid_bloq_decomposition(gate)
    assert_valid_bloq_decomposition(gate**-1)

    q = cirq.LineQubit.range(n)
    state_prep_cirq_circuit = cirq.Circuit(
        cirq.H.on_each(*q), cirq.PhaseGradientGate(num_qubits=n, exponent=-1).on(*q)
    )
    assert np.allclose(cirq.unitary(gate), cirq.unitary(state_prep_cirq_circuit))
    assert np.allclose(
        cirq.unitary(gate**-1), cirq.unitary(cirq.inverse(state_prep_cirq_circuit))
    )
    assert gate.t_complexity().rotations == n - 2
    assert gate.t_complexity().clifford == n + 2


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('exponent', [-0.5, 1, 1 / 10])
@pytest.mark.parametrize('controlled', [True, False])
def test_phase_gradient_gate(n: int, exponent, controlled):
    bloq = PhaseGradientSchoolBook(n, exponent, controlled)
    assert_valid_bloq_decomposition(bloq)
    assert_valid_bloq_decomposition(bloq**-1)
    cirq_gate = cirq.PhaseGradientGate(num_qubits=n, exponent=exponent)
    if controlled:
        cirq_gate = cirq_gate.controlled()
    assert np.allclose(cirq.unitary(bloq), cirq.unitary(cirq_gate))
