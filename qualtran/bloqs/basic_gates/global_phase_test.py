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

from qualtran import CtrlSpec
from qualtran.bloqs.basic_gates.global_phase import _global_phase, GlobalPhase
from qualtran.cirq_interop import cirq_gate_to_bloq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


def test_unitary():
    random_state = np.random.RandomState(2)

    for alpha in random_state.random(size=10):
        coefficient = np.exp(2j * np.pi * alpha)
        bloq = GlobalPhase(exponent=2 * alpha)
        np.testing.assert_allclose(cirq.unitary(bloq), coefficient)


@pytest.mark.parametrize("cv", [0, 1])
def test_controlled(cv: int):
    ctrl_spec = CtrlSpec(cvs=cv)
    random_state = np.random.RandomState(2)
    for alpha in random_state.random(size=10):
        coefficient = np.exp(2j * np.pi * alpha)
        bloq = GlobalPhase(exponent=2 * alpha).controlled(ctrl_spec=ctrl_spec)
        np.testing.assert_allclose(
            cirq.unitary(cirq.GlobalPhaseGate(coefficient).controlled(control_values=[cv])),
            bloq.tensor_contract(),
        )


def test_cirq_interop():
    bloq = GlobalPhase.from_coefficient(1.0j)
    gate = cirq.GlobalPhaseGate(1.0j)

    circuit = bloq.as_composite_bloq().to_cirq_circuit()
    assert cirq.approx_eq(circuit, cirq.Circuit(gate.on()), atol=1e-16)

    assert cirq_gate_to_bloq(gate) == bloq


def test_t_complexity():
    assert GlobalPhase(exponent=0.5).t_complexity() == TComplexity()


def test_global_phase(bloq_autotester):
    bloq_autotester(_global_phase)
