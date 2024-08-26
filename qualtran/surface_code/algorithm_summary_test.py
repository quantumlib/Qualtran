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

import pytest

from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary


@pytest.mark.parametrize(
    ['bloq', 'summary'],
    [
        [
            basic_gates.TGate(is_adjoint=False),
            AlgorithmSummary(n_algo_qubits=1, n_logical_gates=GateCounts(t=1)),
        ],
        [
            basic_gates.Toffoli(),
            AlgorithmSummary(n_algo_qubits=3, n_logical_gates=GateCounts(toffoli=1)),
        ],
        [
            basic_gates.TwoBitCSwap(),
            AlgorithmSummary(n_algo_qubits=3, n_logical_gates=GateCounts(cswap=1)),
        ],
        [mcmt.And(), AlgorithmSummary(n_algo_qubits=3, n_logical_gates=GateCounts(and_bloq=1))],
        [
            basic_gates.ZPowGate(exponent=0.1, global_shift=0.0, eps=1e-11),
            AlgorithmSummary(n_algo_qubits=1, n_logical_gates=GateCounts(rotation=1)),
        ],
        [
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
            AlgorithmSummary(
                n_algo_qubits=10, n_logical_gates=GateCounts(clifford=2, t=1, rotation=7)
            ),
        ],
        [
            mcmt.MultiControlX(cvs=(1, 1, 1)),
            AlgorithmSummary(
                n_algo_qubits=6, n_logical_gates=GateCounts(and_bloq=2, measurement=2, clifford=3)
            ),
        ],
    ],
)
def test_summary_from_bloq(bloq, summary):
    assert AlgorithmSummary.from_bloq(bloq) == summary
