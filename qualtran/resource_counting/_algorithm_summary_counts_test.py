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
import pytest

import qualtran
from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.resource_counting import AlgorithmSummaryCounts, get_cost_value


@pytest.mark.parametrize(
    ['bloq', 'counts'],
    [
        # T Gate
        [basic_gates.TGate(is_adjoint=False), {'t_gates': 1}],
        # Toffoli
        [basic_gates.Toffoli(), {'toffoli_gates': 1}],
        # CSwap
        [basic_gates.TwoBitCSwap(), {'toffoli_gates': 1}],
        # And
        [mcmt.And(), {'toffoli_gates': 1}],
        # Rotations
        [basic_gates.ZPowGate(exponent=0.1, global_shift=0.0, eps=1e-11), {'rotation_gates': 1}],
        [
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
            {'rotation_gates': 12, 'rotation_circuit_depth': 1},
        ],
        # Recursive
        [
            mcmt.MultiControlPauli(cvs=(1, 1, 1), target_gate=cirq.X),
            {'toffoli_gates': 2, 'rotation_circuit_depth': 2},
        ],
    ],
)
def test_algorithm_summary_counts(bloq, counts):
    assert get_cost_value(bloq, AlgorithmSummaryCounts()) == counts
