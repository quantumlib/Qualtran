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

import numpy as np
import pytest

from qualtran.surface_code import quantum_error_correction_scheme_summary as qecs


@pytest.mark.parametrize('qec,want', [(qecs.BeverlandSuperconductingQubits, 3e-4)])
def test_logical_error_rate(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: float):
    assert qec.logical_error_rate(3, 1e-3) == pytest.approx(want)


@pytest.mark.parametrize('qec,want', [[qecs.BeverlandSuperconductingQubits, 242]])
def test_physical_qubits(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: int):
    assert qec.physical_qubits(11) == want


@pytest.mark.parametrize('qec,want', [[qecs.BeverlandSuperconductingQubits, 4.8]])
def test_error_detection_cycle_time(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: float):
    assert qec.error_detection_circuit_time_us(12) == pytest.approx(want)


def test_invert_error_at():
    phys_err = 1e-3
    budgets = np.logspace(-1, -18)
    for budget in budgets:
        d = qecs.FowlerSuperconductingQubits.code_distance_from_budget(
            physical_error_rate=phys_err, budget=budget
        )
        assert d % 2 == 1
        assert (
            qecs.FowlerSuperconductingQubits.logical_error_rate(
                physical_error_rate=phys_err, code_distance=d
            )
            <= budget
        )
        if d > 3:
            assert (
                qecs.FowlerSuperconductingQubits.logical_error_rate(
                    physical_error_rate=phys_err, code_distance=d - 2
                )
                > budget
            )
