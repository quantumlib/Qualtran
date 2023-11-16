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

from qualtran.surface_code import quantum_error_correction_scheme_summary as qecs


@pytest.mark.parametrize(
    'qec,want', [(qecs.Fowler, 1e-3), (qecs.BeverlandSuperConductingQubits, 3e-4)]
)
def test_logical_error_rate(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: float):
    assert qec.logical_error_rate(3, 1e-3) == pytest.approx(want)


@pytest.mark.parametrize(
    'qec,want', [[qecs.BeverlandSuperConductingQubits, 242], [qecs.Fowler, 242]]
)
def test_physical_qubits(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: int):
    assert qec.physical_qubits(11) == want


@pytest.mark.parametrize('qec,want', [[qecs.BeverlandSuperConductingQubits, 4.8], [qecs.Fowler, 1]])
def test_error_detection_cycle_time(qec: qecs.QuantumErrorCorrectionSchemeSummary, want: float):
    assert qec.syndrome_detection_time_us(12) == pytest.approx(want)
