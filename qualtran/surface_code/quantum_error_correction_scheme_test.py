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

from qualtran.surface_code import quantum_error_correction_scheme as qecs
from qualtran.surface_code.physical_parameters import PhysicalParameters


@pytest.mark.parametrize(
    'qec,want',
    [
        [qecs.GateBasedSurfaceCode(error_rate_scaler=0.03, error_rate_threshold=0.01), 3e-7],
        [
            qecs.MeasurementBasedSurfaceCode(error_rate_scaler=0.04, error_rate_threshold=0.09),
            6.77e-12,
        ],
        [
            qecs.MeasurementBasedHastingsHaahCode(
                error_rate_scaler=0.05, error_rate_threshold=0.06
            ),
            6.43e-11,
        ],
    ],
)
def test_logical_error_rate(qec: qecs.QuantumErrorCorrectionScheme, want: float):
    assert qec.logical_error_rate(9, 1e-3) == pytest.approx(want)


@pytest.mark.parametrize(
    'qec,want',
    [
        [qecs.GateBasedSurfaceCode(error_rate_scaler=0.03, error_rate_threshold=0.01), 242],
        [qecs.MeasurementBasedSurfaceCode(error_rate_scaler=0.04, error_rate_threshold=0.09), 242],
        [
            qecs.MeasurementBasedHastingsHaahCode(
                error_rate_scaler=0.05, error_rate_threshold=0.06
            ),
            564,
        ],
    ],
)
def test_physical_qubits(qec: qecs.QuantumErrorCorrectionScheme, want: int):
    assert qec.physical_qubits(11) == want


@pytest.mark.parametrize(
    'qec,want',
    [
        [qecs.GateBasedSurfaceCode(error_rate_scaler=0.03, error_rate_threshold=0.01), 4.8e-6],
        [
            qecs.MeasurementBasedSurfaceCode(error_rate_scaler=0.04, error_rate_threshold=0.09),
            2.4e-5,
        ],
        [
            qecs.MeasurementBasedHastingsHaahCode(
                error_rate_scaler=0.05, error_rate_threshold=0.06
            ),
            3.6e-6,
        ],
    ],
)
def test_logical_time_step(qec: qecs.QuantumErrorCorrectionScheme, want: float):
    assert qec.logical_time_step(
        12, physical_parameters=PhysicalParameters(50e-9, 100e-9, 1e-4)
    ) == pytest.approx(want)
