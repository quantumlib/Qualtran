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

import pytest
from dash.exceptions import PreventUpdate

from qualtran.surface_code import ui


@pytest.mark.parametrize('estimation_model', ui._SUPPORTED_ESTIMATION_MODELS)
def test_ensure_support_for_all_supported_models(estimation_model: str):
    # Make sure the update runs without failure for all supported models.
    _ = ui.update(
        physical_error_rate=1e-4,
        error_budget=1e-3,
        estimation_model=estimation_model,
        algorithm_data=(10**11,) * 6,
        qec_name='FowlerSuperconductingQubits',
        magic_name='FifteenToOne733',
        magic_count=1,
        rotaion_model_name='BeverlandEtAlRotationCost',
    )


@pytest.mark.parametrize(
    'estimation_model,desired',
    [
        (
            ui._GIDNEY_FOLWER_MODEL,
            (
                {'display': 'none'},
                ['Total Number of Toffoli gates'],
                '9e+11',
                {'display': 'none'},
                1,
                {'display': 'block'},
                '2914.93 days',
            ),
        ),
        (
            ui._BEVERLAND_MODEL,
            (
                {'display': 'block'},
                ['Total Number of T gates'],
                '3.6e+12',
                {'display': 'block'},
                18,
                {'display': 'none'},
                '',
            ),
        ),
    ],
)
def test_update(estimation_model: str, desired):
    (
        _,
        display_runtime,
        _,
        magic_name,
        magic_count,
        display_factory_count,
        factory_count,
        display_duration,
        duration,
    ) = ui.update(
        physical_error_rate=1e-4,
        error_budget=1e-3,
        estimation_model=estimation_model,
        algorithm_data=(10**11,) * 6,
        qec_name='FowlerSuperconductingQubits',
        magic_name='FifteenToOne733',
        magic_count=1,
        rotaion_model_name='BeverlandEtAlRotationCost',
    )
    assert (
        display_runtime,
        magic_name,
        magic_count,
        display_factory_count,
        factory_count,
        display_duration,
        duration,
    ) == desired


def test_update_bad_input():
    with pytest.raises(PreventUpdate):
        _ = ui.update(
            physical_error_rate=None,
            error_budget=1e-3,
            estimation_model=ui._SUPPORTED_ESTIMATION_MODELS[0],
            algorithm_data=(10**11,) * 6,
            qec_name='FowlerSuperconductingQubits',
            magic_name='FifteenToOne733',
            magic_count=1,
            rotaion_model_name='BeverlandEtAlRotationCost',
        )
