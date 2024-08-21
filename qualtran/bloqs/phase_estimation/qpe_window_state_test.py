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
import numpy as np

from qualtran.bloqs.phase_estimation.qpe_window_state import (
    _rectangular_window_state_small,
    _rectangular_window_state_symbolic,
    RectangularWindowState,
)


def test_rectangular_window_state_tensor():
    n = 4
    bloq = RectangularWindowState(n)
    np.testing.assert_allclose(bloq.tensor_contract(), np.zeros(2**n) + 1 / 2 ** (n / 2))


def test_rectangular_window_state(bloq_autotester):
    bloq_autotester(_rectangular_window_state_small)
    bloq_autotester(_rectangular_window_state_symbolic)
