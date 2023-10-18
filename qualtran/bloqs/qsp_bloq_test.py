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

from qualtran.bloqs.qsp_bloq import qsp_phase_factors


def test_compute_qsp_phase_factors():
    phases = qsp_phase_factors([0.5, 0.5], [0.5, -0.5])
    assert (phases['theta'] == [0, np.pi / 4]).all()
    assert (phases['phi'] == [0, np.pi]).all()
    assert phases['lambda'] == 0
