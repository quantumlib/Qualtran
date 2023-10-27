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

from qualtran.surface_code.algorithm_summary import AlgorithmSummary


def test_mul():
    assert AlgorithmSummary(t_gates=9) == 3 * AlgorithmSummary(t_gates=3)

    with pytest.raises(TypeError):
        _ = complex(1, 0) * AlgorithmSummary(rotation_gates=1)
