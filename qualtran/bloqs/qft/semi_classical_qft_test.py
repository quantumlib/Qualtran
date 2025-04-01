#  Copyright 2025 Google LLC
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
import sympy

from qualtran.bloqs.qft import semi_classical_qft
from qualtran.resource_counting import get_cost_value, QECGatesCost


@pytest.mark.parametrize('n', [*range(1, 10), sympy.Symbol('n')])
def test_semi_classical_qft_cost(n):
    blq = semi_classical_qft.SemiClassicalQFT(n)
    cost = get_cost_value(blq, QECGatesCost())
    assert cost.rotation == n - 1
    assert cost.clifford == n
    assert cost.measurement == n
