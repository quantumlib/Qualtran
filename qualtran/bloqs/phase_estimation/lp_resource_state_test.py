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

from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.cirq_interop.testing import (
    assert_decompose_is_consistent_with_t_complexity,
    GateHelper,
)


@pytest.mark.parametrize('n', [*range(1, 14, 2)])
def test_prepares_resource_state(n):
    bloq = LPResourceState(n)
    state = GateHelper(bloq).circuit.final_state_vector()
    np.testing.assert_allclose(state, bloq.state())


@pytest.mark.parametrize('n', [*range(1, 14, 2)])
def test_t_complexity(n):
    bloq = LPResourceState(n)
    if n == 1:
        # n=1 fails due to https://github.com/quantumlib/Qualtran/issues/785
        with pytest.raises(AssertionError):
            assert_decompose_is_consistent_with_t_complexity(bloq)
    else:
        assert_decompose_is_consistent_with_t_complexity(bloq)
        assert bloq.t_complexity().t + bloq.t_complexity().rotations == 7 * n + 6
