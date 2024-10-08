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
import pytest

from qualtran.bloqs.max_k_xor_sat.resource.phase_gradient import AcquirePhaseGradient


@pytest.mark.parametrize('n', [3, 4, 5, 10, 12])
@pytest.mark.parametrize('adjoint', [False, True])
def test_tensor(n: int, adjoint: bool):
    phase_grad = AcquirePhaseGradient(n)

    actual = (phase_grad.adjoint() if adjoint else phase_grad).tensor_contract()

    expected = phase_grad.prepare.tensor_contract()
    if adjoint:
        expected = expected.conj().T

    np.testing.assert_allclose(actual, expected)
