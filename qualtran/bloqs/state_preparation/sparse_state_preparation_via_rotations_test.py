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
from numpy.typing import NDArray

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
    _sparse_state_prep_via_rotations,
    SparseStatePreparationViaRotations,
)


def test_examples(bloq_autotester):
    bloq_autotester(_sparse_state_prep_via_rotations)


def get_prepared_state_vector(bloq: SparseStatePreparationViaRotations) -> NDArray[np.complex128]:
    bb = BloqBuilder()
    state = bb.add(IntState(0, bloq.target_bitsize))
    phase_gradient = bb.add(PhaseGradientState(bloq.phase_bitsize))
    state, phase_gradient = bb.add(bloq, target_state=state, phase_gradient=phase_gradient)
    bb.add(PhaseGradientState(bitsize=bloq.phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(state=state)
    result = network.tensor_contract()
    return result


@pytest.mark.slow
def test_prepared_state():
    expected_state = np.array(
        [
            (-0.42677669529663675 - 0.1767766952966366j),
            0,
            (0.17677669529663664 - 0.4267766952966367j),
            (0.17677669529663675 - 0.1767766952966368j),
            0,
            0,
            (0.07322330470336305 - 0.07322330470336309j),
            (0.4267766952966366 - 0.17677669529663692j),
            0,
            (0.42677669529663664 + 0.17677669529663675j),
            0,
            (0.0732233047033631 + 0.17677669529663678j),
            (-0.07322330470336308 - 0.17677669529663678j),
            0,
        ]
    )

    N = len(expected_state)

    bloq = SparseStatePreparationViaRotations.from_sparse_array(expected_state, phase_bitsize=3)
    actual_state = get_prepared_state_vector(bloq)
    np.testing.assert_allclose(np.linalg.norm(actual_state), 1)
    np.testing.assert_allclose(actual_state[:N], expected_state)
    np.testing.assert_allclose(actual_state[N:], 0)
