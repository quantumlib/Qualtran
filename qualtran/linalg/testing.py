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
from numpy.typing import NDArray

from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase


def assert_matrices_almost_equal(A: NDArray, B: NDArray, *, atol: float = 1e-5):
    r"""Asserts that two matrices are close to each other by bounding the matrix norm of their difference.

    Asserts that $\|A - B\| \le \mathrm{atol}$.
    """
    assert A.shape == B.shape
    assert np.linalg.norm(A - B) <= atol


def assert_unitaries_equivalent_upto_global_phase(U: NDArray, V: NDArray, *, atol: float = 1e-5):
    d = unitary_distance_ignoring_global_phase(U, V)
    assert d <= atol, f"unitaries are not equivalent: distance={d} ({atol=})"
