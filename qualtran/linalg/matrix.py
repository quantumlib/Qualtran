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
from cirq.testing.lin_alg_utils import random_orthogonal
from numpy.typing import NDArray


def random_hermitian_matrix(dim: int, random_state: np.random.RandomState) -> NDArray:
    """Return a random Hermitian matrix of given dimension and norm <= 1."""
    a = random_orthogonal(dim, random_state=random_state)
    return (a + a.conj().T) / 2
