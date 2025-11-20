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

from typing import Iterator, Optional

import numpy as np

import qualtran.rotation_synthesis.lattice.grid_operators as go
from qualtran.rotation_synthesis.lattice import geometry, state

ALL_CORE_GRID_OPS = [
    go.ISqrt2,
    go.RSqrt2,
    go.KSqrt2,
    go.ASqrt2,
    go.BSqrt2,
    go.XSqrt2,
    go.ZSqrt2,
    go.KConjSqrt2,
]


def make_psd(rng: np.random.Generator) -> np.ndarray:
    """Returns a random positive semidefinite 2x2 matrix."""
    a = (rng.random((2, 2)) * 2 - 1) * 10
    while np.linalg.matrix_rank(a) != 2:
        a = (rng.random((2, 2)) * 2 - 1) * 10
    ret = a @ a.T
    ret /= np.sqrt(np.linalg.det(ret))
    return ret


def random_grid_operator(
    n_operators: int, n_components: int, seed: Optional[int] = 0
) -> Iterator[go.GridOperator]:
    """Yields the requested number of random grid operators.

    Each grid operator will be built out of `n_components` operators randomly selected
    from {I, R, K, A, B, X, Z, K}

    Args:
        n_operators: the number of operators to yield.
        n_components: the number of base operators to select for each operator.
        seed: Optional seed.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_operators):
        g = go.ISqrt2
        for i in rng.choice(len(ALL_CORE_GRID_OPS), n_components):
            g = g @ ALL_CORE_GRID_OPS[i]
        yield g


def make_states(n: int, seed: Optional[int] = 0) -> Iterator[state.SelingerState]:
    """Yields `n` random states."""
    rng = np.random.default_rng(seed)
    yield from [
        state.SelingerState(geometry.Ellipse(make_psd(rng)), geometry.Ellipse(make_psd(rng)))
        for _ in range(n)
    ]


def make_ellipses(n, seed=0) -> Iterator[geometry.Ellipse]:
    rng = np.random.default_rng(seed)
    yield from [geometry.Ellipse(make_psd(rng), 10 * (rng.random() * 2 - 1)) for _ in range(n)]
