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

import numpy as np

import qualtran.rotation_synthesis.lattice as lattice
from qualtran.rotation_synthesis.lattice import _test_utils as tu


def test_make_psd():
    rng = np.random.default_rng(0)
    for _ in range(10):
        matrix = tu.make_psd(rng)
        assert np.all(np.linalg.eigvals(matrix) >= 0)


def test_make_ellipse():
    for n in range(3, 5):
        es = tuple(tu.make_ellipses(n))
        assert len(es) == n
        for e in es:
            assert isinstance(e, lattice.Ellipse)


def test_make_states():
    for n in range(3, 5):
        states = tuple(tu.make_states(n))
        assert len(states) == n
        for state in states:
            assert isinstance(state, lattice.SelingerState)


def test_make_gridoperators():
    for n in range(3, 5):
        gos = tuple(tu.random_grid_operator(n, 3))
        assert len(gos) == n
        for go in gos:
            assert isinstance(go, lattice.GridOperator)
