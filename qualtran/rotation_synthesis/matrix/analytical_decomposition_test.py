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
import pytest
from scipy import stats

import qualtran.rotation_synthesis.math_config as mc
from qualtran.rotation_synthesis.matrix import analytical_decomposition as ad


def _random_angles(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.random(3) * 2 * np.pi


def _random_su2(n, seed):
    for u in stats.unitary_group(2).rvs(n, seed):
        yield u / np.linalg.det(u) ** 0.5


@pytest.mark.parametrize(["theta1", "phi", "theta2"], _random_angles(100, seed=0))
def test_decomposition_round_trip(theta1, phi, theta2):
    u = ad.unitary_from_zxz(theta1, phi, theta2, mc.NumpyConfig)
    angles = ad.su_unitary_to_zxz_angles(u, config=mc.NumpyConfig)
    v = ad.unitary_from_zxz(*angles, config=mc.NumpyConfig)
    np.testing.assert_allclose(actual=u, desired=v)


@pytest.mark.parametrize("u", _random_su2(100, seed=0))
def test_decomposition_round_trip_start_with_unitary(u):
    angles = ad.su_unitary_to_zxz_angles(u, config=mc.NumpyConfig)
    v = ad.unitary_from_zxz(*angles, config=mc.NumpyConfig)
    np.testing.assert_allclose(actual=u, desired=v)
