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

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.protocols import _clifford_t_synthesis

LAMBDA = rings.ZSqrt2(2, 1)
_CONFIG = mc.with_dps(40)


def _make_tests(n_points: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_points):
        theta = rng.random() * 2 * np.pi
        error = np.power(10, -1 - 4 * rng.random())  # 1e-4 <= eps <= 1e-1
        yield theta, error


@pytest.mark.parametrize(["theta", "error"], _make_tests(20, 0))
def test_diagonal_protocol(theta, error):
    res = _clifford_t_synthesis.diagonal_unitary_approx(theta, error, 100, _CONFIG)
    assert res is not None
    assert res.n <= (-3.02 * np.log2(error) + 1.77) * 1.3
    assert res.diamond_norm_distance_to_rz(theta, _CONFIG) <= error


@pytest.mark.parametrize(["theta", "error"], _make_tests(20, 0))
def test_fallback_protocol(theta, error):
    res = _clifford_t_synthesis.fallback_protocol(
        theta, error, 0.99, 100, _CONFIG, eps_rotation=error
    )
    assert res is not None
    assert res.expected_num_ts(_CONFIG) <= (-1.03 * np.log2(error) + 5.75) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, _CONFIG)
    assert actual_error <= error


@pytest.mark.parametrize(["theta", "error"], _make_tests(20, 0))
def test_mixed_diagonal_protocol(theta, error):
    res = _clifford_t_synthesis.mixed_diagonal_protocol(theta, error, 100, _CONFIG)
    assert res is not None
    assert res.expected_num_ts(_CONFIG) <= (-1.52 * np.log2(error) - 0.01) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, _CONFIG)
    assert actual_error <= error


@pytest.mark.parametrize(["theta", "error"], _make_tests(20, 0))
def test_mixed_fallback_protocol(theta, error):
    res = _clifford_t_synthesis.mixed_fallback_protocol(
        theta, error, 0.99, 100, _CONFIG, fallback_min_eps=1e-5, num_valid_points=2
    )
    if res is None:
        return
    assert res.expected_num_ts(_CONFIG) <= (-0.53 * np.log2(error) + 5) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, _CONFIG)
    assert actual_error <= error


@pytest.mark.parametrize(
    ["u", "eps"], zip(stats.unitary_group(2).rvs(10, 0), np.linspace(1e-3, 1e-1, 10), strict=True)
)
def test_magnitude_approx(u, eps):
    config = mc.with_dps(50)
    res = _clifford_t_synthesis.magnitude_approx(u, eps, 50, config)
    assert res is not None
    assert res.diamond_norm_distance_to_unitary(u, config) <= eps
