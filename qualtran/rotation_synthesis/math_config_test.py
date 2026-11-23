#  Copyright 2026 Google LLC
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

import math

import mpmath
import numpy as np
import pytest

import qualtran.rotation_synthesis._math_config as mc


def test_numpy_config_number_and_sqrt():
    number = mc.NumpyConfig.number(1.25)

    assert isinstance(number, np.longdouble)
    assert mc.NumpyConfig.sqrt(number) == pytest.approx(np.sqrt(number))


@pytest.mark.parametrize(
    ("value", "expected_floor", "expected_ceil"),
    [(2.3, 2, 3), (-2.3, -3, -2), (2.0, 2, 2), (-2.0, -2, -2)],
)
def test_numpy_config_floor_and_ceil(value, expected_floor, expected_ceil):
    assert mc.NumpyConfig.floor(value) == expected_floor
    assert mc.NumpyConfig.ceil(value) == expected_ceil


def test_numpy_config_isclose():
    assert mc.NumpyConfig.isclose(1.0, 1.0 + 1e-8)
    assert not mc.NumpyConfig.isclose(1.0, 1.1)


def test_with_dps_returns_mpmath_config_with_requested_precision():
    original_dps = mpmath.mp.dps
    try:
        config = mc.with_dps(50)

        assert mpmath.mp.dps == 50
        assert config.name == "mpmath_50"
        assert isinstance(config.number("1.25"), mpmath.mpf)
        assert isinstance(config.sqrt(4), mpmath.mpf)
        assert config.floor(mpmath.mpf("-2.3")) == -3
        assert config.ceil(mpmath.mpf("-2.3")) == -2
        assert config.isclose(mpmath.mpf(1), mpmath.mpf(1) + mpmath.mpf("1e-25"))
        assert not config.isclose(mpmath.mpf(1), mpmath.mpf(1) + mpmath.mpf("1e-10"))
    finally:
        mpmath.mp.dps = original_dps


@pytest.mark.parametrize(
    ("x", "y", "eps", "expected"),
    [
        (0.0, 5e-7, 1e-6, True),
        (0.0, 2e-6, 1e-6, False),
        (1e8, 1e8 + 1.0, 1e-6, True),
        (1e8, 1e8 + 101.0, 1e-6, False),
    ],
)
def test_isclose_absolute_and_relative_tolerance(x, y, eps, expected):
    assert mc.isclose(x, y, eps) is expected


@pytest.mark.parametrize("config", [mc.MathConfig(), mc.NumpyConfig])
def test_math_config_transcendental_helpers(config):
    assert config.log(math.e) == pytest.approx(1.0)
    assert config.cos(0.0) == pytest.approx(1.0)
    assert config.sin(0.0) == pytest.approx(0.0)

    assert config.arctan2(1.0, 1.0) == pytest.approx(math.pi / 4)
    assert config.arcsin(0.5) == pytest.approx(math.pi / 6)
    assert config.arccos(0.5) == pytest.approx(math.pi / 3)


def test_math_config_hash_uses_name():
    first = mc.MathConfig(name="custom")
    same_name = mc.MathConfig(name="custom", pi=3.0)

    assert hash(first) == hash(same_name)
    assert hash(first) == hash("custom")
