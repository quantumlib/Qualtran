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

import itertools
import math

import numpy as np
import pytest

from qualtran.rotation_synthesis import rings
from qualtran.rotation_synthesis.rings import _test_utils as tu

_SQRT_2 = math.sqrt(2)


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_value(x: rings.ZSqrt2):
    assert float(x) == pytest.approx(x.a + _SQRT_2 * x.b)


@pytest.mark.parametrize("p", range(10))
@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_power(p: int, x: rings.ZSqrt2):
    a, b = float(x) ** p, float(x**p)
    np.testing.assert_approx_equal(float(x) ** p, float(x**p), err_msg=f"{x=} {p=} {a=} {b=}")


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_conjugate(x: rings.ZSqrt2):
    assert float(x.conj()) == pytest.approx(x.a - _SQRT_2 * x.b)


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_add(x: rings.ZSqrt2, y: rings.ZSqrt2):
    assert float(x + y) == pytest.approx(float(x) + float(y))


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_sub(x: rings.ZSqrt2, y: rings.ZSqrt2):
    assert float(x - y) == pytest.approx(float(x) - float(y))


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_mul(x: rings.ZSqrt2, y: rings.ZSqrt2):
    assert float(x * y) == pytest.approx(float(x) * float(y))


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)])
def test_less_than(x, y):
    assert (x < y) == (float(x) < float(y))


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)])
def test_less_than_equal(x, y):
    assert (x <= y) == (float(x) <= float(y))


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_is_prime_ideal(p):
    for ideal in rings.ZSqrt2.factor_prime(p):
        assert ideal.is_prime_ideal(), f"{ideal=}"


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_factor_prime(p):
    r = rings.ZSqrt2(1, 0)
    for ideal, exponent in rings.ZSqrt2.factor_prime(p).items():
        r = r * ideal**exponent
    assert r == p, f"{p%8}"


@pytest.mark.parametrize("x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
@pytest.mark.parametrize("y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)])
def test_gcd(x, y):
    g = x.gcd(y)
    assert x.is_divisible_by(g)
    assert y.is_divisible_by(g)
    assert (x // g).gcd(y // g) == rings.ZSqrt2(1, 0)
