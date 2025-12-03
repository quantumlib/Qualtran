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
from typing import Optional

import numpy as np
import pytest

from qualtran.rotation_synthesis import _math_config as mc
from qualtran.rotation_synthesis import rings
from qualtran.rotation_synthesis.rings import _test_utils as tu

_Z8 = np.exp(1j * np.pi / 4)
_SQRT_2 = np.sqrt(2)


def _make_pairs(n_pairs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    zw_numbers = tuple(rings.ZW(p) for p in itertools.product(range(-1, 2), repeat=4))
    all_pairs = np.array(tuple(itertools.product(zw_numbers, repeat=2)))
    return all_pairs[rng.choice(len(all_pairs), n_pairs)]


@pytest.mark.parametrize("n", range(31))
@pytest.mark.parametrize("p", itertools.product(range(2), repeat=4))
def test_pow(n, p):
    x = sum(p[i] * _Z8**i for i in range(4))
    got = complex(rings.ZW(p) ** n)
    want = x**n
    assert np.isclose(got, want), f"{x=} {n=} {p=} {got=} {want=}"


@pytest.mark.parametrize("p", itertools.product(range(-2, 3), repeat=4))
def test_conjugate(p):
    v = rings.ZW(p)
    c = v.conjugate()
    assert np.isclose(v.mag(_SQRT_2), c.mag(_SQRT_2))
    assert np.isclose(
        complex(v * v.conjugate()).imag, 0
    ), f"{v=} {v.conjugate()} {v * v.conjugate()}"
    assert np.isclose(v.arg(mc.NumpyConfig) + c.arg(mc.NumpyConfig), 0) or np.isclose(
        v.arg(mc.NumpyConfig) + c.arg(mc.NumpyConfig), 2 * np.pi
    )


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_add(a, b):
    np.testing.assert_allclose(complex(a + b), complex(a) + complex(b))


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_sub(a, b):
    np.testing.assert_allclose(complex(a - b), complex(a) - complex(b))


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_mul(a, b):
    np.testing.assert_allclose(complex(a * b), complex(a) * complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", itertools.product(range(-1, 2), repeat=4))
def test_radd(a, bp):
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a + b), complex(a) + complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", itertools.product(range(-1, 2), repeat=4))
def test_rsub(a, bp):
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a - b), complex(a) - complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", itertools.product(range(-1, 2), repeat=4))
def test_rmul(a, bp):
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a * b), complex(a) * complex(b))


@pytest.mark.parametrize("x", [rings.ZW(p) for p in itertools.product(range(-2, 3), repeat=4)])
def test_conversion_roundtrip(x: rings.ZW):
    assert x == rings.ZW.from_pair(*x.to_zsqrt2())


def _create_random_elements(n: int, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    return [rings.ZW(c) for c in rng.integers(-10, 10, size=(n, 4))]


@pytest.mark.parametrize("x", _create_random_elements(20))
def test_complex_conjugate(x: rings.ZW):
    np.testing.assert_allclose((x * x.conj()).value(_SQRT_2), abs(x.value(_SQRT_2)) ** 2)


@pytest.mark.parametrize("x", _create_random_elements(50))
def test_sqrt2_conjugate(x: rings.ZW):
    a, b, need_w = x.to_zsqrt2()
    if need_w:
        assert x.sqrt2_conj() == rings.ZW.from_pair(a.conj(), b.conj(), False) - rings.ZW(
            [0, 1, 0, 0]
        )
    else:
        assert x.sqrt2_conj() == rings.ZW.from_pair(a.conj(), b.conj(), False)


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13])
@pytest.mark.parametrize("x", _create_random_elements(100))
def test_gcd(p, x):
    y = rings.ZW([p, 0, 0, 0])
    g = y.gcd(x)
    assert y.is_divisible_by(g)
    assert x.is_divisible_by(g)
    # The gcd in Z[W] is unique up to multiplication by a unit of Z[W].
    assert (x // g).gcd(y // g).norm() == 1


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_factor_prime(p):
    r = rings.ZW([1, 0, 0, 0])
    for ideal, exponent in rings.ZW.factor_prime(p).items():
        r = r * ideal**exponent
    if p == 2 or p % 8 == 1:
        assert all(c % p == 0 for c in r.coords)
    else:
        assert r.coords[1:] == (0, 0, 0)
        assert r.coords[0] % p == 0
