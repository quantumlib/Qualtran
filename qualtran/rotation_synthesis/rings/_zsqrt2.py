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

r"""The ring $\mathbb{Z}[\sqrt{2}]$."""

import fractions
import functools
import itertools

import attrs
import numpy as np
from sympy import ntheory

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis._typing as rst

_SQRT_2 = float(np.sqrt(2))


@attrs.frozen(hash=True)
class ZSqrt2:  # noqa: PLW1641 (false positive)
    r"""Elements of the ring $\mathbb{Z}[\sqrt{2}]$.

    Elements of the ring $\mathbb{Z}[\sqrt{2}]$ are represented by $a+b\sqrt{2}$
    for integer $a$ and $b$.

    Attributes:
        a: the integer part of the element.
        b: the coefficients of $\sqrt{2}$.
    """

    a: int = attrs.field(converter=int)
    b: int = attrs.field(converter=int, default=0)

    def __float__(self) -> float:
        return float(self.value(_SQRT_2))

    def value(self, sqrt2: rst.Real) -> rst.Real:
        """Return the floating-point value of the element.

        The method uses the given value of sqrt2 (e.g. np.sqrt(2), mpmath.sqrt(2), ...etc)
        to control the precision of the result.
        """
        return self.a + sqrt2 * self.b

    def __mul__(self, other) -> "ZSqrt2":
        if rst.is_int(other):
            return ZSqrt2(self.a * other, self.b * other)
        if isinstance(other, ZSqrt2):
            return ZSqrt2(
                self.a * other.a + 2 * self.b * other.b, self.a * other.b + self.b * other.a
            )
        return NotImplemented

    def __rmul__(self, other) -> "ZSqrt2":
        if rst.is_int(other):
            return self * other
        return NotImplemented

    @functools.lru_cache(512)
    def _power(self, p: int) -> "ZSqrt2":
        if p == 0:
            return One
        y = One
        x = self
        while p > 1:
            if p & 1:
                y = y * x
            x = x * x
            p >>= 1
        return x * y

    def __pow__(self, other) -> "ZSqrt2":
        assert rst.is_int(other) and other >= 0
        return self._power(int(other))

    def __add__(self, other) -> "ZSqrt2":
        if rst.is_int(other):
            return ZSqrt2(self.a + other, self.b)
        if isinstance(other, ZSqrt2):
            return ZSqrt2(self.a + other.a, self.b + other.b)
        return NotImplemented

    def __sub__(self, other) -> "ZSqrt2":
        if rst.is_int(other):
            return ZSqrt2(self.a - other, self.b)
        if isinstance(other, ZSqrt2):
            return ZSqrt2(self.a - other.a, self.b - other.b)
        return NotImplemented

    def conjugate(self) -> "ZSqrt2":
        """Returns the sqrt2-conjugate"""
        return ZSqrt2(self.a, -self.b)

    def conj(self) -> "ZSqrt2":
        """Returns the sqrt2-conjugate"""
        return self.conjugate()

    def __neg__(self) -> "ZSqrt2":
        return ZSqrt2(-self.a, -self.b)

    def __floordiv__(self, other: "ZSqrt2") -> "ZSqrt2":
        assert isinstance(other, ZSqrt2)
        res = self * other.conj()
        norm = (other * other.conj()).a
        return ZSqrt2(res.a // norm, res.b // norm)

    def is_divisible_by(self, other: "ZSqrt2") -> bool:
        assert isinstance(other, ZSqrt2)
        res = self * other.conj()
        norm = other.norm()
        return res.a % norm == 0 and res.b % norm == 0

    def divide_by_sqrt2(self) -> "ZSqrt2":
        r"""Performs division by sqrt(2).

        Dividing $a+b\sqrt{2}$ by sqrt(2) is only possible when the integer part is even.
        """
        if self.a % 2 == 1:
            raise ValueError(
                f"Division by sqrt(2) of Z[sqrt(2)] is only possible when a is even {self}"
            )
        return ZSqrt2(self.b, self.a // 2)

    def __eq__(self, other) -> bool:
        if isinstance(other, ZSqrt2):
            return self.a == other.a and self.b == other.b
        if rst.is_int(other):
            return self == ZSqrt2(other, 0)
        raise TypeError(f"Comparison is not supported between ZSqrt2 and {type(other)}")

    def __lt__(self, other) -> bool:
        # comparison is done using only integer operations.
        if isinstance(other, ZSqrt2):
            if self.b == other.b:
                return self.a < other.a
            # self.a + self.b * sqrt(2) < other.a + other.b * sqrt(2) imply that either
            #   - ((self.a - other.a) / (other.b - self.b))^2 <= 2
            #   - ((self.a - other.a) / (other.b - self.b))^2 >= 2
            # depending on sign of the numerator and denominator.
            da = self.a - other.a
            db = other.b - self.b
            f = fractions.Fraction(da, db)
            if db < 0:
                if da < 0:
                    return f**2 > 2
                return False
            if da < 0:
                return True
            return f**2 < 2
        if rst.is_int(other):
            return self < ZSqrt2(other, 0)

        raise TypeError(f"Cannot compare ZSqrt2 with {type(other)}")

    def __le__(self, other) -> bool:
        return (self == other) or (self < other)

    def __gt__(self, other) -> bool:
        if rst.is_int(other):
            other = ZSqrt2(other, 0)
        return other < self

    def __ge__(self, other) -> bool:
        return (self == other) or (self > other)

    def gcd(self, other: "ZSqrt2") -> "ZSqrt2":
        """Returns the gcd between the operands using the euclidan algorithm."""
        if self == other == Zero:
            return One
        x = self
        y = other
        while True:
            if x.norm() > y.norm():
                x, y = y, x
            if x == Zero:
                break
            c = y // x
            best = (y.norm(), y)
            for dc in itertools.product(range(2), repeat=2):
                r = c + ZSqrt2(*dc)
                ny = y - r * x
                if ny.norm() < best[0]:
                    best = ny.norm(), ny
            assert best[0] < y.norm()
            y = best[1]
        return y

    @staticmethod
    def factor_prime(p: int) -> dict["ZSqrt2", int]:
        """Factors a prime into the prime ideals of the ring.

        This method follows lemma C.12 from https://arxiv.org/abs/1403.2975.
        """
        assert p > 1
        if p == 2:
            return {ZSqrt2(0, 1): 2}
        if p % 8 in (3, 5):
            return {ZSqrt2(p, 0): 1}
        assert p % 8 in (1, 7)
        x = ntheory.sqrt_mod(2, p)
        assert x is not None
        f = ZSqrt2(x, 1).gcd(ZSqrt2(p, 0))
        sgn = 1
        if (f * f.conj()) == -p:
            sgn = -1
        return {f: 1, sgn * f.conj(): 1}

    def norm(self) -> int:
        r"""Returns the norm of element $N(a + b\sqrt{2}) = |a^2 - 2b^2|$"""
        return abs(self.a**2 - 2 * self.b**2)

    def is_prime_ideal(self) -> bool:
        """Returns whether the element is a prime ideal or not.

        The prime ideals are the factors of the regular primes in the ring:
            - 2 is factored as sqrt(2)^2 => sqrt(2) is prime.
            - primes p = 3 or 5 mod 8 can't be factored in this ring => p is prime.
            - primes p = 1 or 7 mod 8 get factored as x * x.conj() => x and x.conj are primes.
        """
        if self == SQRT_2:
            return True
        if self.b == 0 and self.a % 8 in (3, 5):
            # self is an integer = 3 or 5 mod 8 => a prime ideal only if it is a regular prime.
            return ntheory.isprime(abs(self.a))
        # The element is a prime ideal only if is norm is prime = 1 or 7 mod 8.
        p = abs((self * self.conj()).a)
        return p % 8 in (1, 7) and ntheory.isprime(p)


Zero = ZSqrt2(0, 0)
One = ZSqrt2(1, 0)
SQRT_2 = ZSqrt2(0, 1)
LAMBDA = ZSqrt2(1, 1)
LAMBDA_INV = ZSqrt2(-1, 1)
LAMBDA_CONJ = LAMBDA.conj()


LAMBDA_KLIUCHNIKOV = ZSqrt2(2, 1)
LAMBDA_KLIUCHNIKOV_CONJ = LAMBDA_KLIUCHNIKOV.conj()


@functools.cache
def radius2_at_n(x: ZSqrt2, n: int, config: mc.MathConfig) -> rst.Real:
    return (2 * x**n).value(config.sqrt2)


@functools.cache
def radius_at_n(x: ZSqrt2, n: int, config: mc.MathConfig) -> rst.Real:
    return config.sqrt(radius2_at_n(x, n, config))
