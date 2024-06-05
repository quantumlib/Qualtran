#  Copyright 2023 Google LLC
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

from attrs import frozen

from qualtran.symbolics import SymbolicInt


@frozen
class ECPoint:
    """An elliptic curve point.

    This overrides the addition and multiplication operators to perform
    elliptic curve point addition and multiplication.

    Args:
        x: The x coordinate of the point
        y: The y coordinate of the point
        mod: The prime modulus of the field
        curve_a: The $a$ coefficient of an elliptic curve given in the standard form of
            $y^2 = x^3 + ax + b$.
    """

    x: SymbolicInt
    y: SymbolicInt
    mod: SymbolicInt
    curve_a: SymbolicInt = 0

    @classmethod
    def inf(cls, mod, curve_a):
        """The special point at infinity."""
        return cls(0, 0, mod=mod, curve_a=curve_a)

    def __neg__(self):
        return ECPoint(self.x, (-self.y) % self.mod, mod=self.mod, curve_a=self.curve_a)

    def __add__(self, other):
        if (other.mod != self.mod) or (other.curve_a != self.curve_a):
            raise ValueError('Use consistent mod and curve')

        if self == -other:
            return ECPoint.inf(mod=self.mod, curve_a=self.curve_a)
        if self == ECPoint.inf(mod=self.mod, curve_a=self.curve_a):
            return other
        if other == ECPoint.inf(mod=self.mod, curve_a=self.curve_a):
            return self

        if self == other:
            lam_num = (3 * self.x**2 + self.curve_a) % self.mod
            lam_denom = (2 * self.y) % self.mod
        else:
            lam_num = (other.y - self.y) % self.mod
            lam_denom = (other.x - self.x) % self.mod

        lam = (lam_num * pow(lam_denom, -1, mod=self.mod)) % self.mod
        xr = (lam**2 - other.x - self.x) % self.mod
        yr = (lam * (self.x - xr) - self.y) % self.mod
        return ECPoint(xr, yr, mod=self.mod, curve_a=self.curve_a)

    def __mul__(self, other):
        assert other > 0, other
        x = self
        for _ in range(other - 1):
            x = x + self

        return x

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return f'({self.x}, {self.y})'
