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

from typing import Optional

import attrs
from sympy import ntheory

from qualtran.rotation_synthesis.rings import _zsqrt2, _zw


@attrs.frozen
class IdealInfo:
    """Holds the result of Algorithm 3 of https://arxiv.org/abs/2203.10064"""

    ideal: _zw.ZW
    chi: _zw.ZW
    eta: _zsqrt2.ZSqrt2
    is_inert: bool


def factor_into_generators(
    r: _zsqrt2.ZSqrt2, etas: list[_zsqrt2.ZSqrt2]
) -> tuple[_zsqrt2.ZSqrt2, list[int]]:
    r"""factors $r$ into the given $\eta_i$s

    The function finds $u$ and maximal $e_i$ such that
    $$
        r = u \prod \eta_i^{e_i}
    $$
    """
    exponents = []
    for eta in etas:
        mag = eta.norm()
        assert mag > 1, f"{eta=}"
        e = 0
        while r.is_divisible_by(eta):
            r = r // eta
            e += 1
        exponents.append(e)
    return r, exponents


class CliffordTRelativeNormSolver:
    r"""A relative norm solver for the Clifford+T Gateset.

    The class implements the relative norm solver described in section Algorithm 2 in
    https://arxiv.org/abs/2203.10064. For Clifford+T, $O_K = \mathbb{Z}[\sqrt{2}]$ and
    $O_L = \mathbb{Z}[e^{i pi/4}]$.
    """

    def _get_root(self, chi: _zw.ZW) -> Optional[_zw.ZW]:
        r"""Returns a power of $\omega = \zeta_8$ such that makes $\chi$ a prime ideal or None.

        This function finds if there is a a unit $v$ in the quotient group of units of
        $\mathbb{Z}[e^{i pi/4}]$ and $\mathbb{Z}[\sqrt{2}]$ such that $v \chi$ is a prime ideal
        of $\mathbb{Z}[\sqrt{2}]$. The quotient group contains the powers of $\omega$.
        """
        for v in _zw.One, _zw.Omega, _zw.Omega**2, _zw.Omega**3:
            r = v * chi
            a, b, need_w = r.to_zsqrt2()
            if need_w:
                continue
            if b != _zsqrt2.Zero:
                continue
            if a.is_prime_ideal():
                return v
        return None

    def compute_w(self, unit: _zsqrt2.ZSqrt2) -> Optional[_zw.ZW]:
        r"""Finds a unit in $\mathbb{Z}[e^{i pi/4}]$ such that $ww^*$ equals the given unit.


        units of $\mathbb{Z}[e^{i pi/4}]$ have the form $\omega^n (1 + \sqrt{2})^m$ where
        for $m, n \in \mathbb{Z}$ and $\omega = \zeta_8 = e^{i \pi/4}$.
        units of $\mathbb{Z}[\sqrt{2}]$ have the form $(-1)^n (1 + \sqrt{2})^m$, so a solution
        only exists if the given unit can be written as an even power of $1 + \sqrt{2}$

        Args:
            unit: A unit in $\mathbb{Z}[\sqrt{2}]$
        Returns:
            A unit $\mathbb{Z}[e^{i pi/4}]$ such that $ww^*$ or None if it doesn't exist.
        """
        coefs = [unit.a, unit.b]
        sign = 1 - 2 * any(c < 0 for c in coefs)
        target = abs(coefs[1])
        if target <= 1:
            n = target
        else:
            # iteratively compute successive powers of 1+sqrt(2)
            a, b = 0, 1
            n = 1
            while b < target:
                a, b = b, 2 * b + a
                n += 1
            assert b == target, f"{b=} {target=} {unit=}"
        if n % 2 == 1:
            return None
        n >>= 1
        if sign == -1:
            return (-_zw.One + _zw.SQRT_2) ** n
        return (_zw.One + _zw.SQRT_2) ** n

    def solve(self, r: _zsqrt2.ZSqrt2) -> Optional[_zw.ZW]:
        r"""Returns a solution if it exists or None.

        Args:
            r: An element of Z[sqrt(2)]

        Returns:
            $v \in \mathbb{Z}[e^{i pi/4}]$ such that $v^*v = r$ if it exists
            or None.
        """
        if r == _zsqrt2.Zero:
            return _zw.Zero
        if r == _zsqrt2.One:
            return _zw.One
        norm = r.norm()
        info = []
        for p in ntheory.factorint(norm):
            for ideal in _zw.ZW.factor_prime(p):
                chi = ideal.gcd(_zw.ZW((p, 0, 0, 0)))
                v = self._get_root(chi)
                if v is None:
                    eta = chi * chi.conj()
                else:
                    eta = v * chi
                info.append(IdealInfo(ideal, chi, eta.to_zsqrt2()[0], v is not None))

        unit, exponents = factor_into_generators(r, [i.eta for i in info])
        w = self.compute_w(unit)
        if w is None:
            return None

        m = w
        for e, ideal_info in zip(exponents, info):
            if e == 0:
                continue
            if ideal_info.is_inert:
                if e % 2 == 0:
                    eta = _zw.ZW.from_pair(ideal_info.eta, _zsqrt2.Zero, False)
                    m = m * eta ** (e >> 1)
                else:
                    return None
            else:
                m = m * ideal_info.chi**e
        return m
