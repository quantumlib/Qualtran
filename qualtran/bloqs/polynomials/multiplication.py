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

import abc
from functools import cached_property
from typing import Dict, Optional, Sequence, Set, TYPE_CHECKING, Union

import attrs
import galois
import numpy as np
import sympy
from galois import GF, Poly

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QBit,
    QGF,
    Register,
    Side,
    Signature,
    QDType
)
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.symbolics import ceil, is_symbolic, log2, Shaped, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class MultiplyPolyByOnePlusXkViaKaratsuba(abc.ABC, Bloq):
    r"""Out of place multiplication of $(1 + x^k) fg$

    Applies the transformation
    $$
    \ket{f}\ket{g}\ket{h} \rightarrow \ket{f}{\ket{g}}\ket{h \oplus (1+x^k)fg}
    $$

    Note: While this construction follows Algorithm2 of https://arxiv.org/abs/1910.02849v2,
    it has a slight modification. Namely that the original construction doesn't work in
    some cases where $k < n$. However reversing the order of the first set of CNOTs (line 2)
    makes the construction work for all $k \leq n+1$.

    This construction abstracts Algorithm2 to work with polynomials of any field.

    Args:
        n: The degree of the polynomial ($2^n$ is the size of the galois field).
        k: An integer specifing the shift $1 + x^k$ (or $1 + 2^k$ for galois fields.)

    Registers:
        f: The first polynomial.
        g: The second polyonmial.
        h: The target polynomial.

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 2
    """

    k: SymbolicInt
    
    def __attrs_post_init__(self):
        if is_symbolic(self.k):
            return
        assert self.k > 0

    @cached_property
    @abc.abstractmethod
    def n(self):
        pass

    @cached_property
    @abc.abstractmethod
    def coef_dtype(self):
        pass


    @cached_property
    def l(self):
        return max(0, 2 * self.n - self.k - 1)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f', dtype=self.coef_dtype, shape=(self.n,)),
                Register('g', dtype=self.coef_dtype, shape=(self.n,)),
                Register('h', dtype=self.coef_dtype, shape=(2 * self.k + self.l,)),
            ]
        )

    @abc.abstractmethod
    def _add_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _subtract_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _mult_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', z: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _mult_poly(self, bb: 'BloqBuilder', f_x: np.ndarray['Soquet'], g_x: np.ndarray['Soquet'], h_x: np.ndarray['Soquet']):
        """"""

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'SoquetT', g: 'SoquetT', h: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        n = self.n
        k = self.k
        l = self.l
        if is_symbolic(n) or is_symbolic(k) or is_symbolic(l):
            raise DecomposeTypeError(f"symbolic decomposition is not supported for {self}")
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)
        original = len(h)
        if n > 1:
            # Note: This is the reverse order of what https://arxiv.org/abs/1910.02849v2 has.
            # The is because the reverse order to makes the construction work for k <= n+1.
            for i in reversed(range(l)):
                h[2 * k + i], h[k + i] = self._add_coefs(bb, h[2 * k + i], h[k + i])
            for i in range(k):
                h[k + i], h[i] = self._add_coefs(bb, h[k + i], h[i])

            f, g, h[k : 2 * k + l] = self._mult_poly(bb, 
                f, g, h[k : 2 * k + l]
            )
            for i in range(k):
                h[k + i], h[i] = self._subtract_coefs(bb, h[k + i], h[i])
            for i in range(l):
                h[2 * k + i], h[k + i] = self._subtract_coefs(bb, h[2 * k + i], h[k + i])
        else:
            h[k], h[0] = self._add_coefs(bb, h[k], h[0])
            (f[0], g[0]), h[k] = self._mult_coefs(bb, f[0], g[0], h[k])
            h[k], h[0] = self._subtract_coefs(h[k], h[0])

        assert len(h) == original, f'{original=} {len(h)}'
        return {'f': f, 'g': g, 'h': h}


@attrs.frozen
class PolynomialMultiplicationViaKaratsuba(Bloq):
    r"""Out of place multiplication of binary polynomial multiplication.

    Applies the transformation
    $$
    \ket{f}\ket{g}\ket{h} \rightarrow \ket{f}{\ket{g}}\ket{h \oplus fg}
    $$

    The multiplication cost of this construction is $n^{\log_2{3}}$.

    This construction abstracts Algorithm3 to work with polynomials of any field.

    Args:
        n: The degree of the polynomial ($2^n$ is the size of the galois field).

    Registers:
        f: The first polynomial.
        g: The second polyonmial.
        h: The target polynomial.

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 3
    """

    @cached_property
    @abc.abstractmethod
    def n(self):
        pass

    @cached_property
    @abc.abstractmethod
    def coef_dtype(self):
        pass

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f', dtype=self.coef_dtype, shape=(self.n,)),
                Register('g', dtype=self.coef_dtype, shape=(self.n,)),
                Register('h', dtype=self.coef_dtype, shape=(2 * self.n - 1,)),
            ]
        )

    @property
    def k(self) -> 'SymbolicInt':
        if isinstance(self.n, int):
            return (self.n + 1) >> 1
        return sympy.ceiling(self.n / 2)


    @abc.abstractmethod
    def _add_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _subtract_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _mult_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', z: 'Soquet'):
        """"""

    @abc.abstractmethod
    def _mult_poly(self, bb: 'BloqBuilder', m: int, f_x: np.ndarray['Soquet'], g_x: np.ndarray['Soquet'], h_x: np.ndarray['Soquet']):
        """"""

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'SoquetT', g: 'SoquetT', h: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k, n = self.k, self.n
        if is_symbolic(n) or is_symbolic(k):
            raise DecomposeTypeError(f"symbolic decomposition is not supported for {self}")
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)

        if n == 1:
            (f[0], g[0]), h[0] = self._mult_coefs(f[0], g[0], h[0])
            return {'f': f, 'g': g, 'h': h}

        f[:k], g[:k], h[: 3 * k - 1] = self._mult_poly(bb, k, f[:k], g[:k], h[: 3 * k - 1])
        w = 2 * k + max(0, 2 * (n - k) - k - 1)
        delta = k + w - len(h)
        if delta > 0:
            # This happens for some values (e.g. n=3) where we need to add extra qubits that will always endup in zero state.
            aux = bb.split(bb.allocate(delta, self.coef_dtype))
            h = np.concatenate([h, aux])
            assert isinstance(h, np.ndarray)

        f[k:n], g[k:n], h[k : k + w] = self._mult_poly(bb, n - k, f[k:n], g[k:n], h[k : k + w]
        )
        if delta > 0:
            aux = h[-delta:]
            h = h[:-delta]
            bb.free(bb.join(aux))

        for i in range(n - k):
            f[k + i], f[i] = self._add_coefs(bb, f[k + i], f[i])
        for i in range(n - k):
            g[k + i], g[i] = self._add_coefs(bb, g[k + i], g[i])

        f[:k], g[:k], h[k : 3 * k - 1] = bb.add(  # type: ignore[index]
            attrs.evolve(self, n=k), f=f[:k], g=g[:k], h=h[k : 3 * k - 1]
        )

        for i in range(n - k):
            g[k + i], g[i] = self._subtract_coefs(bb, g[k + i], g[i])

        for i in range(n - k):
            f[k + i], f[i] = self._subtract_coefs(bb, f[k + i], f[i])

        return {'f': f, 'g': g, 'h': h}
