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
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING, Union

import attrs

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QGFPoly,
    Register,
    Signature,
)
from qualtran.bloqs.gf_arithmetic import GF2Addition
from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolyJoin, GFPolySplit
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2PolyAdd(Bloq):
    r"""In place quantum-quantum addition of two polynomials defined over GF($2^m$).

    The bloq implements in place addition of quantum registers $|f(x)\rangle$ and $|g(x)\rangle$
    storing coefficients of two degree-n polynomials defined over GF($2^m$).
    Addition in GF($2^m$) simply reduces to a component wise XOR, which can be implemented via
    CNOT gates.

    $$
        |f(x)\rangle |g(x)\rangle  \rightarrow |f(x)\rangle |f(x) + g(x)\rangle
    $$

    Args:
        qgf_poly: An instance of `QGFPoly` type that defines the data type for quantum
            register $|f(x)\rangle$ storing coefficients of a degree-n polynomial defined
            over GF($2^m$).

    Registers:
        f_x: THRU register that stores coefficients of first polynomial defined over $GF(2^m)$.
        g_x: THRU register that stores coefficients of second polynomial defined over $GF(2^m)$.
    """

    qgf_poly: QGFPoly

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('f_x', dtype=self.qgf_poly), Register('g_x', dtype=self.qgf_poly)]
        )

    def is_symbolic(self):
        return is_symbolic(self.qgf_poly.degree)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, f_x: 'Soquet', g_x: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        f_x = bb.add(GFPolySplit(self.qgf_poly), reg=f_x)
        g_x = bb.add(GFPolySplit(self.qgf_poly), reg=g_x)
        for i in range(self.qgf_poly.degree + 1):
            f_x[i], g_x[i] = bb.add(GF2Addition(self.qgf_poly.qgf.bitsize), x=f_x[i], y=g_x[i])

        f_x = bb.add(GFPolyJoin(self.qgf_poly), reg=f_x)
        g_x = bb.add(GFPolyJoin(self.qgf_poly), reg=g_x)
        return {'f_x': f_x, 'g_x': g_x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {GF2Addition(self.qgf_poly.qgf.bitsize): self.qgf_poly.degree + 1}

    def on_classical_vals(self, *, f_x, g_x) -> Dict[str, 'ClassicalValT']:
        return {'f_x': f_x, 'g_x': f_x + g_x}


@bloq_example
def _gf2_poly_4_8_add() -> GF2PolyAdd:
    from qualtran import QGF, QGFPoly

    qgf_poly = QGFPoly(4, QGF(2, 3))
    gf2_poly_4_8_add = GF2PolyAdd(qgf_poly)
    return gf2_poly_4_8_add


@bloq_example
def _gf2_poly_add_symbolic() -> GF2PolyAdd:
    import sympy

    from qualtran import QGF, QGFPoly

    n, m = sympy.symbols('n, m', positive=True, integers=True)
    qgf_poly = QGFPoly(n, QGF(2, m))
    gf2_poly_add_symbolic = GF2PolyAdd(qgf_poly)
    return gf2_poly_add_symbolic


_GF2_POLY_ADD_DOC = BloqDocSpec(
    bloq_cls=GF2PolyAdd, examples=(_gf2_poly_4_8_add, _gf2_poly_add_symbolic)
)
