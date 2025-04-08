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
import galois

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QGFPoly,
    Register,
    Signature,
)
from qualtran.bloqs.gf_arithmetic import GF2AddK
from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolyJoin, GFPolySplit
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2PolyAddK(Bloq):
    r"""In place addition of a constant polynomial defined over GF($2^m$).

    The bloq implements in place addition of a classical constant polynomial $g(x)$ and
    a quantum register $|f(x)\rangle$ storing coefficients of a degree-n polynomial defined
    over GF($2^m$). Addition in GF($2^m$) simply reduces to a component wise XOR, which can
    be implemented via X gates.

    $$
        |f(x)\rangle  \rightarrow |f(x) + g(x)\rangle
    $$

    Args:
        qgf_poly: An instance of `QGFPoly` type that defines the data type for quantum
            register $|f(x)\rangle$ storing coefficients of a degree-n polynomial defined
            over GF($2^m$).
        g_x: An instance of `galois.Poly` that specifies that constant polynomial g(x)
            defined over GF($2^m$) that should be added to the input register f(x).

    Registers:
        f_x: Input THRU register that stores coefficients of polynomial defined over $GF(2^m)$.
    """

    qgf_poly: QGFPoly
    g_x: galois.Poly = attrs.field()

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('f_x', dtype=self.qgf_poly)])

    @g_x.validator
    def _validate_g_x(self, attribute, value):
        if not is_symbolic(self.qgf_poly.degree):
            if value.degree > self.qgf_poly.degree:
                raise ValueError(f"Degree of constant polynomial must be <= {self.qgf_poly.degree}")
        if not is_symbolic(self.qgf_poly.degree, self.qgf_poly.qgf):
            if not value.field is self.qgf_poly.qgf.gf_type:
                raise ValueError(
                    f"Constant polynomial must be defined over galois field {self.qgf_poly.qgf.gf_type}"
                )

    def is_symbolic(self):
        return is_symbolic(self.qgf_poly.degree)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, f_x: 'Soquet') -> Dict[str, 'Soquet']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        f_x = bb.add(GFPolySplit(self.qgf_poly), reg=f_x)
        g_x = self.qgf_poly.to_gf_coefficients(self.g_x)
        for i in range(self.qgf_poly.degree + 1):
            f_x[i] = bb.add(GF2AddK(self.qgf_poly.qgf.bitsize, int(g_x[i])), x=f_x[i])

        f_x = bb.add(GFPolyJoin(self.qgf_poly), reg=f_x)
        return {'f_x': f_x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if self.is_symbolic():
            k = ssa.new_symbol('g_x')
            return {GF2AddK(self.qgf_poly.qgf.bitsize, k): self.qgf_poly.degree + 1}
        return super().build_call_graph(ssa)

    def on_classical_vals(self, *, f_x) -> Dict[str, 'ClassicalValT']:
        return {'f_x': f_x + self.g_x}


@bloq_example
def _gf2_poly_4_8_add_k() -> GF2PolyAddK:
    from galois import Poly

    from qualtran import QGF, QGFPoly

    qgf_poly = QGFPoly(4, QGF(2, 3))
    g_x = Poly(qgf_poly.qgf.gf_type([1, 2, 3, 4, 5]))
    gf2_poly_4_8_add_k = GF2PolyAddK(qgf_poly, g_x)
    return gf2_poly_4_8_add_k


@bloq_example
def _gf2_poly_add_k_symbolic() -> GF2PolyAddK:
    import sympy
    from galois import Poly

    from qualtran import QGF, QGFPoly

    n, m = sympy.symbols('n, m', positive=True, integers=True)
    qgf_poly = QGFPoly(n, QGF(2, m))
    gf2_poly_add_k_symbolic = GF2PolyAddK(qgf_poly, Poly([0, 0, 0, 0]))
    return gf2_poly_add_k_symbolic


_GF2_POLY_ADD_K_DOC = BloqDocSpec(
    bloq_cls=GF2PolyAddK, examples=(_gf2_poly_4_8_add_k, _gf2_poly_add_k_symbolic)
)
