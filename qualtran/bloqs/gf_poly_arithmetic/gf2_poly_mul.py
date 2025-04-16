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
import numpy as np

import attrs

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QGFPoly,
    Register,
    Signature,
    QGF,
)
from qualtran.bloqs.gf_arithmetic import GF2MulViaKaratsuba, GF2Addition
from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolyJoin, GFPolySplit
from qualtran.symbolics import is_symbolic
import qualtran.bloqs.polynomials as qp
if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class MultGFPolyByOnePlusXkViaKaratsuba(qp.MultiplyPolyByOnePlusXkViaKaratsuba):
    qgf_poly: QGFPoly

    @cached_property
    def n(self):
        return self.qgf_poly.degree + 1

    @cached_property
    def coef_dtype(self):
        return self.qgf_poly.qgf
    
    def _add_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        return bb.add(GF2Addition(self.coef_dtype.bitsize), x=x, y=y)

    def _subtract_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        return bb.add(GF2Addition(self.coef_dtype.bitsize), x=x, y=y)

    def _mult_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', z: 'Soquet'):
        return bb.add(GF2MulViaKaratsuba(self.coef_dtype), x=x, y=y, z=z)

    def _mult_poly(self, bb: 'BloqBuilder', f_x: np.ndarray['Soquet'], g_x: np.ndarray['Soquet'], h_x: np.ndarray['Soquet']):
        return bb.add(MultGFPolyViaKaratsuba(self.qgf_poly), f=f_x, g=g_x, h=h_x)


@attrs.frozen
class MultGFPolyViaKaratsuba(qp.PolynomialMultiplicationViaKaratsuba):
    qgf_poly: QGFPoly

    @cached_property
    def n(self):
        return self.qgf_poly.degree + 1

    @cached_property
    def coef_dtype(self):
        return self.qgf_poly.qgf
    
    def _add_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        return bb.add(GF2Addition(self.coef_dtype.bitsize), x=x, y=y)

    def _subtract_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'):
        return bb.add(GF2Addition(self.coef_dtype.bitsize), x=x, y=y)

    def _mult_coefs(self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', z: 'Soquet'):
        return bb.add(GF2MulViaKaratsuba(self.coef_dtype), x=x, y=y, z=z)

    def _mult_poly(self, bb: 'BloqBuilder', m: int, f_x: np.ndarray['Soquet'], g_x: np.ndarray['Soquet'], h_x: np.ndarray['Soquet']):
        return bb.add(MultGFPolyByOnePlusXkViaKaratsuba(attrs.evolve(self.qgf_poly, degree=m-1)), f=f_x, g=g_x, h=h_x)

