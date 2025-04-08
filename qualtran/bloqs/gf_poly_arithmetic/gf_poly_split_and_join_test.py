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

import numpy as np
import pytest
import sympy
from galois import Poly

from qualtran import QGF, QGFPoly
from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
    _gf_poly_join,
    _gf_poly_split,
    GFPolyJoin,
    GFPolySplit,
)


def test_gf_poly_split(bloq_autotester):
    bloq_autotester(_gf_poly_split)


def test_gf_poly_join(bloq_autotester):
    bloq_autotester(_gf_poly_join)


def test_no_symbolic_degree():
    n = sympy.Symbol('n')
    with pytest.raises(ValueError, match=r'.*cannot have a symbolic data type\.'):
        GFPolySplit(QGFPoly(n, QGF(2, 3)))

    with pytest.raises(ValueError, match=r'.*cannot have a symbolic data type\.'):
        GFPolyJoin(QGFPoly(n, QGF(2, 3)))


def test_classical_sim():
    bloq = _gf_poly_split.make()
    p = Poly(bloq.dtype.qgf.gf_type([1, 2, 3, 4]))
    coeffs = bloq.call_classically(reg=p)[0]  # type: ignore[arg-type]
    assert np.all(coeffs == [0, 1, 2, 3, 4])
    assert bloq.adjoint().call_classically(reg=coeffs)[0] == p  # type: ignore[arg-type]


def test_tensor_sim():
    bloq = GFPolySplit(QGFPoly(2, QGF(2, 2)))
    assert np.all(bloq.tensor_contract() == np.eye(2 ** (3 * 2)))
    assert np.all(bloq.adjoint().tensor_contract() == np.eye(2 ** (3 * 2)))
