#  Copyright 2026 Google LLC
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

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QGF
from qualtran.symbolics import ceil, is_symbolic, log2


def test_qgf():
    qgf_256 = QGF(characteristic=2, degree=8)
    assert str(qgf_256) == 'QGF(2**8)'
    assert qgf_256.num_qubits == 8
    p, m = sympy.symbols('p, m', integer=True, positive=True)
    qgf_pm = QGF(characteristic=p, degree=m)
    assert qgf_pm.num_qubits == ceil(log2(p**m))
    assert is_symbolic(qgf_pm)


def test_qgf_to_and_from_bits():
    from galois import GF

    qgf_256 = QGF(2, 8)
    gf256 = GF(2**8)
    assert [*qgf_256.get_classical_domain()] == [*range(256)]
    a, b = qgf_256.to_bits(gf256(21)), qgf_256.to_bits(gf256(22))
    c = qgf_256.from_bits(list(np.bitwise_xor(a, b)))
    assert c == gf256(21) + gf256(22)
    for x in gf256.elements:
        assert x == gf256.Vector(qgf_256.to_bits(x))

    with pytest.raises(ValueError):
        qgf_256.to_bits(21)  # type: ignore[arg-type]
    assert_to_and_from_bits_array_consistent(qgf_256, gf256([*range(256)]))


def test_qgf_with_default_poly_is_compatible():
    qgf_one = QGF(2, 4)

    qgf_two = QGF(2, 4, irreducible_poly=qgf_one.gf_type.irreducible_poly)

    assert qgf_one == qgf_two
