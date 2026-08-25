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

pytest.importorskip('galois')

from qualtran.dtype import assert_to_and_from_bits_array_consistent, CGF, QGF
from qualtran.dtype.gf._gf import _GF
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


def test_qgf_domain_and_validation_arr():
    qgf = QGF(2, 8)
    arr = np.array(list(qgf.get_classical_domain()))
    qgf.assert_valid_classical_val_array(arr)


def test_qgf_validation_errs():
    with pytest.raises(ValueError):
        QGF(2, 8).assert_valid_classical_val(2**8)  # type: ignore[arg-type]


def test_qgf_equality_and_hashing():
    import galois

    q1 = QGF(2, 8)
    q2 = QGF(2, 8)
    p_deg = galois.Poly.Degrees([8, 4, 3, 2, 0])
    q3 = QGF(2, 8, irreducible_poly=p_deg)
    p_diff = galois.Poly.Degrees([8, 4, 3, 1, 0])
    q4 = QGF(2, 8, irreducible_poly=p_diff)
    q5 = QGF(2, 7)
    q6 = QGF(3, 3)

    assert q1 == q2
    assert q1 == q3
    assert q1 != q4
    assert q1 != q5
    assert q1 != q6
    assert hash(q1) == hash(q2)
    assert hash(q1) == hash(q3)
    assert hash(q1) != hash(q4)

    # Distinct types
    cgf = CGF(2, 8)
    assert q1 != cgf
    gf_enc = _GF(2, 8)
    assert q1 != gf_enc
    assert hash(q1) != hash(cgf)

    # Set / dict lookup
    s = {q1, q4, q5, cgf}
    assert q2 in s
    assert q3 in s
    assert len(s) == 4

    # Symbolic equality & hashing
    p, m = sympy.symbols('p, m', integer=True, positive=True)
    sym_q1 = QGF(characteristic=p, degree=m)
    sym_q2 = QGF(characteristic=p, degree=m)
    sym_q3 = QGF(characteristic=p, degree=m + 1)
    assert sym_q1 == sym_q2
    assert sym_q1 != sym_q3
    assert hash(sym_q1) == hash(sym_q2)
