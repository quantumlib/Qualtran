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

import sympy

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QGF, QGFPoly
from qualtran.symbolics import ceil, is_symbolic, log2


def test_qgf_poly():
    qgf_poly_4_8 = QGFPoly(4, QGF(characteristic=2, degree=3))
    assert str(qgf_poly_4_8) == 'QGFPoly(4, QGF(2**3))'
    assert qgf_poly_4_8.num_qubits == 5 * 3
    n, p, m = sympy.symbols('n, p, m', integer=True, positive=True)
    qgf_poly_n_pm = QGFPoly(n, QGF(characteristic=p, degree=m))
    assert qgf_poly_n_pm.num_qubits == (n + 1) * ceil(log2(p**m))
    assert is_symbolic(qgf_poly_n_pm)


def test_qgf_poly_to_and_from_bits():
    qgf_4 = QGF(2, 2)
    qgf_poly_2_4 = QGFPoly(2, qgf_4)
    assert_to_and_from_bits_array_consistent(qgf_poly_2_4, [*qgf_poly_2_4.get_classical_domain()])
