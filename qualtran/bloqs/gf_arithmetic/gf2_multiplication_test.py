#  Copyright 2024 Google LLC
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
from galois import GF, Poly

import qualtran.testing as qlt_testing
from qualtran import QGF
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
    _gf2_multiplication_symbolic,
    _gf16_multiplication,
    GF2Multiplication,
    MultiplyPolyByConstantMod,
    SynthesizeLRCircuit,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.testing import assert_consistent_classical_action


def test_gf16_multiplication(bloq_autotester):
    bloq_autotester(_gf16_multiplication)


def test_gf2_multiplication_symbolic(bloq_autotester):
    bloq_autotester(_gf2_multiplication_symbolic)


def test_synthesize_lr_circuit():
    m = 2
    matrix = GF2Multiplication(m).reduction_matrix_q
    bloq = SynthesizeLRCircuit(matrix)
    bloq_adj = bloq.adjoint()
    QGFM, GFM = QGF(2, m), GF(2**m)
    for i in GFM.elements:
        bloq_out = bloq.call_classically(q=np.array(QGFM.to_bits(i)))[0]
        bloq_adj_out = bloq_adj.call_classically(q=bloq_out)[0]
        assert isinstance(bloq_adj_out, np.ndarray)
        assert i == QGFM.from_bits([*bloq_adj_out])


@pytest.mark.slow
@pytest.mark.parametrize('m', [3, 4, 5])
def test_synthesize_lr_circuit_slow(m):
    matrix = GF2Multiplication(m).reduction_matrix_q
    bloq = SynthesizeLRCircuit(matrix)
    bloq_adj = bloq.adjoint()
    QGFM, GFM = QGF(2, m), GF(2**m)
    for i in GFM.elements:
        bloq_out = bloq.call_classically(q=np.array(QGFM.to_bits(i)))[0]
        bloq_adj_out = bloq_adj.call_classically(q=bloq_out)[0]
        assert isinstance(bloq_adj_out, np.ndarray)
        assert i == QGFM.from_bits([*bloq_adj_out])


def test_gf2_plus_equal_prod_classical_sim_quick():
    m = 2
    bloq = GF2Multiplication(m, plus_equal_prod=True)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements, result=GFM.elements)


@pytest.mark.slow
def test_gf2_plus_equal_prod_classical_sim():
    m = 3
    bloq = GF2Multiplication(m, plus_equal_prod=True)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements, result=GFM.elements)


def test_gf2_multiplication_classical_sim_quick():
    m = 2
    bloq = GF2Multiplication(m, plus_equal_prod=False)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements)


@pytest.mark.slow
@pytest.mark.parametrize('m', [3, 4, 5])
def test_gf2_multiplication_classical_sim(m):
    bloq = GF2Multiplication(m, plus_equal_prod=False)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements)


@pytest.mark.parametrize('m_x', [Poly.Degrees([2, 1, 0]), Poly.Degrees([3, 1, 0])])
def test_multiply_by_constant_mod_classical_action(m_x):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = MultiplyPolyByConstantMod(f_x, m_x)
        cblq = blq.decompose_bloq()
        for g_x in elements:
            g = np.zeros(n, dtype=int)
            g[-len(g_x.coeffs) :] = [int(x) for x in g_x.coeffs]
            np.testing.assert_allclose(blq.call_classically(g=g)[0], cblq.call_classically(g=g)[0])


@pytest.mark.parametrize(
    ['m_x', 'cnot_count'], [[Poly.Degrees([2, 1, 0]), 0], [Poly.Degrees([3, 1, 0]), 0]]
)
def test_multiply_by_constant_mod_cost(m_x, cnot_count):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = MultiplyPolyByConstantMod(f_x, m_x)
        cost = get_cost_value(blq, QECGatesCost())
        assert cost.total_t_count() == 0
        assert cost.clifford < n**2


@pytest.mark.parametrize('m_x', [Poly.Degrees([2, 1, 0]), Poly.Degrees([3, 1, 0])])
def test_multiply_by_constant_mod_decomposition(m_x):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = MultiplyPolyByConstantMod(f_x, m_x)
        qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('m_x', [Poly.Degrees([2, 1, 0]), Poly.Degrees([3, 1, 0])])
def test_multiply_by_constant_mod_counts(m_x):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = MultiplyPolyByConstantMod(f_x, m_x)
        qlt_testing.assert_equivalent_bloq_counts(blq)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('gf2_multiplication')
