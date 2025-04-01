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

import itertools

import numpy as np
import pytest
from galois import GF, Poly

import qualtran.testing as qlt_testing
from qualtran import QGF
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
    _gf2_multiplication_symbolic,
    _gf16_multiplication,
    BinaryPolynomialMultiplication,
    GF2MulK,
    GF2Multiplication,
    GF2MulViaKaratsuba,
    GF2ShiftRight,
    MultiplyPolyByOnePlusXk,
    SynthesizeLRCircuit,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_consistent_classical_action


def test_gf16_multiplication(bloq_autotester):
    bloq_autotester(_gf16_multiplication)


def test_gf2_multiplication_symbolic(bloq_autotester):
    bloq_autotester(_gf2_multiplication_symbolic)


@pytest.mark.parametrize('m', [2, 4, 6, 8])
def test_synthesize_lr_circuit(m: int):
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
        blq = GF2MulK.from_polynomials(f_x, m_x)
        cblq = blq.decompose_bloq()
        for g in gf.elements[1:]:
            assert blq.call_classically(g=g) == cblq.call_classically(g=g)


@pytest.mark.parametrize(
    ['m_x', 'f_x', 'cnot_count'],
    [
        [Poly.Degrees([3, 1, 0]), Poly.Degrees([2, 0]), 2],
        [Poly.Degrees([3, 1, 0]), Poly.Degrees([2, 1, 0]), 5],
        [Poly.Degrees([2, 1, 0]), Poly.Degrees([1]), 1],
        [Poly.Degrees([2, 1, 0]), Poly.Degrees([0]), 0],
    ],
)
def test_multiply_by_constant_mod_cost(m_x, f_x, cnot_count):
    blq = GF2MulK.from_polynomials(f_x, m_x)
    cost = get_cost_value(blq, QECGatesCost())
    assert cost.total_t_count() == 0
    assert cost.clifford == cnot_count


@pytest.mark.parametrize('m_x', [Poly.Degrees([2, 1, 0]), Poly.Degrees([3, 1, 0])])
def test_multiply_by_constant_mod_decomposition(m_x):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = GF2MulK.from_polynomials(f_x, m_x)
        qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('m_x', [Poly.Degrees([2, 1, 0]), Poly.Degrees([3, 1, 0])])
def test_multiply_by_constant_mod_counts(m_x):
    n = len(m_x.coeffs) - 1
    gf = GF(2, n, irreducible_poly=m_x)
    QGFM = QGF(2, n)
    elements = [Poly(tuple(QGFM.to_bits(i))) for i in gf.elements[1:]]
    for f_x in elements:
        blq = GF2MulK.from_polynomials(f_x, m_x)
        qlt_testing.assert_equivalent_bloq_counts(blq, generalizer=ignore_split_join)


def test_invalid_GF2MulK_args_raises():
    gf = GF(2, 3)
    x = GF(2, 4)(1)
    with pytest.raises(AssertionError):
        _ = GF2MulK(x, gf)  # type: ignore[arg-type]


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('gf2_multiplication')


@pytest.mark.parametrize(['n', 'k'], [(n, k) for n in range(1, 6) for k in range(1, n + 2)])
def test_multiply_by_xk_decomposition(n, k):
    blq = MultiplyPolyByOnePlusXk(n, k)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize(['n', 'k'], [(n, k) for n in range(1, 6) for k in range(1, n + 2)])
def test_multiply_by_xk_bloq_counts(n, k):
    blq = MultiplyPolyByOnePlusXk(n, k)
    qlt_testing.assert_equivalent_bloq_counts(blq)


@pytest.mark.parametrize(['n', 'k'], [(n, k) for n in range(1, 4) for k in range(1, n + 2)])
def test_multiply_by_xk_classical_action(n, k):
    blq = MultiplyPolyByOnePlusXk(n, k)
    fg_polys = tuple(itertools.product(range(2), repeat=n))[1:]
    h_polys = [*itertools.product(range(2), repeat=blq.signature[-1].shape[0])]

    qlt_testing.assert_consistent_classical_action(blq, f=fg_polys, g=fg_polys, h=h_polys)


@pytest.mark.slow
@pytest.mark.parametrize(['n', 'k'], [(n, k) for n in range(4, 6) for k in range(1, n + 2)])
def test_multiply_by_xk_classical_action_slow(n, k):
    blq = MultiplyPolyByOnePlusXk(n, k)
    fg_polys = tuple(itertools.product(range(2), repeat=n))[1:]
    h_polys = [*itertools.product(range(2), repeat=blq.signature[-1].shape[0])]
    h_polys = [
        h_polys[i] for i in np.random.choice(len(h_polys), min(len(h_polys), 20), replace=False)
    ]

    qlt_testing.assert_consistent_classical_action(blq, f=fg_polys, g=fg_polys, h=h_polys)


@pytest.mark.parametrize('n', range(1, 10))
def test_binary_mult_decomposition(n):
    blq = BinaryPolynomialMultiplication(n)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('n', range(1, 10))
def test_binary_mult_bloq_counts(n):
    blq = BinaryPolynomialMultiplication(n)
    qlt_testing.assert_equivalent_bloq_counts(
        blq, generalizer=(ignore_split_join, ignore_alloc_free)
    )


@pytest.mark.parametrize('n', range(1, 4))
def test_binary_mult_classical_action(n):
    blq = BinaryPolynomialMultiplication(n)
    fg_polys = tuple(itertools.product(range(2), repeat=n))[1:]
    h_polys = [[0] * blq.signature[-1].shape[0]]

    qlt_testing.assert_consistent_classical_action(blq, f=fg_polys, g=fg_polys, h=h_polys)


# @pytest.mark.slow
@pytest.mark.parametrize('n', range(4, 7))
def test_binary_mult_classical_action_slow(n):
    blq = BinaryPolynomialMultiplication(n)
    fg_polys = tuple(itertools.product(range(2), repeat=n))[1:]
    h_polys = [[0] * blq.signature[-1].shape[0]]

    qlt_testing.assert_consistent_classical_action(blq, f=fg_polys, g=fg_polys, h=h_polys)


@pytest.mark.parametrize('log_n', [*range(10 + 1)])
def test_binary_mult_toffoli_cost(log_n):
    # Toffoli cost is n^log2(3), when n = 2^k we get (2^k)^log2(3) = 3^k
    # CNOT count is is upper bounded by (10 + 1/3) n^log2(3)
    n = 2**log_n
    blq = BinaryPolynomialMultiplication(n)
    cost = get_cost_value(blq, QECGatesCost())
    assert cost.clifford < (10 + 1 / 3) * 3**log_n
    counts = cost.total_t_and_ccz_count()
    assert counts['n_t'] == 0
    assert counts['n_ccz'] == 3**log_n


@pytest.mark.parametrize('m_x', [[1, 0], [2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])  # x + 1
@pytest.mark.parametrize('k', range(1, 5))
def test_GF2ShiftRight_decomposition(m_x, k):
    blq = GF2ShiftRight(m_x, k)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('m_x', [[1, 0], [2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])  # x + 1
@pytest.mark.parametrize('k', range(1, 5))
def test_GF2ShiftRight_bloq_counts(m_x, k):
    blq = GF2ShiftRight(m_x, k)
    qlt_testing.assert_equivalent_bloq_counts(blq, generalizer=ignore_split_join)


@pytest.mark.parametrize('m_x', [[1, 0], [2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])  # x + 1
@pytest.mark.parametrize('k', range(1, 5))
def test_GF2ShiftRight_complexity(m_x, k):
    blq = GF2ShiftRight(m_x, k)
    cost = get_cost_value(blq, QECGatesCost())
    clifford = k * (len(m_x) - 2) if len(m_x) > 2 else 0
    assert cost.clifford == clifford
    assert cost.total_t_count() == 0


@pytest.mark.parametrize('m_x', [[1, 0], [2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])  # x + 1
@pytest.mark.parametrize('k', range(1, 5))
def test_GF2ShiftRight_classical_action(m_x, k):
    blq = GF2ShiftRight(m_x, k)
    qlt_testing.assert_consistent_classical_action(blq, f=blq.gf.elements)


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])
def test_gf2mulmod_decomposition(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])
def test_gf2mulmod_bloq_counts(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    qlt_testing.assert_equivalent_bloq_counts(
        blq, generalizer=(ignore_split_join, ignore_alloc_free)
    )


@pytest.mark.parametrize('m_x', [[2, 1, 0], [8, 4, 3, 1, 0]])
def test_gf2mulmod_complexity(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    cost = get_cost_value(blq, QECGatesCost())
    # The toffoli cost is n^log2(3) .. when n = 2^k we get toffoli cost = 3^k
    n = max(m_x)
    k = n.bit_length() - 1
    assert cost.total_toffoli_only() == 3**k


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0]])
def test_gf2mulmod_classical_action(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    qlt_testing.assert_consistent_classical_action(blq, x=blq.gf.elements, y=blq.gf.elements)


@pytest.mark.slow
def test_gf2mulmod_classical_action_slow():
    m_x = [8, 4, 3, 1, 0]
    blq = GF2MulViaKaratsuba(m_x)
    xs = blq.gf.elements[np.random.choice(2**8, 10)]
    ys = blq.gf.elements[np.random.choice(2**8, 10)]
    qlt_testing.assert_consistent_classical_action(blq, x=xs, y=ys)


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0]])
def test_gf2mulmod_classical_action_adjoint(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    adjoint = blq.adjoint()
    for i, j in np.random.random_integers(0, len(blq.gf.elements) - 1, (10, 2)):
        f = blq.gf.elements[i]
        g = blq.gf.elements[j]
        a, b, c = blq.call_classically(x=f, y=g)
        a, b = adjoint.call_classically(x=a, y=b, result=c)
        assert a == f and b == g


@pytest.mark.parametrize('m_x', [[2, 1, 0], [8, 4, 3, 1, 0], [16, 5, 3, 1, 0]])
def test_gf2mulmod_classical_complexity(m_x):
    blq = GF2MulViaKaratsuba(m_x)
    cost = get_cost_value(blq, QECGatesCost()).total_t_and_ccz_count()
    assert cost['n_t'] == 0
    # The toffoli cost is n^log2(3) ... for n = 2^k we get toffli count = 3^k
    n = max(m_x)
    k = n.bit_length() - 1
    assert cost['n_ccz'] == 3**k


def test_gf2mul_invalid_input_raises():
    with pytest.raises(ValueError):
        _ = GF2MulViaKaratsuba([0, 1])  # type: ignore[arg-type]
