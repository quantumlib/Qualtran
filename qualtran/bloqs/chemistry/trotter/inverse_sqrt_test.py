#  Copyright 2023 Google LLC
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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.trotter.inverse_sqrt import (
    _nr_inv_sqrt,
    _poly_inv_sqrt,
    build_qrom_data_for_poly_fit,
    get_inverse_square_root_poly_coeffs,
    NewtonRaphsonApproxInverseSquareRoot,
    PolynmomialEvaluationInverseSquareRoot,
)
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_fixed_point


def test_newton_raphson_inverse_sqrt(bloq_autotester):
    bloq_autotester(_nr_inv_sqrt)


def test_poly_eval_inverse_sqrt(bloq_autotester):
    bloq_autotester(_poly_inv_sqrt)


def test_newton_raphson_inverse_sqrt_bloq_counts():
    bloq = NewtonRaphsonApproxInverseSquareRoot(7, 8, 12)
    _, counts = bloq.call_graph()
    assert counts[TGate()] == 1632


def test_poly_eval_inverse_sqrt_bloq_counts():
    bloq = PolynmomialEvaluationInverseSquareRoot(7, 8, 12)
    _, counts = bloq.call_graph()
    assert counts[TGate()] == 744


def fixed_point_to_float(x: int, width: int) -> float:
    bits = iter_bits(int(x), width)
    approx_val = np.sum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
    return approx_val


@pytest.mark.parametrize("bitsize, poly_bitsize", ((3, 32), (6, 15), (8, 30), (7, 12)))
def test_build_qrom_data(bitsize, poly_bitsize):
    poly_coeffs = get_inverse_square_root_poly_coeffs()
    qrom_data = build_qrom_data_for_poly_fit(bitsize, poly_bitsize, poly_coeffs)
    poly_coeffs_a, poly_coeffs_b = get_inverse_square_root_poly_coeffs()
    for c in range(4):
        unique = np.unique(qrom_data[c], axis=0)[::-1]
        assert len(unique) == 2 * bitsize
        a_bits = unique[::2]
        coeff_as_float = [fixed_point_to_float(c, poly_bitsize) for c in a_bits]
        for k in range(2, len(unique) // 2):
            np.isclose(
                coeff_as_float[k], poly_coeffs_a[c] / 2 ** (k / 2), atol=1 / 2**poly_bitsize
            )
        b_bits = unique[1::2]
        coeff_as_float = [fixed_point_to_float(c, poly_bitsize) for c in b_bits]
        for k in range(2, len(unique) // 2):
            np.isclose(
                coeff_as_float[k], poly_coeffs_b[c] / 2 ** (k / 2), atol=1 / 2**poly_bitsize
            )


def multiply_fixed_point_float_by_int(
    fp_int: int, intg: int, width_float: int, width_int: int
) -> int:
    assert width_float >= width_int
    result = 0
    for l, lambda_l in enumerate(f"{intg:0{width_int}b}"):
        for k, kappa_k in enumerate(f"{fp_int:0{width_float}b}"):
            result += int(lambda_l) * int(kappa_k) * 2 ** (width_int + width_float - l - k - 2)
    return result


def multiply_fixed_point_floats(a: int, b: int, width: int) -> int:
    result = 0
    for l, lambda_l in enumerate(f"{a:0{width}b}"):
        for k, kappa_k in enumerate(f"{b:0{width}b}"):
            if k + l + 2 <= width:
                result += int(lambda_l) * int(kappa_k) * 2 ** ((width - k - l - 2))
    return result


def test_multiply_float_int():
    float_width = 24
    int_width = 8
    val = np.random.random()
    fp_bits = iter_bits_fixed_point(val, float_width)
    fp_int = int(''.join(str(b) for b in fp_bits), 2)
    int_val = np.random.randint(0, 2**int_width - 1)
    result = multiply_fixed_point_float_by_int(fp_int, int_val, float_width, int_width)
    assert abs(result / 2**float_width - int_val * val) <= int_width * 2 ** (
        int_width - float_width
    )


def test_multiply_floats():
    float_width = 24
    a = np.random.random()
    b = np.random.random()
    bits = iter_bits_fixed_point(a, float_width)
    fp_a = int(''.join(str(b) for b in bits), 2)
    bits = iter_bits_fixed_point(b, float_width)
    fp_b = int(''.join(str(b) for b in bits), 2)
    result = multiply_fixed_point_floats(fp_a, fp_b, float_width)
    assert abs(result / 2**float_width - a * b) <= (float_width + 1) / 2**float_width
