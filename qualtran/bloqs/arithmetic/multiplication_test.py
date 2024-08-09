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

import cirq
import numpy as np
import pytest

from qualtran import BloqBuilder, QUInt, Register
from qualtran.bloqs.arithmetic.multiplication import (
    _invert_real_number,
    _multiply_two_reals,
    _plus_equal_product,
    _product,
    _scale_int_by_real,
    _square,
    _square_real_number,
    _sum_of_squares,
    MultiplyTwoReals,
    PlusEqualProduct,
    Product,
    ScaleIntByReal,
    Square,
    SquareRealNumber,
    SumOfSquares,
)
from qualtran.bloqs.arithmetic.subtraction import Subtract
from qualtran.bloqs.basic_gates import CNOT, IntState, Toffoli, XGate
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.symbolics import HasLength
from qualtran.testing import execute_notebook


def test_square_auto(bloq_autotester):
    bloq_autotester(_square)


def test_sum_of_squares_auto(bloq_autotester):
    bloq_autotester(_sum_of_squares)


def test_product_auto(bloq_autotester):
    bloq_autotester(_product)


def test_scale_int_by_real_auto(bloq_autotester):
    bloq_autotester(_scale_int_by_real)


def test_multiply_two_reals_auto(bloq_autotester):
    bloq_autotester(_multiply_two_reals)


def test_square_real_number_auto(bloq_autotester):
    bloq_autotester(_square_real_number)


def test_plus_equals_product_auto(bloq_autotester):
    bloq_autotester(_plus_equal_product)


def test_invert_real_number_auto(bloq_autotester):
    bloq_autotester(_invert_real_number)


def test_square():
    bb = BloqBuilder()
    bitsize = 4

    q0 = bb.add(IntState(10, bitsize))
    q0, q1 = bb.add(Square(bitsize), a=q0)
    cbloq = bb.finalize(val=q0, result=q1)
    cbloq.t_complexity()
    assert cbloq.on_classical_vals() == {'val': 10, 'result': 100}
    assert cbloq.tensor_contract().reshape(2**bitsize, 4**bitsize)[10, 100] == 1
    num_toff = 12
    assert t_complexity(Square(bitsize)) == TComplexity(t=4 * num_toff)

    bb = BloqBuilder()
    val, result = bb.add_from(cbloq)
    a = bb.add(Square(bitsize).adjoint(), a=val, result=result)
    cbloq = bb.finalize(a=a)
    assert cbloq.on_classical_vals(a=10, result=100) == {'a': 10}
    assert cbloq.tensor_contract()[10] == 1


def test_sum_of_squares():
    bb = BloqBuilder()
    bitsize = 4
    k = 3
    inp = bb.add_register(Register("input", QUInt(bitsize=bitsize), shape=(k,)))
    assert inp is not None
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp)
    cbloq = bb.finalize(input=inp, result=out)
    assert SumOfSquares(bitsize, k).signature[1].bitsize == 2 * bitsize + 2
    num_toff = k * bitsize**2 - bitsize - 1
    assert t_complexity(cbloq) == TComplexity(t=4 * num_toff)


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    num_toff = 2 * bitsize * mbits - max(bitsize, mbits)
    assert t_complexity(cbloq) == TComplexity(t=4 * num_toff)


def test_scale_int_by_real():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 8)
    q0, q1, q2 = bb.add(ScaleIntByReal(15, 8), real_in=q0, int_in=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    num_toff = 15 * (2 * 8 - 1) - 8**2
    assert t_complexity(cbloq) == TComplexity(t=4 * num_toff)


def test_multiply_two_reals():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(MultiplyTwoReals(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    num_toff = 15**2 - 15 - 1
    assert t_complexity(cbloq) == TComplexity(t=4 * num_toff)


def test_square_real_number():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(SquareRealNumber(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    num_toff = 15**2 // 2 - 4
    assert t_complexity(cbloq) == TComplexity(t=4 * num_toff)


def test_plus_equal_product():
    a_bit, b_bit, res_bit = 2, 2, 4
    num_bits = a_bit + b_bit + res_bit
    bloq = PlusEqualProduct(a_bit, b_bit, res_bit)
    basis_map = {}
    for a in range(2**a_bit):
        for b in range(2**b_bit):
            for result in range(2**res_bit):
                res_out = (result + a * b) % 2**res_bit
                # Test Bloq style classical simulation.
                assert bloq.call_classically(a=a, b=b, result=result) == (a, b, res_out)
                # Prepare basis states mapping for cirq-style simulation.
                input_state_str = f'{a:0{a_bit}b}' + f'{b:0{b_bit}b}' + f'{result:0{res_bit}b}'
                input_state = int(input_state_str, 2)
                output_state_str = f'{a:0{a_bit}b}' + f'{b:0{b_bit}b}' + f'{res_out:0{res_bit}b}'
                output_state = int(output_state_str, 2)
                basis_map[input_state] = output_state
    # Test cirq style simulation.
    assert len(basis_map) == len(set(basis_map.values()))
    circuit = cirq.Circuit(bloq.on(*cirq.LineQubit.range(num_bits)))
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)

    # TODO: The T-complexity here is approximate.
    assert t_complexity(bloq) == TComplexity(t=8 * max(a_bit, b_bit) ** 2)


def test_invert_real_number():
    bitsize = 10
    num_frac = 7
    num_int = bitsize - num_frac
    num_iters = int(np.ceil(np.log2(bitsize)))
    bloq = _invert_real_number()
    cost = (
        Toffoli().t_complexity() * (num_int - 1)
        + CNOT().t_complexity() * (2 + num_int - 1)
        + MultiControlX(cvs=HasLength(num_int)).t_complexity()
        + XGate().t_complexity()
        + num_iters * SquareRealNumber(bitsize).t_complexity()
        + num_iters * MultiplyTwoReals(bitsize).t_complexity()
        + num_iters * ScaleIntByReal(bitsize, 2).t_complexity()
        + num_iters * Subtract(QUInt(bitsize)).t_complexity()
    )
    assert bloq.t_complexity() == cost


@pytest.mark.notebook
def test_multiplication_notebook():
    execute_notebook('multiplication')
