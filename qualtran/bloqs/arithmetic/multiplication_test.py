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
from fxpmath import Fxp

from qualtran import BloqBuilder, QFxp, QUInt, Register
from qualtran._infra.data_types import val_to_fxp
from qualtran.bloqs.arithmetic.multiplication import (
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
from qualtran.bloqs.basic_gates import IntState
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


def test_square():
    bb = BloqBuilder()
    bitsize = 4

    q0 = bb.add(IntState(10, bitsize))
    q0, q1 = bb.add(Square(bitsize), a=q0)
    cbloq = bb.finalize(val=q0, result=q1)
    cbloq.t_complexity()
    assert cbloq.on_classical_vals() == {'val': 10, 'result': 100}
    assert cbloq.tensor_contract().reshape(2**bitsize, 4**bitsize)[10, 100] == 1

    bb = BloqBuilder()
    val, result = bb.add_from(cbloq)
    a = bb.add(Square(bitsize).adjoint(), a=val, result=result)
    cbloq = bb.finalize(a=a)
    assert cbloq.on_classical_vals(a=10, result=100) == {'a': 10}
    assert cbloq.tensor_contract()[10] == 1


def test_sum_of_squares():
    bitsize = 4
    k = 3
    bb = BloqBuilder()
    inp = bb.add_register(Register("input", QUInt(bitsize=bitsize), shape=(k,)))
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp)
    cbloq = bb.finalize(input=inp, result=out)
    assert SumOfSquares(bitsize, k).signature[1].bitsize == 2 * bitsize + 2
    assert cbloq.t_complexity().t == 172
    cvals = cbloq.on_classical_vals(input=[2, 2, 2])
    assert (cvals['input'] == [2, 2, 2]).all()
    assert cvals['result'] == 12


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()
    cvals = cbloq.on_classical_vals(a=2, b=3)
    assert cvals['result'] == 6
    bb = BloqBuilder()
    q0 = bb.add(IntState(10, bitsize))
    q1 = bb.add(IntState(4, mbits))
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    assert cbloq.on_classical_vals() == {'a': 10, 'b': 4, 'result': 40}


def test_scale_int_by_real():
    bb = BloqBuilder()
    q0 = bb.add_register_from_dtype('a', QFxp(15, 15))
    q1 = bb.add_register_from_dtype('b', QUInt(8))
    q0, q1, q2 = bb.add(ScaleIntByReal(15, 8), real_in=q0, int_in=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()
    a_fxp = val_to_fxp(0.554, num_bits=15 + 8 - 1, num_frac=15)
    cvals = cbloq.on_classical_vals(a=a_fxp, b=3)
    assert cvals['result'] == Fxp(1.6484375, dtype='fxp-u15/7')
    bb = BloqBuilder()
    q0 = bb.add_register_from_dtype('a', QFxp(15, 15))
    q1 = bb.add_register_from_dtype('b', QUInt(8))
    q0, q1, q2 = bb.add(ScaleIntByReal(15, 8), real_in=q0, int_in=q1)
    q0, q1 = bb.add(ScaleIntByReal(15, 8).adjoint(), real_in=q0, int_in=q1, result=q2)
    cbloq = bb.finalize(a=q0, b=q1)
    assert cbloq.on_classical_vals(a=a_fxp, b=3) == {'a': a_fxp, 'b': 3}


def test_multiply_two_reals():
    bb = BloqBuilder()
    bitsize = 15
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    q0, q1, q2 = bb.add(MultiplyTwoReals(bitsize), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()
    a = np.pi / 7
    b = 0.332
    a_fxp = val_to_fxp(a, num_bits=bitsize, num_frac=bitsize)
    b_fxp = val_to_fxp(b, num_bits=bitsize, num_frac=bitsize)
    cvals = cbloq.on_classical_vals(a=a_fxp, b=b_fxp)
    assert cvals['result'] == Fxp(0.148834228515625, dtype=f'fxp-u{bitsize}/{bitsize}')
    bb = BloqBuilder()
    q0 = bb.add_register_from_dtype('a', QFxp(bitsize, bitsize))
    q1 = bb.add_register_from_dtype('b', QFxp(bitsize, bitsize))
    q0, q1, q2 = bb.add(MultiplyTwoReals(bitsize), a=q0, b=q1)
    q0, q1 = bb.add(MultiplyTwoReals(bitsize).adjoint(), a=q0, b=q1, result=q2)
    cbloq = bb.finalize(a=q0, b=q1)
    assert cbloq.on_classical_vals(a=a_fxp, b=b_fxp) == {'a': a_fxp, 'b': b_fxp}


def test_square_real_number():
    bb = BloqBuilder()
    bitsize = 15
    q0 = bb.add_register('a', bitsize)
    q0, q1 = bb.add(SquareRealNumber(bitsize), a=q0)
    cbloq = bb.finalize(a=q0, result=q1)
    a_fxp = val_to_fxp(0.554, num_bits=bitsize, num_frac=bitsize)
    cvals = cbloq.on_classical_vals(a=a_fxp)
    assert cvals['result'] == Fxp(0.3065185546875, dtype=f'fxp-u{bitsize}/{bitsize}')
    bb = BloqBuilder()
    q0 = bb.add_register_from_dtype('a', QFxp(bitsize, bitsize))
    q0, q1 = bb.add(SquareRealNumber(bitsize), a=q0)
    q0 = bb.add(SquareRealNumber(bitsize).adjoint(), a=q0, result=q1)
    cbloq = bb.finalize(a=q0)
    assert cbloq.on_classical_vals(a=a_fxp) == {'a': a_fxp}


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


@pytest.mark.notebook
def test_multiplication_notebook():
    execute_notebook('multiplication')
