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
import cirq
import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, QAny, QInt, QMontgomeryUInt, QUInt
from qualtran.bloqs.arithmetic.bitwise import (
    _bitwise_not,
    _bitwise_not_symb,
    _xor,
    _xor_symb,
    _xork,
    BitwiseNot,
    Xor,
    XorK,
)
from qualtran.bloqs.basic_gates import IntEffect, IntState


def test_examples(bloq_autotester):
    bloq_autotester(_xork)


def test_xork_classical_sim():
    dtype = QUInt(6)
    k = 0b010110
    bloq = XorK(dtype, k)
    dbloq = bloq.decompose_bloq()
    cbloq = dbloq.controlled()

    for x in dtype.get_classical_domain():
        (x_out,) = bloq.call_classically(x=x)
        assert x_out == x ^ k

        (x_out,) = dbloq.call_classically(x=x)
        assert x_out == x ^ k

        ctrl_out, x_out = cbloq.call_classically(ctrl=0, x=x)
        assert ctrl_out == 0
        assert x_out == x

        ctrl_out, x_out = cbloq.call_classically(ctrl=1, x=x)
        assert ctrl_out == 1
        assert x_out == x ^ k


def test_xork_diagram():
    bloq = XorK(QUInt(4), 0b0110)
    circuit = bloq.as_composite_bloq().to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        '''
x0: ───⊕6───
       │
x1: ───⊕6───
       │
x2: ───⊕6───
       │
x3: ───⊕6───
    ''',
    )


def test_xor(bloq_autotester):
    bloq_autotester(_xor)


def test_xor_symb(bloq_autotester):
    bloq_autotester(_xor_symb)


@pytest.mark.parametrize("dtype", [QAny(4), QUInt(4)])
def test_xor_call_classically(dtype):
    bloq = Xor(dtype)
    domain = (
        dtype.get_classical_domain() if not isinstance(dtype, QAny) else range(2**dtype.bitsize)
    )
    for x in domain:
        for y in domain:
            x_out, y_out = bloq.call_classically(x=x, y=y)
            assert x_out == x and y_out == x ^ y
            x_out, y_out = bloq.decompose_bloq().call_classically(x=x, y=y)
            assert x_out == x and y_out == x ^ y


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [QAny(4), QUInt(4)])
def test_xor_tensor(dtype):
    bloq = Xor(dtype)
    domain = (
        dtype.get_classical_domain() if not isinstance(dtype, QAny) else range(2**dtype.bitsize)
    )
    for x in domain:
        for y in domain:
            bb = BloqBuilder()
            x_soq = bb.add(IntState(x, 4))
            y_soq = bb.add(IntState(y, 4))
            x_soq, y_soq = bb.add_t(bloq, x=x_soq, y=y_soq)
            bb.add(IntEffect(x, 4), val=x_soq)
            cbloq = bb.finalize(y=y_soq)

            np.testing.assert_allclose(
                cbloq.tensor_contract(), IntState(x ^ y, 4).tensor_contract()
            )


def test_xor_diagram():
    bloq = Xor(QUInt(4))
    circuit = bloq.as_composite_bloq().to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        '''
x0: ───x─────
       │
x1: ───x─────
       │
x2: ───x─────
       │
x3: ───x─────
       │
y0: ───x⊕y───
       │
y1: ───x⊕y───
       │
y2: ───x⊕y───
       │
y3: ───x⊕y───
    ''',
    )


def test_bitwise_not_examples(bloq_autotester):
    bloq_autotester(_bitwise_not)
    bloq_autotester(_bitwise_not_symb)


@pytest.mark.parametrize("n", [4, 5])
def test_bitwise_not_tensor(n):
    bloq = BitwiseNot(QUInt(n))

    matrix = np.zeros((2**n, 2**n))
    for i in range(2**n):
        matrix[i, ~i] = 1

    np.testing.assert_allclose(bloq.tensor_contract(), matrix)


def test_bitwise_not_diagram():
    bloq = BitwiseNot(QUInt(4))
    circuit = bloq.as_composite_bloq().to_cirq_circuit()
    cirq.testing.assert_has_diagram(
        circuit,
        '''
x0: ───~x───
       │
x1: ───~x───
       │
x2: ───~x───
       │
x3: ───~x───
    ''',
    )


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt, QInt])
@pytest.mark.parametrize('bitsize', range(2, 6))
def test_bitwisenot_classical_action(dtype, bitsize):
    b = BitwiseNot(dtype(bitsize))
    if dtype is QInt:
        valid_range = range(-(2 ** (bitsize - 1)), 2 ** (bitsize - 1))
    else:
        valid_range = range(2**bitsize)
    qlt_testing.assert_consistent_classical_action(b, x=valid_range)
