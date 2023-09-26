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

from qualtran import BloqBuilder, Register
from qualtran.bloqs.arithmetic import Add, GreaterThan, Product, Square, SumOfSquares
from qualtran.testing import execute_notebook


def _make_add():
    from qualtran.bloqs.arithmetic import Add

    return Add(bitsize=4)


def _make_square():
    from qualtran.bloqs.arithmetic import Square

    return Square(bitsize=8)


def _make_sum_of_squares():
    from qualtran.bloqs.arithmetic import SumOfSquares

    return SumOfSquares(bitsize=8, k=4)


def _make_product():
    from qualtran.bloqs.arithmetic import Product

    return Product(a_bitsize=4, b_bitsize=6)


def _make_greater_than():
    from qualtran.bloqs.arithmetic import GreaterThan

    return GreaterThan(bitsize=4)


def test_add():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    a, b = bb.add(Add(bitsize), a=q0, b=q1)
    cbloq = bb.finalize(a=a, b=b)


def test_square():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('result', 2 * bitsize)
    q0, q1 = bb.add(Square(bitsize), a=q0, result=q1)
    cbloq = bb.finalize(a=q0, result=q1)


def test_sum_of_squares():
    bb = BloqBuilder()
    bitsize = 4
    k = 3
    inp = bb.add_register(Register("input", bitsize=bitsize, shape=(k,)))
    out = bb.add_register(Register("result", bitsize=2 * bitsize + 1))
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp, result=out)
    cbloq = bb.finalize(input=inp, result=out)


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q2 = bb.add_register('result', 2 * max(bitsize, mbits))
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1, result=q2)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)


def test_greater_than():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    anc = bb.add_register('result', 1)
    q0, q1, anc = bb.add(GreaterThan(bitsize), a=q0, b=q1, result=anc)
    cbloq = bb.finalize(a=q0, b=q1, result=anc)


def test_notebook():
    execute_notebook('arithmetic')
