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
import itertools
from typing import TYPE_CHECKING

import pytest

import qualtran as qlt
import qualtran.dtype as qdt
import qualtran.testing as qlt_testing

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar, QVarT


@qlt.bloqify
def maj(bb: 'BloqBuilder', ck, ik, tk):
    # Figure 2, left part: uses one AND
    ck, ik = bb.CNOT(ck, ik)
    ck, tk = bb.CNOT(ck, tk)
    [ik, tk], ckp1 = bb.And([ik, tk])
    ck, ckp1 = bb.CNOT(ck, ckp1)
    return {'ck': ck, 'ik': ik, 'tk': tk, 'ckp1': ckp1}


def test_maj():
    bloq = maj.make(qlt.qsig(ck=1, ik=1, tk=1))
    qlt_testing.assert_valid_cbloq(bloq)


def test_maj_logic():
    bloq = maj.make(qlt.qsig(ck=1, ik=1, tk=1))

    for c, i, t in itertools.product([0, 1], repeat=3):
        c_out, i_out, t_out, ckp1 = bloq.call_classically(ck=c, ik=i, tk=t)
        expected = c + i + t
        assert ckp1 == int(expected >= 2)


def test_maj_truth_table():
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table,
        get_classical_truth_table,
    )

    bloq = maj.make(qlt.qsig(ck=1, ik=1, tk=1))
    tt = format_classical_truth_table(*get_classical_truth_table(bloq))
    assert tt == """\
ck  ik  tk  |  ck  ik  tk  ckp1
--------------------------------
0, 0, 0 -> 0, 0, 0, 0
0, 0, 1 -> 0, 0, 1, 0
0, 1, 0 -> 0, 1, 0, 0
0, 1, 1 -> 0, 1, 1, 1
1, 0, 0 -> 1, 1, 1, 0
1, 0, 1 -> 1, 1, 0, 1
1, 1, 0 -> 1, 0, 1, 1
1, 1, 1 -> 1, 0, 0, 1"""


@qlt.bloqify
def un_maj(bb: 'BloqBuilder', ck, ik, tk, ckp1):
    """Uncompute the carry bits.

    This is most of the right part of the building block in Figure 2. We ommit the final
    CNOT(ik, tk), leaving that responsibility to the outer circuit. This is in the spirit of
    Figure 1, which shows the CNOT(ik, tk) gates distinct from the recursive structure.
    By factoring it in this way, `maj` and `un_maj` only concern themselves with
    propogating the carry bits.
    """
    ck, ckp1 = bb.CNOT(ck, ckp1)
    [ik, tk] = bb.UnAnd([ik, tk], ckp1)
    ck, ik = bb.CNOT(ck, ik)
    return {'ck': ck, 'ik': ik, 'tk': tk}


@qlt.bloqify
def add_bits(bb: 'BloqBuilder', i: 'QVarT', t: 'QVarT'):
    assert i.shape == t.shape
    assert len(i.shape) == 1, '1d array'
    n = i.shape[0]

    # First bit: no input carry
    [i[0], t[0]], c0 = bb.And([i[0], t[0]])  # type: ignore
    c = [c0]

    # Ripple-carry
    for k in range(1, n - 1):
        c[k - 1], i[k], t[k], cnext = maj(bb, c[k - 1], i[k], t[k])
        c.append(cnext)

    # Last bit: no output carry
    k = n - 1
    c[k - 1], t[k] = bb.CNOT(c[k - 1], t[k])  # type: ignore

    # Un-ripple-carry
    for k in range(n - 1 - 1, 1 - 1, -1):
        c[k - 1], i[k], t[k] = un_maj(bb, ck=c[k - 1], ik=i[k], tk=t[k], ckp1=c[k])

    # First bit
    i[0], t[0] = bb.UnAnd([i[0], t[0]], c[0])

    for k in range(n):
        i[k], t[k] = bb.CNOT(i[k], t[k])  # type: ignore

    return {'i': i, 't': t}


@qlt.bloqify
def add(bb: 'BloqBuilder', a: 'QVar', b: 'QVar'):
    assert a.dtype.num_bits == b.dtype.num_bits
    assert a.dtype.num_bits > 1

    i = a[:]  # split
    t = b[:]  # split

    # Account for endianness!
    i_out, t_out = add_bits.inline(bb, i=i[::-1], t=t[::-1])
    return {'a': bb.join(i_out[::-1], dtype=a.dtype), 'b': bb.join(t_out[::-1], dtype=b.dtype)}  # type: ignore


def test_add():
    bloq = add.make(qlt.qsig(a=qdt.QUInt(5), b=qdt.QUInt(5)))
    qlt_testing.assert_valid_cbloq(bloq)


def test_add_logic():
    n = 5
    bloq = add.make(qlt.qsig(a=qdt.QUInt(n), b=qdt.QUInt(n)))

    for a in range(2**n):
        for b in range(2**n):
            a_out, apb = bloq.call_classically(a=a, b=b)
            assert apb == (a + b) % (2**n)


def test_add_edge_case():
    n = 1
    with pytest.raises(AssertionError):
        _ = add.make(qlt.qsig(a=qdt.QUInt(n), b=qdt.QUInt(n)))


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('bloqify-adder')
