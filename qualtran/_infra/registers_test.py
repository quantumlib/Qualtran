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
import sympy

from qualtran import BQUInt, QAny, QBit, QInt, Register, Side, Signature
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.symbolics import is_symbolic


def test_register():
    r = Register("my_reg", QAny(5))
    assert r.name == 'my_reg'
    assert r.bitsize == 5
    assert r.shape == tuple()
    assert r.side == Side.THRU
    assert r.total_bits() == 5

    assert r == r.adjoint()


def test_multidim_register():
    r = Register("my_reg", QBit(), shape=(2, 3), side=Side.RIGHT)
    idxs = list(r.all_idxs())
    assert len(idxs) == 2 * 3

    assert not r.side & Side.LEFT
    assert r.side & Side.THRU
    assert r.total_bits() == 2 * 3

    assert r.adjoint() == Register("my_reg", QBit(), shape=(2, 3), side=Side.LEFT)


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    dtypes = [BQUInt(n, N), BQUInt(m, M)]
    regs = [Register(sym, dtype) for sym, dtype in zip(['x', 'y'], dtypes)]
    for x in range(int(dtypes[0].iteration_length)):
        for y in range(int(dtypes[1].iteration_length)):
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)

    assert np.prod(tuple(int(dtype.iteration_length) for dtype in dtypes)) == N * M


def test_selection_registers_consistent():
    with pytest.raises(ValueError, match=".*iteration length is too large "):
        _ = Register('a', BQUInt(3, 10))

    selection_reg = Signature(
        [
            Register('n', BQUInt(bitsize=3, iteration_length=5)),
            Register('m', BQUInt(bitsize=4, iteration_length=12)),
        ]
    )
    assert selection_reg[0] == Register('n', BQUInt(3, 5))
    assert selection_reg[1] == Register('m', BQUInt(4, 12))
    assert selection_reg[:1] == tuple([Register('n', BQUInt(3, 5))])


def test_registers_getitem_raises():
    g = Signature.build(a=4, b=3, c=2)
    with pytest.raises(TypeError, match="indices must be integers or slices"):
        _ = g[2.5]  # type: ignore[call-overload]

    selection_reg = Signature([Register('n', BQUInt(bitsize=3, iteration_length=5))])
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = selection_reg[2.5]  # type: ignore[call-overload]


def test_signature():
    r1 = Register("r1", QAny(5))
    r2 = Register("r2", QAny(2))
    r3 = Register("r3", QBit())
    signature = Signature([r1, r2, r3])
    assert len(signature) == 3
    assert signature.n_qubits() == 8

    assert signature[0] == r1
    assert signature[1] == r2
    assert signature[2] == r3

    assert signature[0:1] == (r1,)
    assert signature[0:2] == (r1, r2)
    assert signature[1:3] == (r2, r3)

    assert list(signature) == [r1, r2, r3]

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }
    assert list(get_named_qubits(signature.lefts())) == list(expected_named_qubits.keys())
    assert all(
        (a == b).all()
        for a, b in zip(
            get_named_qubits(signature.lefts()).values(), expected_named_qubits.values()
        )
    )
    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [
            q for v in get_named_qubits(Signature(reg_order).lefts()).values() for q in v
        ]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


def test_signature_symbolic():
    n_x, n_y = sympy.symbols('n_x n_y')
    signature = Signature.build(x=n_x, y=n_y)
    assert signature.n_qubits() == n_x + n_y
    assert str(signature.n_qubits()) == 'n_x + n_y'


def test_signature_build():
    sig1 = Signature([Register("r1", QAny(5)), Register("r2", QAny(2))])
    sig2 = Signature.build(r1=5, r2=2)
    assert sig1 == sig2
    assert sig1.n_qubits() == 7
    sig1 = Signature([Register("r1", QInt(7)), Register("r2", QBit())])
    sig2 = Signature.build_from_dtypes(r1=QInt(7), r2=QBit())
    assert sig1 == sig2
    sig1 = Signature([Register("r1", QInt(7))])
    sig2 = Signature.build_from_dtypes(r1=QInt(7), r2=QAny(0))
    assert sig1 == sig2


def test_and_regs():
    signature = Signature(
        [Register('control', QAny(2)), Register('target', QBit(), side=Side.RIGHT)]
    )
    assert list(signature.lefts()) == [Register('control', QAny(2))]
    assert list(signature.rights()) == [
        Register('control', QAny(2)),
        Register('target', QBit(), side=Side.RIGHT),
    ]
    assert signature.n_qubits() == 3

    adj = signature.adjoint()
    assert list(adj.rights()) == [Register('control', QAny(2))]
    assert list(adj.lefts()) == [
        Register('control', QAny(2)),
        Register('target', QBit(), side=Side.LEFT),
    ]
    assert adj.n_qubits() == 3


def test_agg_split():
    n_targets = 3
    sig = Signature(
        [
            Register('control', QBit()),
            Register('target', QAny(n_targets), shape=tuple(), side=Side.LEFT),
            Register('target', QBit(), shape=(n_targets,), side=Side.RIGHT),
        ]
    )
    assert len(list(sig.groups())) == 2
    assert sorted([k for k, v in sig.groups()]) == ['control', 'target']
    assert len(list(sig.lefts())) == 2
    assert len(list(sig.rights())) == 2
    assert sig.n_qubits() == n_targets + 1


def test_get_named_qubits_multidim():
    regs = Signature([Register('q', shape=(2, 3), dtype=QAny(4))])
    quregs = get_named_qubits(regs.lefts())
    assert quregs['q'].shape == (2, 3, 4)
    assert quregs['q'][1, 2, 3] == cirq.NamedQubit('q[1, 2][3]')


def test_duplicate_names():
    regs = Signature(
        [Register('control', QBit(), side=Side.LEFT), Register('control', QBit(), side=Side.RIGHT)]
    )
    assert len(list(regs.lefts())) == 1

    with pytest.raises(ValueError, match=r'.*control is specified more than once per side.'):
        Signature([Register('control', QBit()), Register('control', QBit())])


def test_dtypes_converter():
    r1 = Register("my_reg", QAny(5))
    r2 = Register("my_reg", QAny(5))
    assert r1 == r2
    r1 = Register("my_reg", QBit())
    r2 = Register("my_reg", QBit())
    assert r1 == r2
    r1 = Register("my_reg", QAny(5))
    r2 = Register("my_reg", QInt(5))
    assert r1 != r2


def test_is_symbolic():
    r = Register("my_reg", QAny(sympy.Symbol("x")))
    assert is_symbolic(r)
    r = Register("my_reg", QAny(2), shape=sympy.symbols("x y"))
    assert is_symbolic(r)
