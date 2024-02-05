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

from qualtran import QAny, QBit, QInt, Register, SelectionRegister, Side, Signature
from qualtran._infra.gate_with_registers import get_named_qubits


def test_register():
    r = Register("my_reg", 5)
    assert r.name == 'my_reg'
    assert r.bitsize == 5
    assert r.shape == tuple()
    assert r.side == Side.THRU
    assert r.total_bits() == 5

    assert r == r.adjoint()


def test_multidim_register():
    r = Register("my_reg", bitsize=1, shape=(2, 3), side=Side.RIGHT)
    idxs = list(r.all_idxs())
    assert len(idxs) == 2 * 3

    assert not r.side & Side.LEFT
    assert r.side & Side.THRU
    assert r.total_bits() == 2 * 3

    assert r.adjoint() == Register("my_reg", bitsize=1, shape=(2, 3), side=Side.LEFT)


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    regs = [SelectionRegister('x', n, N), SelectionRegister('y', m, M)]
    for x in range(regs[0].iteration_length):
        for y in range(regs[1].iteration_length):
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)

    assert np.prod(tuple(reg.iteration_length for reg in regs)) == N * M


def test_selection_registers_consistent():
    with pytest.raises(ValueError, match="iteration length must be in "):
        _ = SelectionRegister('a', 3, 10)

    with pytest.raises(ValueError, match="should be flat"):
        _ = SelectionRegister('a', bitsize=1, shape=(3, 5), iteration_length=5)

    selection_reg = Signature(
        [
            SelectionRegister('n', bitsize=3, iteration_length=5),
            SelectionRegister('m', bitsize=4, iteration_length=12),
        ]
    )
    assert selection_reg[0] == SelectionRegister('n', 3, 5)
    assert selection_reg[1] == SelectionRegister('m', 4, 12)
    assert selection_reg[:1] == tuple([SelectionRegister('n', 3, 5)])


def test_registers_getitem_raises():
    g = Signature.build(a=4, b=3, c=2)
    with pytest.raises(TypeError, match="indices must be integers or slices"):
        _ = g[2.5]

    selection_reg = Signature([SelectionRegister('n', bitsize=3, iteration_length=5)])
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = selection_reg[2.5]


def test_signature():
    r1 = Register("r1", 5)
    r2 = Register("r2", 2)
    r3 = Register("r3", 1)
    signature = Signature([r1, r2, r3])
    assert len(signature) == 3

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


def test_signature_build():
    sig1 = Signature([Register("r1", 5), Register("r2", 2)])
    sig2 = Signature.build(r1=5, r2=2)
    assert sig1 == sig2


def test_and_regs():
    signature = Signature([Register('control', 2), Register('target', 1, side=Side.RIGHT)])
    assert list(signature.lefts()) == [Register('control', 2)]
    assert list(signature.rights()) == [
        Register('control', 2),
        Register('target', 1, side=Side.RIGHT),
    ]

    adj = signature.adjoint()
    assert list(adj.rights()) == [Register('control', 2)]
    assert list(adj.lefts()) == [Register('control', 2), Register('target', 1, side=Side.LEFT)]


def test_agg_split():
    n_targets = 3
    sig = Signature(
        [
            Register('control', 1),
            Register('target', bitsize=n_targets, shape=tuple(), side=Side.LEFT),
            Register('target', bitsize=1, shape=(n_targets,), side=Side.RIGHT),
        ]
    )
    assert len(list(sig.groups())) == 2
    assert sorted([k for k, v in sig.groups()]) == ['control', 'target']
    assert len(list(sig.lefts())) == 2
    assert len(list(sig.rights())) == 2


def test_get_named_qubits_multidim():
    regs = Signature([Register('q', shape=(2, 3), bitsize=4)])
    quregs = get_named_qubits(regs.lefts())
    assert quregs['q'].shape == (2, 3, 4)
    assert quregs['q'][1, 2, 3] == cirq.NamedQubit('q[1, 2][3]')


def test_duplicate_names():
    regs = Signature(
        [Register('control', 1, side=Side.LEFT), Register('control', 1, side=Side.RIGHT)]
    )
    assert len(list(regs.lefts())) == 1

    with pytest.raises(ValueError, match=r'.*control is specified more than once per side.'):
        Signature([Register('control', 1), Register('control', 1)])


def test_dtypes_converter():
    r1 = Register("my_reg", 5)
    r2 = Register("my_reg", QAny(5))
    assert r1 == r2
    r1 = Register("my_reg", 1)
    r2 = Register("my_reg", QBit())
    assert r1 == r2
    r2 = Register("my_reg", 5)
    r2 = Register("my_reg", QInt(5))
    assert r1 != r2
