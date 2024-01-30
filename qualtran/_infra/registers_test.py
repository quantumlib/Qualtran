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

from qualtran import Register, Side, Signature
from qualtran._infra.data_types import BoundedQInt, QAny, QBit
from qualtran._infra.gate_with_registers import get_named_qubits


def test_register():
    r = Register[QAny]("my_reg", dtype=QAny(5))
    assert r.name == 'my_reg'
    assert r.dtype.bitsize == 5
    assert r.shape == tuple()
    assert r.side == Side.THRU
    assert r.total_bits() == 5

    assert r == r.adjoint()


def test_multidim_register():
    r = Register[QBit]("my_reg", dtype=QBit(), shape=(2, 3), side=Side.RIGHT)
    idxs = list(r.all_idxs())
    assert len(idxs) == 2 * 3

    assert not r.side & Side.LEFT
    assert r.side & Side.THRU
    assert r.total_bits() == 2 * 3

    assert r.adjoint() == Register("my_reg", dtype=QBit(), shape=(2, 3), side=Side.LEFT)


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    regs = [
        Register[BoundedQInt]('x', dtype=BoundedQInt(n, range(N))),
        Register[BoundedQInt]('y', dtype=BoundedQInt(m, range(M))),
    ]
    for x in regs[0].dtype.iteration_range:
        for y in regs[1].dtype.iteration_range:
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)

    assert np.prod(tuple(len(reg.dtype.iteration_range) for reg in regs)) == N * M


def test_selection_registers_consistent():
    with pytest.raises(ValueError, match="BoundedQInt iteration length "):
        _ = Register[BoundedQInt]('a', dtype=BoundedQInt(3, range(10)))

    # Doesn't raise without selection register
    # with pytest.raises(ValueError, match="should be flat"):
    #     _ = Register[BoundedQInt]('a', dtype=BoundedQInt(3, range(5)), shape=(3, 5))

    selection_reg = Signature(
        [
            Register[BoundedQInt]('n', dtype=BoundedQInt(3, range(5))),
            Register[BoundedQInt]('m', dtype=BoundedQInt(4, range(12))),
        ]
    )
    assert selection_reg[0] == Register[BoundedQInt]('n', dtype=BoundedQInt(3, range(5)))
    assert selection_reg[1] == Register[BoundedQInt]('m', dtype=BoundedQInt(4, range(12)))
    assert selection_reg[:1] == tuple([Register[BoundedQInt]('n', dtype=BoundedQInt(3, range(5)))])


def test_registers_getitem_raises():
    g = Signature.build(a=4, b=3, c=2)
    with pytest.raises(TypeError, match="indices must be integers or slices"):
        _ = g[2.5]

    selection_reg = Signature([Register[BoundedQInt]('n', dtype=BoundedQInt(3, range(5)))])
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = selection_reg[2.5]


def test_signature():
    r1 = Register[QAny]("r1", QAny(5))
    r2 = Register[QAny]("r2", QAny(2))
    r3 = Register[QAny]("r3", QBit())
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
    sig1 = Signature([Register[QAny]("r1", QAny(5)), Register[QAny]("r2", QAny(2))])
    sig2 = Signature.build(r1=5, r2=2)
    assert sig1 == sig2


def test_and_regs():
    signature = Signature(
        [Register('control', QAny(2)), Register('target', dtype=QBit(), side=Side.RIGHT)]
    )
    assert list(signature.lefts()) == [Register('control', QAny(2))]
    assert list(signature.rights()) == [
        Register('control', QAny(2)),
        Register('target', dtype=QBit(), side=Side.RIGHT),
    ]

    adj = signature.adjoint()
    assert list(adj.rights()) == [Register('control', QAny(2))]
    assert list(adj.lefts()) == [
        Register('control', QAny(2)),
        Register('target', dtype=QBit(), side=Side.LEFT),
    ]


def test_agg_split():
    n_targets = 3
    sig = Signature(
        [
            Register[QBit]('control', 1),
            Register[QAny]('target', dtype=QAny(n_targets), shape=tuple(), side=Side.LEFT),
            Register[QBit]('target', dtype=QBit(), shape=(n_targets,), side=Side.RIGHT),
        ]
    )
    assert len(list(sig.groups())) == 2
    assert sorted([k for k, v in sig.groups()]) == ['control', 'target']
    assert len(list(sig.lefts())) == 2
    assert len(list(sig.rights())) == 2


def test_get_named_qubits_multidim():
    regs = Signature([Register[QAny]('q', dtype=QAny(bitsize=4), shape=(2, 3))])
    quregs = get_named_qubits(regs.lefts())
    assert quregs['q'].shape == (2, 3, 4)
    assert quregs['q'][1, 2, 3] == cirq.NamedQubit('q[1, 2][3]')


def test_duplicate_names():
    regs = Signature(
        [
            Register[QBit]('control', QBit(), side=Side.LEFT),
            Register[QBit]('control', QBit(), side=Side.RIGHT),
        ]
    )
    assert len(list(regs.lefts())) == 1

    with pytest.raises(ValueError, match=r'.*control is specified more than once per side.'):
        Signature([Register[QBit]('control', QBit()), Register[QBit]('control', QBit())])
