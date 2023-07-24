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
import pytest

from qualtran import Register, Side, Signature


def test_register():
    r = Register("my_reg", 5)
    assert r.name == 'my_reg'
    assert r.bitsize == 5
    assert r.shape == tuple()
    assert r.side == Side.THRU
    assert r.total_bits() == 5


def test_multidim_register():
    r = Register("my_reg", bitsize=1, shape=(2, 3), side=Side.RIGHT)
    idxs = list(r.all_idxs())
    assert len(idxs) == 2 * 3

    assert not r.side & Side.LEFT
    assert r.side & Side.THRU
    assert r.total_bits() == 2 * 3


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
    assert list(signature.get_cirq_quregs().keys()) == list(expected_named_qubits.keys())
    assert all(
        (a == b).all()
        for a, b in zip(signature.get_cirq_quregs().values(), expected_named_qubits.values())
    )
    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [q for v in Signature(reg_order).get_cirq_quregs().values() for q in v]
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
    regs = Signature([Register('matt', shape=(2, 3), bitsize=4)])
    quregs = regs.get_cirq_quregs()
    assert quregs['matt'].shape == (2, 3, 4)
    assert quregs['matt'][1, 2, 3] == cirq.NamedQubit('matt[1, 2, 3]')


def test_duplicate_names():
    regs = Signature(
        [Register('control', 1, side=Side.LEFT), Register('control', 1, side=Side.RIGHT)]
    )
    assert len(list(regs.lefts())) == 1

    with pytest.raises(ValueError, match=r'.*control is specified more than once per side.'):
        Signature([Register('control', 1), Register('control', 1)])
