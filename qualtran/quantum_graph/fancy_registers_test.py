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


def test_registers():
    r1 = Register("r1", 5)
    r2 = Register("r2", 2)
    r3 = Register("r3", 1)
    regs = Signature([r1, r2, r3])
    assert len(regs) == 3

    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3

    assert regs[0:1] == Signature([r1])
    assert regs[0:2] == Signature([r1, r2])
    assert regs[1:3] == Signature([r2, r3])

    assert regs["r1"] == r1
    assert regs["r2"] == r2
    assert regs["r3"] == r3

    assert list(regs) == [r1, r2, r3]

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }
    assert regs.get_cirq_quregs() == expected_named_qubits
    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [q for v in Signature(reg_order).get_cirq_quregs().values() for q in v]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


def test_registers_getitem_raises():
    g = Signature.build(a=4, b=3, c=2)
    with pytest.raises(IndexError, match="must be of the type"):
        _ = g[2.5]


def test_registers_build():
    regs1 = Signature([Register("r1", 5), Register("r2", 2)])
    regs2 = Signature.build(r1=5, r2=2)
    assert regs1 == regs2


def test_and_regs():
    regs = Signature([Register('control', 2), Register('target', 1, side=Side.RIGHT)])
    assert list(regs.lefts()) == [Register('control', 2)]
    assert list(regs.rights()) == [Register('control', 2), Register('target', 1, side=Side.RIGHT)]

    assert regs['control'] == Register('control', 2)
    with pytest.raises(KeyError):
        _ = regs['target']


def test_agg_split():
    n_targets = 3
    regs = Signature(
        [
            Register('control', 1),
            Register('target', bitsize=n_targets, shape=tuple(), side=Side.LEFT),
            Register('target', bitsize=1, shape=(n_targets,), side=Side.RIGHT),
        ]
    )
    assert len(list(regs.groups())) == 2
    assert sorted([k for k, v in regs.groups()]) == ['control', 'target']
    assert len(list(regs.lefts())) == 2
    assert len(list(regs.rights())) == 2


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
