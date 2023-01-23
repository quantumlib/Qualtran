import cirq
import pytest

from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side


def test_register():
    r = FancyRegister("my_reg", 5)
    assert r.name == 'my_reg'
    assert r.bitsize == 5
    assert r.wireshape == tuple()
    assert r.side == Side.THRU


def test_multidim_register():
    r = FancyRegister("my_reg", bitsize=1, wireshape=(2, 3), side=Side.RIGHT)
    idxs = list(r.wire_idxs())
    assert len(idxs) == 2 * 3

    assert not r.side & Side.LEFT
    assert r.side & Side.THRU


def test_registers():
    r1 = FancyRegister("r1", 5)
    r2 = FancyRegister("r2", 2)
    r3 = FancyRegister("r3", 1)
    regs = FancyRegisters([r1, r2, r3])
    assert len(regs) == 3

    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3

    assert regs[0:1] == FancyRegisters([r1])
    assert regs[0:2] == FancyRegisters([r1, r2])
    assert regs[1:3] == FancyRegisters([r2, r3])

    assert regs["r1"] == r1
    assert regs["r2"] == r2
    assert regs["r3"] == r3

    assert list(regs) == [r1, r2, r3]

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }
    assert regs.get_named_qubits() == expected_named_qubits
    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [
            q for v in FancyRegisters(reg_order).get_named_qubits().values() for q in v
        ]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


def test_registers_getitem_raises():
    g = FancyRegisters.build(a=4, b=3, c=2)
    with pytest.raises(IndexError, match="must be of the type"):
        _ = g[2.5]


def test_registers_build():
    regs1 = FancyRegisters([FancyRegister("r1", 5), FancyRegister("r2", 2)])
    regs2 = FancyRegisters.build(r1=5, r2=2)
    assert regs1 == regs2


def test_and_regs():
    regs = FancyRegisters(
        [FancyRegister('control', 2), FancyRegister('target', 1, side=Side.RIGHT)]
    )
    assert list(regs.lefts()) == [FancyRegister('control', 2)]
    assert list(regs.rights()) == [
        FancyRegister('control', 2),
        FancyRegister('target', 1, side=Side.RIGHT),
    ]

    assert regs['control'] == FancyRegister('control', 2)
    with pytest.raises(KeyError):
        _ = regs['target']


def test_agg_split():
    n_targets = 3
    regs = FancyRegisters(
        [
            FancyRegister('control', 1),
            FancyRegister('target', bitsize=n_targets, wireshape=tuple(), side=Side.LEFT),
            FancyRegister('target', bitsize=1, wireshape=(n_targets,), side=Side.RIGHT),
        ]
    )
    assert len(list(regs.groups())) == 2
    assert sorted([k for k, v in regs.groups()]) == ['control', 'target']
    assert len(list(regs.lefts())) == 2
    assert len(list(regs.rights())) == 2


def test_get_named_qubits_multidim():
    regs = FancyRegisters([FancyRegister('matt', wireshape=(2, 3), bitsize=4)])
    quregs = regs.get_named_qubits()
    assert quregs['matt'].shape == (2, 3, 4)
    assert quregs['matt'][1, 2, 3] == cirq.NamedQubit('matt[1, 2, 3]')
