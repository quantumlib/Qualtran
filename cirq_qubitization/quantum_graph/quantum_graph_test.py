import pytest

from cirq_qubitization.quantum_graph.bloq_test import TestBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


def test_bloq_instance():
    tb1 = TestBloq()
    tb2 = TestBloq()
    assert tb1 == tb2
    assert BloqInstance(tb1, i=1) == BloqInstance(tb2, i=1)
    assert BloqInstance(tb1, i=1) != BloqInstance(tb2, i=2)


def test_dangling():
    assert LeftDangle is LeftDangle
    assert RightDangle is RightDangle
    assert LeftDangle is not RightDangle
    assert RightDangle is not LeftDangle

    assert isinstance(LeftDangle, DanglingT)
    assert isinstance(RightDangle, DanglingT)

    assert LeftDangle == LeftDangle
    assert RightDangle == RightDangle
    assert LeftDangle != RightDangle

    with pytest.raises(ValueError, match='Do not instantiate.*'):
        my_new_dangle = DanglingT('hi mom')


def test_dangling_hash():
    assert hash(LeftDangle) != hash(RightDangle)
    my_d = {LeftDangle: 'left', RightDangle: 'right'}
    assert my_d[LeftDangle] == 'left'
    assert my_d[RightDangle] == 'right'


def test_soquet():
    soq = Soquet(BloqInstance(TestBloq(), i=0), FancyRegister('x', 10))
    assert soq.reg.side is Side.THRU
    assert soq.idx == ()
    assert soq.pretty() == 'x'


def test_soquet_idxed():
    binst = BloqInstance(TestBloq(), i=0)
    reg = FancyRegister('y', 10, wireshape=(10, 2))

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg)

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))

    soq = Soquet(binst, reg, idx=(5, 0))
    assert soq.pretty() == 'y[5, 0]'

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))
