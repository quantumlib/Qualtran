import pytest

from qualtran import BloqInstance, DanglingT, LeftDangle, Register, RightDangle, Side, Soquet
from qualtran.components.bloq_test import TestCNOT


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
    soq = Soquet(BloqInstance(TestCNOT(), i=0), Register('x', 10))
    assert soq.reg.side is Side.THRU
    assert soq.idx == ()
    assert soq.pretty() == 'x'


def test_soquet_idxed():
    binst = BloqInstance(TestCNOT(), i=0)
    reg = Register('y', 10, shape=(10, 2))

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg)

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))

    soq = Soquet(binst, reg, idx=(5, 0))
    assert soq.pretty() == 'y[5, 0]'

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))


def test_bloq_instance():
    binst_a = BloqInstance(TestCNOT(), i=1)
    binst_b = BloqInstance(TestCNOT(), i=1)
    assert binst_a == binst_b
    assert str(binst_a) == 'TestCNOT()<1>'
