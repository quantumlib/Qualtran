import pytest

from cirq_qubitization.quantum_graph.quantum_graph import LeftDangle, RightDangle, DanglingT


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
