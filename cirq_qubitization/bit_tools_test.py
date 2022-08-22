import pytest

from cirq_qubitization.bit_tools import iter_bits


def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]
