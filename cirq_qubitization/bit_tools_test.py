import pytest

from cirq_qubitization.bit_tools import iter_bits, iter_bits_twos_complement


def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]


def test_iter_bits_twos():
    assert list(iter_bits_twos_complement(0, 4)) == [0, 0, 0, 0]
    assert list(iter_bits_twos_complement(1, 4)) == [0, 0, 0, 1]
    assert list(iter_bits_twos_complement(-2, 4)) == [1, 1, 1, 0]
    assert list(iter_bits_twos_complement(-3, 4)) == [1, 1, 0, 1]
