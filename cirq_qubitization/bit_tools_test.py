import math
import random

import pytest

from cirq_qubitization.bit_tools import (
    float_as_fixed_width_int,
    iter_bits,
    iter_bits_fixed_point,
    iter_bits_twos_complement,
)


def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(0, 3, signed=True)) == [0, 0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(1, 2, signed=True)) == [0, 1]
    assert list(iter_bits(-1, 2, signed=True)) == [1, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(2, 3, signed=True)) == [0, 1, 0]
    assert list(iter_bits(-2, 3, signed=True)) == [1, 1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]


def test_iter_bits_twos():
    assert list(iter_bits_twos_complement(0, 4)) == [0, 0, 0, 0]
    assert list(iter_bits_twos_complement(1, 4)) == [0, 0, 0, 1]
    assert list(iter_bits_twos_complement(-2, 4)) == [1, 1, 1, 0]
    assert list(iter_bits_twos_complement(-3, 4)) == [1, 1, 0, 1]


random.seed(1234)


@pytest.mark.parametrize('val', [random.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_iter_bits_fixed_point(val, width, signed):
    if (val < 0) and not signed:
        with pytest.raises(AssertionError):
            _ = [*iter_bits_fixed_point(val, width, signed=signed)]
    else:
        bits = [*iter_bits_fixed_point(val, width, signed=signed)]
        if signed:
            sign, bits = bits[0], bits[1:]
            assert sign == (1 if val < 0 else 0)
        val = abs(val)
        approx_val = math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
        unsigned_width = width - 1 if signed else width
        assert math.isclose(
            val, approx_val, abs_tol=1 / 2**unsigned_width
        ), f'{val}:{approx_val}:{width}'
        bits_from_int = [
            *iter_bits(float_as_fixed_width_int(val, unsigned_width + 1)[1], unsigned_width)
        ]
        assert bits == bits_from_int
