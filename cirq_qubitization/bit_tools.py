from typing import Iterator

import numpy as np


def iter_bits(val: int, width: int) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.
    """
    if val.bit_length() > width:
        raise ValueError(f"{val} exceeds width {width}.")
    if val < 0:
        raise ValueError(f"{val} is negative.")
    for b in f'{val:0{width}b}':
        yield int(b)


def iter_bits_twos_complement(val: int, width: int) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first. Allows for negative values and represents these using twos
    complement.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.
    """
    if (val.bit_length() - 1) // 2 > width:
        raise ValueError(f"{val} exceeds width {width}.")
    mask = (1 << width) - 1
    for b in f'{val&mask:0{width}b}':
        yield int(b)


def iter_bits_fixed_point(val: float, width: int) -> Iterator[int]:
    """Represent the floating point number 0 <= val <= 1 using `width` bits.

    $$ val = \sum_{b=0}^{width - 1} val[b] / 2^{1+b} $$

    Args:
        val: Floating point number in [0, 1]
        width: The number of output bits in fixed point binary representation of `val`.
    """
    assert 0 <= val <= 1, f"val must be between [0, 1]"
    for b in range(width):
        val = val * 2
        out_bit = np.floor(val)
        val = val - out_bit
        yield int(out_bit)


def float_as_fixed_width_int(val: float, width: int) -> int:
    """Returns a `width` length fixed point binary representation of `val` where 0 <= val <= 1."""
    return int(''.join(str(b) for b in iter_bits_fixed_point(val, width)), 2)
