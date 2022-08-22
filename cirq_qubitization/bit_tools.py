from typing import Iterator


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
    for b in f'{val:0{width}b}':
        yield int(b)
