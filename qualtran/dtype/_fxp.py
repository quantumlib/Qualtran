#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Iterable, List, Sequence, TYPE_CHECKING, Union

import attrs

from qualtran.symbolics import is_symbolic, SymbolicInt

from ._base import BitEncoding, CDType, QDType

if TYPE_CHECKING:
    import fxpmath


@attrs.frozen
class _Fxp(BitEncoding[int]):
    r"""Fixed point type to represent real numbers.

    To hook into the classical simulator, we use fixed-width integers to represent
    values of this type. See `to_fixed_width_int` for details.
    In particular, the user should call `QFxp.to_fixed_width_int(float_value)`
    before passing a value to `bloq.call_classically`.

    See https://github.com/quantumlib/Qualtran/issues/1219 for discussion on alternatives
    and future upgrades.


    Args:
        bitsize: The total number of qubits used to represent the integer and
            fractional part combined.
        num_frac: The number of qubits used to represent the fractional part of the real number.
        signed: Whether the number is signed or not.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    signed: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.bitsize) and self.bitsize == 1 and self.signed:
            raise ValueError("bitsize must be > 1.")
        if not is_symbolic(self.bitsize) and not is_symbolic(self.num_frac):
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the Fxp is signed.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    @property
    def num_int(self) -> SymbolicInt:
        """Number of bits for the integral part."""
        return self.bitsize - self.num_frac

    @property
    def _int_encoding(self) -> BitEncoding[int]:
        # The corresponding dtype for the raw integer encoding.
        from qualtran.dtype._int import _Int
        from qualtran.dtype._uint import _UInt

        return _Int(self.bitsize) if self.signed else _UInt(self.bitsize)

    def get_domain(self) -> Iterable[int]:
        # Use the classical domain for the underlying raw integer encoding.
        yield from self._int_encoding.get_domain()

    def to_bits(self, x: int) -> List[int]:
        # Use the underlying raw integer encoding.
        return self._int_encoding.to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        # Use the underlying raw integer encoding.
        return self._int_encoding.from_bits(bits)

    def assert_valid_val(self, val: int, debug_str: str = 'val'):
        # Verify using the underlying raw integer encoding.
        self._int_encoding.assert_valid_val(val, debug_str)

    def to_fixed_width_int(
        self,
        x: Union[float, 'fxpmath.Fxp'],
        *,
        require_exact: bool = False,
        complement: bool = True,
    ) -> int:
        """Returns the interpretation of the binary representation of `x` as an integer.

        The returned value is an integer equal to `round(x * 2**self.num_frac)`.
        That is, the input value `x` is converted to a fixed-point binary value
        of `self.num_int` integral bits and `self.num_frac` fractional bits,
        and then re-interpreted as an integer by dropping the decimal point.

        Args:
            x: input real number
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.
        """
        bits = self._fxp_to_bits(x, require_exact=require_exact, complement=complement)
        return self._int_encoding.from_bits(bits)

    def float_from_fixed_width_int(self, x: int) -> float:
        """Helper to convert from the fixed-width-int representation to a true floating point value.

        Here `x` is the internal value used by the classical simulator.
        See `to_fixed_width_int` for conventions.
        """
        return x / 2**self.num_frac

    def __str__(self):
        if self.signed:
            return f'_Fxp({self.bitsize}, {self.num_frac}, True)'
        else:
            return f'_Fxp({self.bitsize}, {self.num_frac})'

    def fxp_dtype_template(self) -> 'fxpmath.Fxp':
        """A template of the `fxpmath.Fxp` data type for classical values.

        To construct an `fxpmath.Fxp` with this config, one can use:
        `Fxp(float_value, like=_Fxp(...).fxp_dtype_template)`,
        or given an existing value `some_fxp_value: Fxp`:
        `some_fxp_value.like(_Fxp(...).fxp_dtype_template)`.

        The following Fxp configuration is used:
         - op_sizing='same' and const_op_sizing='same' ensure that the returned
           object is not resized to a bigger fixed point number when doing
           operations with other Fxp objects.
         - shifting='trunc' ensures that when shifting the Fxp integer to
           left / right; the digits are truncated and no rounding occurs
         - overflow='wrap' ensures that when performing operations where result
           overflows, the overflowed digits are simply discarded.

        Support for `fxpmath.Fxp` is experimental, and does not hook into the classical
        simulator protocol. Once the library choice for fixed-point classical real
        values is finalized, the code will be updated to use the new functionality
        instead of delegating to raw integer values (see above).
        """
        import fxpmath

        if is_symbolic(self.bitsize) or is_symbolic(self.num_frac):
            raise ValueError(
                f"Cannot construct Fxp template for symbolic bitsizes: {self.bitsize=}, {self.num_frac=}"
            )

        return fxpmath.Fxp(
            None,
            n_word=self.bitsize,
            n_frac=self.num_frac,
            signed=self.signed,
            op_sizing='same',
            const_op_sizing='same',
            shifting='trunc',
            overflow='wrap',
        )

    def _get_domain_fxp(self) -> Iterable['fxpmath.Fxp']:
        import fxpmath

        for x in self._int_encoding.get_domain():
            yield fxpmath.Fxp(x / 2**self.num_frac, like=self.fxp_dtype_template())

    def _fxp_to_bits(
        self, x: Union[float, 'fxpmath.Fxp'], require_exact: bool = True, complement: bool = True
    ) -> List[int]:
        """Yields individual bits corresponding to binary representation of `x`.

        Args:
            x: The value to encode.
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.

        Raises:
            ValueError: If `x` is negative but this `_Fxp` is not signed.
        """
        import fxpmath

        if require_exact:
            self._assert_valid_val(x)
        if x < 0 and not self.signed:
            raise ValueError(f"unsigned _Fxp cannot represent {x}.")
        if self.signed and not complement:
            sign = int(x < 0)
            x = abs(x)
        fxp = x if isinstance(x, fxpmath.Fxp) else fxpmath.Fxp(x)
        bits = [int(x) for x in fxp.like(self.fxp_dtype_template()).bin()]
        if self.signed and not complement:
            bits[0] = sign
        return bits

    def _from_bits_to_fxp(self, bits: Sequence[int]) -> 'fxpmath.Fxp':
        import fxpmath

        if is_symbolic(self.num_frac):
            raise ValueError(f"Symbolic {self.num_frac} cannot be represented using Fxp")
        bits_bin = "".join(str(x) for x in bits[:])
        fxp_bin = "0b" + bits_bin[: -int(self.num_frac)] + "." + bits_bin[-int(self.num_frac) :]
        return fxpmath.Fxp(fxp_bin, like=self.fxp_dtype_template())

    def _assert_valid_val(self, val: Union[float, 'fxpmath.Fxp'], debug_str: str = 'val'):
        import fxpmath

        fxp_val = val if isinstance(val, fxpmath.Fxp) else fxpmath.Fxp(val)
        if fxp_val.get_val() != fxp_val.like(self.fxp_dtype_template()).get_val():
            raise ValueError(
                f"{debug_str}={val} cannot be accurately represented using Fxp {fxp_val}"
            )


@attrs.frozen
class QFxp(QDType[int]):
    r"""Fixed point quantum type to represent real numbers.

    A real number can be approximately represented in fixed point using `num_int`
    bits for the integer part and `num_frac` bits for the fractional part. If the
    real number is signed we store negative values in two's complement form. The first
    bit can therefore be treated as the sign bit in such cases (0 for +, 1 for -).
    In total there are `bitsize = (num_int + num_frac)` bits used to represent the number.
    E.g. Using `(bitsize = 8, num_frac = 6, signed = False)` then
    $\pi \approx 3.140625 = 11.001001$, where the . represents the decimal place.

    We can specify a fixed point real number by the tuple bitsize, num_frac and
    signed, with num_int determined as `(bitsize - num_frac)`.

    **Classical Simulation:**

    To hook into the classical simulator, we use fixed-width integers to represent
    values of this type. See `to_fixed_width_int` for details.
    In particular, the user should call `QFxp.to_fixed_width_int(float_value)`
    before passing a value to `bloq.call_classically`.

    The corresponding raw qdtype is either an QUInt (when `signed=False`) or
    QInt (when `signed=True`) of the same bitsize. This is the data type used
    to represent classical values during simulation, and convert to and from bits
    for intermediate values.

    For example, `QFxp(6, 4)` has 2 int bits and 4 frac bits, and the corresponding
    int type is `QUInt(6)`. So a true classical value of `10.0011` will have a raw
    integer representation of `100011`.

    Args:
        bitsize: The total number of qubits used to represent the integer and
            fractional part combined.
        num_frac: The number of qubits used to represent the fractional part of the real number.
        signed: Whether the number is signed or not.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    signed: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.bitsize) and self.bitsize == 1 and self.signed:
            raise ValueError("num_qubits must be > 1.")
        if not is_symbolic(self.bitsize) and not is_symbolic(self.num_frac):
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the QFxp is signed.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    @property
    def _bit_encoding(self) -> _Fxp:
        return _Fxp(bitsize=self.bitsize, num_frac=self.num_frac, signed=self.signed)

    @property
    def num_int(self) -> SymbolicInt:
        """Number of bits for the integral part."""
        return self._bit_encoding.num_int

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.num_frac)

    def to_fixed_width_int(
        self,
        x: Union[float, 'fxpmath.Fxp'],
        *,
        require_exact: bool = False,
        complement: bool = True,
    ) -> int:
        """Returns the interpretation of the binary representation of `x` as an integer.

        See class docstring section on "Classical Simulation" for more details on
        the choice of this representation.

        The returned value is an integer equal to `round(x * 2**self.num_frac)`.
        That is, the input value `x` is converted to a fixed-point binary value
        of `self.num_int` integral bits and `self.num_frac` fractional bits,
        and then re-interpreted as an integer by dropping the decimal point.

        For example, consider `QFxp(6, 4).to_fixed_width_int(1.5)`. As `1.5` is `0b01.1000`
        in this representation, the returned value would be `0b011000` = 24.

        For negative values, we use twos complement form. So in
        `QFxp(6, 4, signed=True).to_fixed_width_int(-1.5)`, the input is `0b10.1000`,
        which is interpreted as `0b101000` = -24.

        Args:
            x: input floating point value
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.
        """
        return self._bit_encoding.to_fixed_width_int(
            x=x, require_exact=require_exact, complement=complement
        )

    def float_from_fixed_width_int(self, x: int) -> float:
        """Helper to convert from the fixed-width-int representation to a true floating point value.

        Here `x` is the internal value used by the classical simulator.
        See `to_fixed_width_int` for conventions.

        See class docstring section on "Classical Simulation" for more details on
        the choice of this representation.
        """
        return self._bit_encoding.float_from_fixed_width_int(x=x)

    def __str__(self):
        if self.signed:
            return f'QFxp({self.bitsize}, {self.num_frac}, True)'
        else:
            return f'QFxp({self.bitsize}, {self.num_frac})'

    def fxp_dtype_template(self) -> 'fxpmath.Fxp':
        """A template of the `Fxp` data type for classical values.

        To construct an `Fxp` with this config, one can use:
        `Fxp(float_value, like=QFxp(...).fxp_dtype_template)`,
        or given an existing value `some_fxp_value: Fxp`:
        `some_fxp_value.like(QFxp(...).fxp_dtype_template)`.

        The following Fxp configuration is used:
         - op_sizing='same' and const_op_sizing='same' ensure that the returned
           object is not resized to a bigger fixed point number when doing
           operations with other Fxp objects.
         - shifting='trunc' ensures that when shifting the Fxp integer to
           left / right; the digits are truncated and no rounding occurs
         - overflow='wrap' ensures that when performing operations where result
           overflows, the overflowed digits are simply discarded.

        Support for `fxpmath.Fxp` is experimental, and does not hook into the classical
        simulator protocol. Once the library choice for fixed-point classical real
        values is finalized, the code will be updated to use the new functionality
        instead of delegating to raw integer values (see above).
        """
        return self._bit_encoding.fxp_dtype_template()


@attrs.frozen
class CFxp(CDType[int]):
    r"""Fixed point classical type to represent real numbers.

    This follows the same conventions as `QFxp`. See that class documentation for details.

    Args:
        bitsize: The total number of qubits used to represent the integer and
            fractional part combined.
        num_frac: The number of qubits used to represent the fractional part of the real number.
        signed: Whether the number is signed or not.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    signed: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.bitsize) and self.bitsize == 1 and self.signed:
            raise ValueError("num_qubits must be > 1.")
        if not is_symbolic(self.bitsize) and not is_symbolic(self.num_frac):
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the QFxp is signed.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    @property
    def _bit_encoding(self) -> _Fxp:
        return _Fxp(bitsize=self.bitsize, num_frac=self.num_frac, signed=self.signed)

    @property
    def num_int(self) -> SymbolicInt:
        """Number of bits for the integral part."""
        return self._bit_encoding.num_int

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.num_frac)

    def __str__(self):
        if self.signed:
            return f'CFxp({self.bitsize}, {self.num_frac}, True)'
        else:
            return f'CFxp({self.bitsize}, {self.num_frac})'
