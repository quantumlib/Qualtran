#  Copyright 2023 Google LLC
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

from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QFxp,
    QInt,
    QIntOnesComp,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import Text, TextBox
from qualtran.symbolics.types import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class ToContiguousIndex(Bloq):
    r"""Build a contiguous register s from mu and nu.

    $$
        s = \nu (\nu + 1) / 2 + \mu
    $$

    Assuming nu is zero indexed (in contrast to the THC paper which assumes 1,
    hence the slightly different formula).

    Args:
        bitsize: number of bits for mu and nu registers.
        s_bitsize: Number of bits for contiguous register.

    Registers:
        mu: input register
        nu: input register
        s: output contiguous register

    References:
        [Even more efficient quantum computations of chemistry through
        tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Eq. 29.
    """

    bitsize: int
    s_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QUInt(self.bitsize)),
                Register("nu", QUInt(bitsize=self.bitsize)),
                Register("s", QUInt(bitsize=self.s_bitsize)),
            ]
        )

    def on_classical_vals(
        self, mu: 'ClassicalValT', nu: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'mu': mu, 'nu': nu, 's': nu * (nu + 1) // 2 + mu}

    def _t_complexity_(self) -> 'TComplexity':
        num_toffoli = self.bitsize**2 + self.bitsize - 1
        return TComplexity(t=4 * num_toffoli)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        if reg is None:
            return Text('')
        if reg.name == 'mu':
            return TextBox('μ')
        elif reg.name == 'nu':
            return TextBox(r'ν')
        else:
            text = r'⊕ν(ν-1)/2+μ'
            return TextBox(text)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.bitsize**2 + self.bitsize - 1
        return {(Toffoli(), num_toffoli)}


@bloq_example
def _to_contg_index() -> ToContiguousIndex:
    to_contg_index = ToContiguousIndex(bitsize=4, s_bitsize=8)
    return to_contg_index


_TO_CONTG_INDX = BloqDocSpec(bloq_cls=ToContiguousIndex, examples=[_to_contg_index])


@frozen
class SignedIntegerToTwosComplement(Bloq):
    """Convert a register storing the signed integer representation to two's complement inplace.

    Args:
        bitsize: size of the register.

    Registers:
        x: input signed integer (ones' complement) register.
        y: output signed integer register in two's complement.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767).
        page 24, 4th paragraph from the bottom.
    """

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x', QIntOnesComp(self.bitsize), side=Side.LEFT),
                Register('y', QInt(self.bitsize), side=Side.RIGHT),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Take the sign qubit as a control and cnot the remaining qubits, then
        # add it to the remaining n-1 bits.
        return {(Toffoli(), (self.bitsize - 2))}


@bloq_example
def _signed_to_twos() -> SignedIntegerToTwosComplement:
    signed_to_twos = SignedIntegerToTwosComplement(bitsize=10)
    return signed_to_twos


_SIGNED_TO_TWOS = BloqDocSpec(bloq_cls=SignedIntegerToTwosComplement, examples=[_signed_to_twos])


@frozen
class ToFxp(Bloq):
    """Convert a register storing a `QUInt` to a `QFxp`, optionally dividing by a power of two.

    Args:
        a_dtype: Quantum datatype used to represent the input.
        b_dtype: Quantum datatype used to represent the output.
        divide_by: Power of two to divide by when converting (default: 0).

    Registers:
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized output register (register b above).
    """

    a_dtype: QUInt
    b_dtype: QFxp = field()
    divide_by: SymbolicInt = 0

    @b_dtype.validator
    def _b_dtype_validate(self, field, val):
        if (
            is_symbolic(val.bitsize)
            or is_symbolic(self.a_dtype.bitsize)
            or is_symbolic(self.divide_by)
        ):
            return
        if val.num_int < self.a_dtype.bitsize - self.divide_by or val.num_frac < self.divide_by:
            raise ValueError("b_dtype not big enough for a_dtype")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(a=self.a_dtype, b=self.b_dtype)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(CNOT(), self.a_dtype.bitsize)}

    def build_composite_bloq(self, bb: BloqBuilder, a: Soquet, b: Soquet) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.a_dtype.bitsize)
            or is_symbolic(self.b_dtype.num_frac)
            or is_symbolic(self.divide_by)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}.")

        a_bits = bb.split(a)
        b_bits = bb.split(b)
        offset = self.b_dtype.num_frac - self.divide_by
        for i in range(len(a_bits)):
            a_bits[i], b_bits[offset + i] = bb.add_t(
                CNOT(), ctrl=a_bits[i], target=b_bits[offset + i]
            )

        return {"a": bb.join(a_bits), "b": bb.join(b_bits)}


@bloq_example
def _to_fxp() -> ToFxp:
    from qualtran import QFxp, QUInt

    to_fxp = ToFxp(QUInt(4), QFxp(4, 2, False), 2)
    return to_fxp


_TO_FXP = BloqDocSpec(bloq_cls=ToFxp, examples=[_to_fxp])
