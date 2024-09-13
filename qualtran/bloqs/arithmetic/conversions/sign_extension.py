#  Copyright 2024 Google LLC
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
from typing import TYPE_CHECKING, Union

import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QInt, QIntOnesComp, Register, Side, Signature
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class SignExtend(Bloq):
    """Sign-Extend a value to a value of larger bitsize.

    Useful to implement arithmetic operations with differing operand bitsizes.
    A sign extension copies the MSB into the new bits of the wider value. For
    example: a 4-bit to 6-bit sign-extension of `1010` gives `111010`.

    See :class:`SignTruncate` for the adjoint operation.

    Args:
        inp_dtype: input data type.
        out_dtype: output data type. must be same class as `inp_dtype`,
                   and have larger bitsize.

    Registers:
        x (LEFT): the input register of type `inp_dtype`
        y (RIGHT): the output register of type `out_dtype`
    """

    inp_dtype: Union[QInt, QIntOnesComp]
    out_dtype: Union[QInt, QIntOnesComp]

    def __attrs_post_init__(self):
        if not isinstance(self.inp_dtype, type(self.out_dtype)):
            raise ValueError(
                f"Expected same input and output base types, got: {self.inp_dtype}, {self.out_dtype}"
            )

        if not is_symbolic(self.extend_bitsize) and self.extend_bitsize <= 0:
            raise ValueError(
                f"input bitsize {self.inp_dtype.num_qubits} must be smaller than "
                f"output bitsize {self.out_dtype.num_qubits}"
            )

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.inp_dtype, side=Side.LEFT),
                Register('y', self.out_dtype, side=Side.RIGHT),
            ]
        )

    @cached_property
    def extend_bitsize(self):
        return self.out_dtype.num_qubits - self.inp_dtype.num_qubits

    def adjoint(self) -> 'SignTruncate':
        return SignTruncate(self.out_dtype, self.inp_dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        extend_ys = bb.allocate(self.extend_bitsize)
        xs = bb.split(x)

        xs[0], extend_ys = bb.add(
            MultiTargetCNOT(self.extend_bitsize), control=xs[0], targets=extend_ys
        )

        extend_ys = bb.split(extend_ys)
        y = bb.join(np.concatenate([extend_ys, xs]), dtype=self.out_dtype)

        return {'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {MultiTargetCNOT(self.extend_bitsize): 1}

    def on_classical_vals(self, x: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        return {'y': x}


@bloq_example
def _sign_extend() -> SignExtend:
    from qualtran import QInt

    sign_extend = SignExtend(QInt(8), QInt(16))
    return sign_extend


_SIGN_EXTEND_DOC = BloqDocSpec(bloq_cls=SignExtend, examples=[_sign_extend])


@frozen
class SignTruncate(Bloq):
    """Truncate a signed value to a smaller bitsize.

    Useful to implement arithmetic operations with differing operand bitsizes.
    A signed truncation xors the MSB (sign bit) into the bits to drop, and
    deallocates them.

    See :class:`SignExtend` for the adjoint operation.


    Args:
        inp_dtype: input data type.
        out_dtype: output data type. must be same class as `inp_dtype`,
                   and have smaller bitsize.

    Registers:
        x (LEFT): the input register of type `inp_dtype`
        y (RIGHT): the output register of type `out_dtype`
    """

    inp_dtype: Union[QInt, QIntOnesComp]
    out_dtype: Union[QInt, QIntOnesComp]

    def __attrs_post_init__(self):
        if not isinstance(self.inp_dtype, type(self.out_dtype)):
            raise ValueError(
                f"Expected same input and output base types, got: {self.inp_dtype}, {self.out_dtype}"
            )

        if not is_symbolic(self.truncate_bitsize) and self.truncate_bitsize <= 0:
            raise ValueError(
                f"input bitsize {self.inp_dtype.num_qubits} must be larger than "
                f"output bitsize {self.out_dtype.num_qubits}"
            )

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.inp_dtype, side=Side.LEFT),
                Register('y', self.out_dtype, side=Side.RIGHT),
            ]
        )

    @cached_property
    def truncate_bitsize(self):
        return self.inp_dtype.num_qubits - self.out_dtype.num_qubits

    def adjoint(self) -> 'SignExtend':
        return SignExtend(self.out_dtype, self.inp_dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        xs = bb.split(x)
        bits_to_drop, xs = xs[: self.truncate_bitsize], xs[self.truncate_bitsize :]

        xs[0], bits_to_drop = bb.add(
            MultiTargetCNOT(self.truncate_bitsize), control=xs[0], targets=bb.join(bits_to_drop)
        )
        bb.free(bits_to_drop)
        x = bb.join(xs)

        return {'y': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {MultiTargetCNOT(self.truncate_bitsize): 1}

    def on_classical_vals(self, x: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        bits = self.inp_dtype.to_bits(int(x))

        bits_to_drop = bits[: self.truncate_bitsize]
        sign_bit = int(bits[self.truncate_bitsize])
        if any(b != sign_bit for b in bits_to_drop):
            raise ValueError(f"{bits_to_drop=} must be equal to the {sign_bit=}!")

        y = self.out_dtype.from_bits(bits[self.truncate_bitsize :])
        return {'y': y}


@bloq_example
def _sign_truncate() -> SignTruncate:
    from qualtran import QInt

    sign_truncate = SignTruncate(QInt(16), QInt(8))
    return sign_truncate


_SIGN_TRUNCATE_DOC = BloqDocSpec(bloq_cls=SignTruncate, examples=[_sign_truncate])
