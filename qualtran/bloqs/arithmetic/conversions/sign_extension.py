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

import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QDType,
    QFxp,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import is_symbolic


@frozen
class SignExtend(Bloq):
    """Sign-Extend a value to a value of larger bitsize.

    Useful to implement arithmetic operations with differing operand bitsizes.
    A sign extension copies the MSB into the new bits of the wider value. For
    example: a 4-bit to 6-bit sign-extension of `1010` gives `111010`.


    Args:
        inp_dtype: input data type.
        out_dtype: output data type. must be same class as `inp_dtype`,
                   and have larger bitsize.

    Registers:
        x (LEFT): the input register of type `inp_dtype`
        y (RIGHT): the output register of type `out_dtype`
    """

    inp_dtype: QDType
    out_dtype: QDType

    def __attrs_post_init__(self):
        if not isinstance(self.inp_dtype, type(self.out_dtype)):
            raise ValueError(
                f"Expected same input and output base types, got: {self.inp_dtype}, {self.out_dtype}"
            )

        if isinstance(self.out_dtype, QFxp):
            assert isinstance(self.inp_dtype, QFxp)  # checked above, but mypy does not realize

            if self.out_dtype.num_frac != self.inp_dtype.num_frac:
                raise ValueError(
                    f"Expected same fractional sizes for QFxp, got: {self.inp_dtype.num_frac}, {self.out_dtype.num_frac}"
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

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        extend_ys = bb.allocate(self.extend_bitsize)
        xs = bb.split(x)

        xs[0], extend_ys = bb.add(
            MultiTargetCNOT(self.extend_bitsize), control=xs[0], targets=extend_ys
        )

        extend_ys = bb.split(extend_ys)
        y = bb.join(np.concatenate([extend_ys, xs]), dtype=self.out_dtype)

        return {'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> set['BloqCountT']:
        return {(MultiTargetCNOT(self.extend_bitsize), 1)}


@bloq_example
def _sign_extend() -> SignExtend:
    from qualtran import QInt

    sign_extend = SignExtend(QInt(8), QInt(16))
    return sign_extend


@bloq_example
def _sign_extend_fxp() -> SignExtend:
    from qualtran import QFxp

    sign_extend_fxp = SignExtend(QFxp(8, 4, signed=True), QFxp(16, 4, signed=True))
    return sign_extend_fxp


_SIGN_EXTEND_DOC = BloqDocSpec(bloq_cls=SignExtend, examples=[_sign_extend, _sign_extend_fxp])
