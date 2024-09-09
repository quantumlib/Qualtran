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
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QInt, QIntOnesComp, Register, Side, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Take the sign qubit as a control and cnot the remaining qubits, then
        # add it to the remaining n-1 bits.
        return {Toffoli(): (self.bitsize - 2)}


@bloq_example
def _signed_to_twos() -> SignedIntegerToTwosComplement:
    signed_to_twos = SignedIntegerToTwosComplement(bitsize=10)
    return signed_to_twos


_SIGNED_TO_TWOS = BloqDocSpec(bloq_cls=SignedIntegerToTwosComplement, examples=[_signed_to_twos])
