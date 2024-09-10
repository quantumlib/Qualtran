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
from typing import Dict

from attrs import frozen

from qualtran import bloq_example, BloqBuilder, BloqDocSpec, QAny, Signature, SoquetT
from qualtran.bloqs.basic_gates import GlobalPhase
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat, SymbolicInt


@frozen
class Phase(BlockEncoding):
    r"""Apply a phase to a block encoding.

    Given $B[A]$ as a $(\alpha, a, \epsilon)$-block encoding of $A$, produces a
    $(\alpha, a, \epsilon)$-block encoding of $\exp(i\pi\phi)A$.

    Args:
        block_encoding: The block encoding to apply a phase to.
        phi: The phase angle.
        eps: The precision of the phase angle.

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register (present only if bitsize > 0).
    """

    block_encoding: BlockEncoding
    phi: SymbolicFloat
    eps: SymbolicFloat

    @property
    def alpha(self) -> SymbolicFloat:
        return self.block_encoding.alpha

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.block_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return self.block_encoding.resource_bitsize

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.block_encoding.epsilon + self.eps

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if ancilla_bitsize is 0, not present
        )

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self.block_encoding.signal_state

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {self.block_encoding: 1, GlobalPhase(exponent=self.phi, eps=self.eps): 1}

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        bb.add(GlobalPhase(exponent=self.phi, eps=self.eps))

        return bb.add_d(self.block_encoding, **soqs)

    def __str__(self) -> str:
        return f"B[exp({self.phi}i){str(self.block_encoding)[2:-1]}]"


@bloq_example
def _phase_block_encoding() -> Phase:
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.unitary import Unitary

    phase_block_encoding = Phase(Unitary(Hadamard()), phi=0.25, eps=0)
    return phase_block_encoding


_PHASE_DOC = BloqDocSpec(bloq_cls=Phase, examples=[_phase_block_encoding])
