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

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, QAny, Side, Signature, SoquetT
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat, SymbolicInt


@frozen
class Unitary(BlockEncoding):
    r"""Trivial block encoding of a unitary operator.

    Builds the block encoding as
    $
        B[U] = U
    $
    where $U$ is a unitary operator. Here, $B[U]$ is a $(1, 0, 0)$-block encoding of $U$.

    Args:
        U: The unitary operator to block-encode.
        alpha: The normalization factor (default 1).
        ancilla_bitsize: The number of ancilla bits (default 0).
        resource_bitsize: The number of resource bits (default 0).
        epsilon: The precision parameter (default 0).

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register (present only if bitsize > 0).
    """

    U: Bloq
    alpha: SymbolicFloat = 1
    ancilla_bitsize: SymbolicInt = 0
    resource_bitsize: SymbolicInt = 0
    epsilon: SymbolicFloat = 0

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return sum(r.bitsize for r in self.U.signature)

    def __attrs_post_init__(self):
        if not all(r.side == Side.THRU for r in self.U.signature):
            raise ValueError("Block encoded unitary must have all THRU registers.")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {self.U: 1}

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        partitions = [(self.signature.get_left("system"), tuple(r.name for r in self.U.signature))]
        return {
            "system": bb.add_and_partition(self.U, partitions=partitions, system=system),
            **soqs,
        }

    def __str__(self) -> str:
        return f"B[{self.U}]"


@bloq_example
def _unitary_block_encoding() -> Unitary:
    from qualtran.bloqs.basic_gates import TGate

    unitary_block_encoding = Unitary(TGate())
    return unitary_block_encoding


@bloq_example
def _unitary_block_encoding_properties() -> Unitary:
    from attrs import evolve

    from qualtran.bloqs.basic_gates import TGate

    unitary_block_encoding_properties = evolve(
        Unitary(TGate()), alpha=0.5, ancilla_bitsize=2, resource_bitsize=1, epsilon=0.01
    )
    return unitary_block_encoding_properties


_UNITARY_DOC = BloqDocSpec(
    bloq_cls=Unitary, examples=[_unitary_block_encoding, _unitary_block_encoding_properties]
)
