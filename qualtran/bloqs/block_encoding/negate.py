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
from typing import Dict, List, Set, Tuple, Union

from attrs import frozen

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import XGate, ZGate
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat, SymbolicInt


@frozen
class Negate(BlockEncoding):
    r"""Negation of a block encoding.

    Given $B[A]$ as a $(\alpha, a, \epsilon)$-block encoding of $A$, produces a
    $(\alpha, a, \epsilon)$-block encoding $B[-A]$ of $-A$.
    This Bloq uses one resource qubit to induce a phase flip and restores the qubit to zero.

    Args:
        block_encoding: The block encoding to negate.

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register.
    """

    block_encoding: BlockEncoding

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
        return self.block_encoding.resource_bitsize + 1

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.block_encoding.epsilon

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),
        )

    def pretty_name(self) -> str:
        return f"B[-{self.block_encoding.pretty_name()[2:-1]}]"

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.signature.rights())

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("resource"),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),) if self.ancilla_bitsize > 0 else ()

    @property
    def signal_state(self) -> PrepareOracle:
        # This method will be implemented in the future after PrepareOracle
        # is updated for the BlockEncoding interface.
        # GitHub issue: https://github.com/quantumlib/Qualtran/issues/1104
        raise NotImplementedError

    def build_call_graph(self, ssa: SympySymbolAllocator) -> Set[BloqCountT]:
        return {(self.block_encoding, 1)}

    def build_composite_bloq(
        self, bb: BloqBuilder, resource: Soquet, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        resource_bits = bb.split(resource)
        resource_bits[0] = bb.add(XGate(), q=resource_bits[0])
        resource_bits[0] = bb.add(ZGate(), q=resource_bits[0])
        resource_bits[0] = bb.add(XGate(), q=resource_bits[0])
        resource = bb.join(resource_bits)

        partition: List[Tuple[Register, List[Union[str, Unused]]]] = [
            (self.signature.get_left("system"), ["system"])
        ]
        ancilla = self.signature._lefts.get("ancilla")
        if ancilla is not None:
            partition.append((ancilla, ["ancilla"]))
        resource_regs: List[Union[str, Unused]] = [Unused(1)]
        assert isinstance(self.block_encoding.resource_bitsize, int)
        if self.block_encoding.resource_bitsize > 0:
            resource_regs.append("resource")
        partition.append((self.signature.get_left("resource"), resource_regs))

        return bb.add_d(
            AutoPartition(self.block_encoding, partition, left_only=False),
            resource=resource,
            **soqs,
        )


@bloq_example
def _negate_block_encoding() -> Negate:
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    negate_block_encoding = Negate(Unitary(TGate()))
    return negate_block_encoding


_NEGATE_DOC = BloqDocSpec(
    bloq_cls=Negate,
    import_line="from qualtran.bloqs.block_encoding import Negate",
    examples=[_negate_block_encoding],
)
