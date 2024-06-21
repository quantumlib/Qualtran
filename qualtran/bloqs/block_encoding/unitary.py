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
from typing import Dict, Tuple

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QDType,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.symbolics import SymbolicFloat


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
        dtype: The quantum data type of the system `U` operates over.

    Registers:
        system: The system register.
        ancilla: The ancilla register (size 0).
        resource: The resource register (size 0).
    """

    U: Bloq
    dtype: QDType
    alpha: SymbolicFloat = 1
    num_ancillas: int = 0
    num_resource: int = 0
    epsilon: SymbolicFloat = 0

    def __attrs_post_init__(self):
        assert self.U.signature == self.U.signature.adjoint()
        assert self.dtype.num_qubits == sum(r.bitsize for r in self.U.signature.rights())

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(system=self.dtype, ancilla=QAny(0), resource=QAny(0))

    def pretty_name(self) -> str:
        return f"B[{self.U.pretty_name()}]"

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.signature.rights())

    junk_registers: Tuple[Register, ...] = ()
    selection_registers: Tuple[Register, ...] = ()

    @property
    def signal_state(self) -> PrepareOracle:
        raise NotImplementedError

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT, resource: SoquetT
    ) -> Dict[str, SoquetT]:
        partitions = [
            (self.signature.get_left("system"), tuple(r.name for r in self.U.signature.lefts()))
        ]
        return {
            "system": bb.add_and_partition(self.U, partitions=partitions, system=system),
            "ancilla": ancilla,
            "resource": resource,
        }


@bloq_example
def _unitary_block_encoding() -> Unitary:
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import TGate

    unitary_block_encoding = Unitary(TGate(), dtype=QBit())
    return unitary_block_encoding


_UNITARY_DOC = BloqDocSpec(
    bloq_cls=Unitary,
    import_line="from qualtran.bloqs.block_encoding import Unitary",
    examples=[_unitary_block_encoding],
)
