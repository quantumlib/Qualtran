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
from typing import Dict, Sequence, Tuple, TYPE_CHECKING

from attrs import field, frozen

from qualtran import bloq_example, BloqDocSpec, QAny, Register, Soquet
from qualtran.bloqs.basic_gates import Identity
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics.types import SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder


@frozen
class PrepareIdentity(PrepareOracle):
    """An identity PrepareOracle.

    This is helpful for creating a reflection about zero and as a signal state for block encodings.

    Args:
        selection_regs: The selection registers for state prepareation. These
            are the incilla the state will be prepared over.

    Registers:
        selection_registers: The selection registers.
    """

    selection_regs: Tuple[Register, ...] = field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )

    @classmethod
    def from_bitsizes(cls, bitsizes: Sequence[SymbolicInt]) -> 'PrepareIdentity':
        """Create an identity prepare oracle from register bitsizes.

        Args:
            bitsizes: A list of bitsizes for the selection registers.
        """
        return cls(tuple(Register(f'reg{i}_', QAny(b)) for i, b in enumerate(bitsizes)))

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.selection_regs

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return ()

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: Soquet) -> Dict[str, Soquet]:
        for label, soq in soqs.items():
            soqs[label] = bb.add(Identity(soq.reg.bitsize), q=soq)
        return soqs

    def adjoint(self) -> 'PrepareIdentity':
        return self


@bloq_example(generalizer=ignore_split_join)
def _prepare_identity() -> PrepareIdentity:
    prepare_identity = PrepareIdentity.from_bitsizes((10, 4, 1))
    return prepare_identity


_PREPARE_IDENTITY_DOC = BloqDocSpec(
    bloq_cls=PrepareIdentity,
    import_line='from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity',
    examples=(_prepare_identity,),
)
