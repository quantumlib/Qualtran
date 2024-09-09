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
from typing import Dict, Tuple

from attr import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import ssum, SymbolicFloat, SymbolicInt
from qualtran.symbolics.types import is_symbolic


@frozen
class BlackBoxPrepare(Bloq):
    """Provide a black-box interface to `Prepare` bloqs.

    This wrapper uses `Partition` to combine descriptive selection
    registers into one register named "selection".

    Args:
        prepare: The bloq following the `Prepare` interface to wrap.

    Registers:
        selection: selection register.
        junk: Additional junk registers not prepared upon.
    """

    prepare: PrepareOracle

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register(name='selection', dtype=QAny(self.selection_bitsize)),)

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register(name='junk', dtype=QAny(self.junk_bitsize)),)

    @cached_property
    def junk_bitsize(self) -> SymbolicInt:
        return ssum(r.total_bits() for r in self.prepare.junk_registers)

    @cached_property
    def selection_bitsize(self) -> SymbolicInt:
        return ssum(r.total_bits() for r in self.prepare.selection_registers)

    @cached_property
    def l1_norm_of_coeffs(self) -> SymbolicFloat:
        return self.prepare.l1_norm_of_coeffs

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(selection=self.selection_bitsize, junk=self.junk_bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if self.selection_bitsize == 0:
            return soqs
        partitions = [
            (self.selection_registers[0], [r.name for r in self.prepare.selection_registers])
        ]
        if is_symbolic(self.junk_bitsize) or self.junk_bitsize > 0:
            partitions.append(
                (self.junk_registers[0], [r.name for r in self.prepare.junk_registers])
            )
        return bb.add_d(AutoPartition(self.prepare, partitions), **soqs)

    def __str__(self) -> str:
        return 'Prep'


@bloq_example
def _black_box_prepare() -> BlackBoxPrepare:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard

    prepare = PrepareHubbard(2, 2, 1, 4)
    black_box_prepare = BlackBoxPrepare(prepare=prepare)
    return black_box_prepare


_BLACK_BOX_PREPARE_DOC = BloqDocSpec(
    bloq_cls=BlackBoxPrepare,
    import_line='from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare',
    examples=[_black_box_prepare],
)
