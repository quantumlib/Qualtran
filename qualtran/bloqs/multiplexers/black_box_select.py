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
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.multiplexers.select_base import SelectOracle


@frozen
class BlackBoxSelect(Bloq):
    r"""A 'black box' Select bloq.

    The `SELECT` operation applies the $l$'th unitary $U_{l}$ on the system register
    when the selection register stores integer $l$.
    When implementing specific `SelectOracle` bloqs, it is helpful to have multiple selection
    registers each with semantic meaning. For example: you could have spatial or spin coordinates
    on different, named registers. The `SelectOracle` interface encourages this. `BlackBoxSelect`
    uses the properties on the `SelectOracle` interface to provide a "black box" view of a select
    operation that just has a selection and system register.
    During decomposition, this bloq will use the `Partition` utility bloq to partition
    and route the parts of the unified selection register to the `Select` bloq.

    Args:
        select: The bloq implementing the `SelectOracle` interface.

    Registers:
        selection: The combined selection register
        system: The combined system register
    """

    select: SelectOracle

    def pretty_name(self) -> str:
        return 'SELECT'

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                name='selection',
                dtype=QAny((sum(r.total_bits() for r in self.select.selection_registers))),
            ),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                name='system',
                dtype=QAny((sum(r.total_bits() for r in self.select.target_registers))),
            ),
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    @cached_property
    def selection_bitsize(self) -> int:
        return self.selection_registers[0].bitsize

    @cached_property
    def system_bitsize(self) -> int:
        return self.target_registers[0].bitsize

    def build_composite_bloq(
        self, bb: BloqBuilder, selection: SoquetT, system: SoquetT
    ) -> Dict[str, SoquetT]:
        # includes selection registers and any selection registers used by PREPARE
        sel_regs = self.select.selection_registers
        sel_part = Partition(self.selection_bitsize, regs=sel_regs)
        sel_out_regs = bb.add_t(sel_part, x=selection)
        sys_regs = tuple(self.select.target_registers)
        sys_part = Partition(self.system_bitsize, regs=sys_regs)
        sys_out_regs = bb.add_t(sys_part, x=system)
        out_regs = bb.add_t(
            self.select,
            **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)},
            **{reg.name: sp for reg, sp in zip(sys_regs, sys_out_regs)},
        )
        sel_out_regs = out_regs[: len(sel_regs)]
        sys_out_regs = out_regs[len(sel_regs) :]
        selection = bb.add(
            sel_part.adjoint(), **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)}
        )
        system = bb.add(
            sys_part.adjoint(), **{reg.name: sp for reg, sp in zip(sys_regs, sys_out_regs)}
        )
        return {'selection': selection, 'system': system}


@bloq_example
def _black_box_select() -> BlackBoxSelect:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import SelectHubbard

    select = SelectHubbard(2, 2)
    black_box_select = BlackBoxSelect(select=select)
    return black_box_select


_BLACK_BOX_SELECT_DOC = BloqDocSpec(
    bloq_cls=BlackBoxSelect,
    import_line='from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect',
    examples=[_black_box_select],
)
