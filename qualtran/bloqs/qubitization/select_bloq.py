"""the Select interface and associated bloqs.

Note: we cannot name this module `select` or it breaks the builtin `selectors` module.
"""
import abc
from functools import cached_property
from typing import Dict, List

from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.util_bloqs import Partition


class Select(Bloq, metaclass=abc.ABCMeta):
    r"""The SELECT Oracle interface.

    The action of a SELECT oracle on a selection register $|l\rangle$ and target register
    $|\Psi\rangle$ can be defined as:

    $$
        \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    In other words, the `SELECT` oracle applies $l$'th unitary $U_{l}$ on the target register
    $|\Psi\rangle$ when the selection register stores integer $l$.

    $$
        \mathrm{SELECT}|l\rangle |\Psi\rangle = |l\rangle U_{l}|\Psi\rangle
    $$

    Implementations of `Select` must return `control_registers`, `selection_registers`,
    and `system_registers` so that registers can be routed by `BlackBoxSelect`.
    """

    @property
    @abc.abstractmethod
    def control_registers(self) -> List[Register]:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> List[Register]:
        ...

    @property
    @abc.abstractmethod
    def system_register(self) -> Register:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.control_registers, *self.selection_registers, self.system_register])


@frozen
class BlackBoxSelect(Bloq):
    """A 'black box' Select bloq.

    The `SELECT` operation applies the $l$'th unitary $U_{l}$ on the system register
    when the selection register stores integer $l$.

    When implementing specific `Select` bloqs, it is helpful to have multiple selection
    registers each with semantic meaning. For example: you could have spatial or spin coordinates
    on different, named registers. The `Select` interface encourages this. `BlackBoxSelect`
    uses the properties on the `Select` interface to provide a "black box" view of a select
    operation that just has a selection and system register.

    During decomposition, this bloq will use the `Partition` utility bloq to partition
    and route the parts of the unified selection register to the `Select` bloq.

    ArgS:
        select: The bloq implementing the `Select` interface.

    Registers:
     - selection: The combined selection register
     - system: The combined system register

    """

    select: Select

    @cached_property
    def selection_bitsize(self):
        return sum(reg.total_bits() for reg in self.select.selection_registers)

    @property
    def system_bitsize(self):
        return self.select.system_register.bitsize

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(
            selection=self.selection_bitsize, system=self.select.system_register.bitsize
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, selection: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        # Set up partitioners
        sel_regs = self.select.selection_registers
        # sys_regs = self.select.system_registers
        partition_sel = Partition(n=self.selection_bitsize, regs=tuple(sel_regs))
        # partition_sys = Partition(n=self.system_bitsize, regs=tuple(sys_regs))

        # partition
        sel_parts = bb.add(partition_sel, x=selection)
        # sys_parts = bb.add(partition_sys, x=system)

        # call select
        ret = bb.add(
            self.select,
            **{reg.name: sp for reg, sp in zip(sel_regs, sel_parts)},
            **{self.select.system_register.name: system}
            # **{reg.name: sp for reg, sp in zip(sys_regs, sys_parts)},
        )
        sel_parts = ret[: len(sel_regs)]
        sys_parts = ret[len(sel_regs) :]
        assert len(sys_parts) == 1
        system = sys_parts[0]
        ctrl = {}

        # un-partition
        (selection,) = bb.add(
            partition_sel.dagger(), **{reg.name: sp for reg, sp in zip(sel_regs, sel_parts)}
        )
        # (system,) = bb.add(
        #     partition_sys.dagger(), **{reg.name: sp for reg, sp in zip(sys_regs, sys_parts)}
        # )
        return {'selection': selection, 'system': system, **ctrl}

    def short_name(self) -> str:
        return 'BBSelect'


@frozen
class DummySelect(Select):
    n: int = 32

    @property
    def control_registers(self) -> List[Register]:
        return []

    @property
    def selection_registers(self) -> List[Register]:
        return list(Signature.build(p=self.n, q=self.n, spin=1))

    @property
    def system_register(self) -> Register:
        return Register(name='psi', bitsize=128)

    def short_name(self) -> str:
        return 'Sel(p,q,s)'
