from functools import cached_property

from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters


@frozen
class ControlledBloq(Bloq):
    """A controlled version of `cbloq`."""

    subbloq: Bloq

    def pretty_name(self) -> str:
        return f'C[{self.subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self.subbloq.short_name()}]'

    def __str__(self) -> str:
        return f'C[{self.subbloq}]'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [FancyRegister(name="control", bitsize=1)] + list(self.subbloq.registers)
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        if not isinstance(self.subbloq, CompositeBloq):
            return ControlledBloq(self.subbloq.decompose_bloq()).decompose_bloq()

        bb, init_soqs = CompositeBloqBuilder.from_registers(
            self.subbloq.registers, add_registers_allowed=True
        )
        ctrl = bb.add_register('control', 1)

        binst_map = {}
        for binst, soqs in self.subbloq.iter_bloqsoqs(in_soqs=init_soqs, binst_map=binst_map):
            new_bloq = ControlledBloq(binst.bloq)
            new_binst, (ctrl, *_) = bb.add_2(new_bloq, control=ctrl, **soqs)
            binst_map[binst] = new_binst

        return bb.finalize(control=ctrl, **self.subbloq.final_soqs(binst_map))
