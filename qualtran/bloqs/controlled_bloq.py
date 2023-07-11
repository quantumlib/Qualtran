"""Bloqs that wrap subbloqs."""

from functools import cached_property
from typing import List, Tuple

from attrs import field, frozen

from qualtran import Bloq, BloqBuilder, CompositeBloq, Register, Signature, Soquet, SoquetT
from qualtran.drawing import Circle, WireSymbol


def _no_nesting_ctrls_yet(instance, field, val):
    # https://github.com/quantumlib/cirq-qubitization/issues/149
    assert isinstance(val, Bloq)
    if 'ctrl' in [reg.name for reg in val.signature]:
        raise NotImplementedError("`ControlledBloq` doesn't support nesting yet.") from None


@frozen
class ControlledBloq(Bloq):
    """A controlled version of `subbloq`."""

    subbloq: Bloq = field(validator=_no_nesting_ctrls_yet)

    def pretty_name(self) -> str:
        return f'C[{self.subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self.subbloq.short_name()}]'

    def __str__(self) -> str:
        return f'C[{self.subbloq}]'

    @cached_property
    def signature(self) -> Signature:
        return Signature((Register(name="ctrl", bitsize=1),) + tuple(self.subbloq.signature))

    def decompose_bloq(self) -> 'CompositeBloq':
        if not isinstance(self.subbloq, CompositeBloq):
            return ControlledBloq(self.subbloq.decompose_bloq()).decompose_bloq()

        bb, initial_soqs = BloqBuilder.from_signature(self.signature)
        ctrl = initial_soqs['ctrl']

        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        for binst, in_soqs, old_out_soqs in self.subbloq.iter_bloqsoqs():
            in_soqs = bb.map_soqs(in_soqs, soq_map)
            ctrl, *new_out_soqs = binst.bloq.add_controlled(bb, ctrl=ctrl, **in_soqs)
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = bb.map_soqs(self.subbloq.final_soqs(), soq_map)
        return bb.finalize(ctrl=ctrl, **fsoqs)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        return self.subbloq.wire_symbol(soq)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        return self.subbloq.wire_symbol(soq)
