"""Bloqs that wrap subbloqs."""

from functools import cached_property
from typing import List, Tuple

from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    CtrlRegister,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.drawing import Circle, WireSymbol


@frozen
class ControlledBloq(Bloq):
    """A controlled version of `subbloq`."""

    subbloq: Bloq
    ctrl_name: str = 'ctrl'
    cv: int = 1

    def __attrs_post_init__(self):
        # https://github.com/quantumlib/cirq-qubitization/issues/149
        if self.ctrl_name in [reg.name for reg in self.subbloq.signature]:
            raise NotImplementedError("`ControlledBloq` doesn't support nesting yet.") from None

    @classmethod
    def from_ctrl_register(cls, subbloq: Bloq, ctrl_reg: CtrlRegister):
        if ctrl_reg.bitsize != 1:
            raise ValueError("Currently only bitsize of 1 is supported.")
        if ctrl_reg.shape != ():
            raise ValueError("Currently only 0-dim ctrlspecs are supported.")
        return cls(subbloq=subbloq, ctrl_name=ctrl_reg.name, cv=ctrl_reg.cv)

    def pretty_name(self) -> str:
        return f'C[{self.subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self.subbloq.short_name()}]'

    def __str__(self) -> str:
        return f'C[{self.subbloq}]'

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            (Register(name=self.ctrl_name, bitsize=1),) + tuple(self.subbloq.signature)
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        sub_cbloq = self.subbloq.decompose_bloq()
        ctrl_reg = CtrlRegister(name=self.ctrl_name, bitsize=1, shape=(), cv=self.cv)

        bb, initial_soqs = BloqBuilder.from_signature(self.signature)
        ctrl = initial_soqs[self.ctrl_name]

        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        for binst, in_soqs, old_out_soqs in sub_cbloq.iter_bloqsoqs():
            in_soqs = bb.map_soqs(in_soqs, soq_map)
            ctrl, *new_out_soqs = binst.bloq.add_controlled(
                bb, soqs=in_soqs, ctrl_soq=ctrl, ctrl_reg=ctrl_reg
            )
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = bb.map_soqs(sub_cbloq.final_soqs(), soq_map)
        return bb.finalize(**{self.ctrl_name: ctrl}, **fsoqs)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == self.ctrl_name:
            return Circle(filled=self.cv == 1)
        return self.subbloq.wire_symbol(soq)
