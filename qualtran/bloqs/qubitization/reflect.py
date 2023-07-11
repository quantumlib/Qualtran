from functools import cached_property
from typing import Dict, Optional

import attrs
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.ctrl_pauli import CtrlPauli
from qualtran.bloqs.qubitization.prepare import BlackBoxPrepare
from qualtran.drawing import Circle, TextBox


@frozen
class Reflect(Bloq):
    r"""Applies reflection around a state prepared by `prepare`

    $\def\select{\mathrm{SELECT}} \def\prepare{\mathrm{PREPARE}} \def\r{\rangle} \def\l{\langle}$
    Applies $R_s = I - 2|s\r\l s|$ using $R_s = P^†(I - 2|0\r\l 0|)P$ where $P$ prepares the state
    along which we want to reflect: $P|0\r = |s\r$.

    The reflection operator that adds a $-1$ phase to all states in the subspace spanned by $|s\r$.

    Args:
        prepare: The prepare bloq.
        cv: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    prepare: BlackBoxPrepare
    cv: Optional[int] = None

    @cached_property
    def signature(self) -> Signature:
        registers = [] if self.cv is None else [Register('ctrl', 1)]
        registers.append(Register('selection', self.prepare.bitsize))
        return Signature(registers)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        pauli = '-Z'
        if self.cv is None:
            phase_trg = bb.allocate(n=1)
        else:
            phase_trg = soqs.pop('ctrl')
            if self.cv == 1:
                pauli = '+Z'

        sel = soqs.pop('selection')

        (sel,) = bb.add(self.prepare.dagger(), selection=sel)
        sel, phase_trg = bb.add(
            CtrlPauli(pauli, ctrl_bitsize=sel.reg.bitsize), ctrl=sel, target=phase_trg
        )
        (sel,) = bb.add(self.prepare, selection=sel)

        ret_soqs = {'selection': sel}
        if self.cv is None:
            bb.free(phase_trg)
        else:
            ret_soqs['ctrl'] = phase_trg
        return ret_soqs

    def controlled(self) -> 'Bloq':
        assert self.cv is None
        return attrs.evolve(self, cv=1)

    def short_name(self) -> str:
        return f'R[{self.prepare.short_name()}]'

    def wire_symbol(self, soq: Soquet):
        if soq.reg.name == 'ctrl':
            assert self.cv is not None
            return Circle(filled=self.cv == 1)
        assert soq.reg.name == 'selection'
        return TextBox(text=self.short_name())
