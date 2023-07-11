from functools import cached_property

from attrs import frozen

from qualtran import Bloq, Signature


@frozen
class CtrlPauli(Bloq):
    """A multi-controlled Pauli gate.

    This is currently a placeholder gate.

    Attributes:
        pauli: one of +/- X, Y, or Z
        ctrl_bitsize: The bitsize of the control register
        cv: The control value, between 0 and 2^ctrl_bitsize.
    """

    pauli: str
    ctrl_bitsize: int
    cv: int = 0

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=self.ctrl_bitsize, target=1)

    def short_name(self) -> str:
        return self.pauli
