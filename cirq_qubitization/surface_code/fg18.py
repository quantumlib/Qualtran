from typing import Tuple

from cirq_qubitization.surface_code.magic_state_factory import MagicStateFactory, SimpleTFactory


class FG18Factory(SimpleTFactory):
    """Two-level 15-to-1 distillation.

    Explicit l0d=7, l1d = 15, l2d=31.


    References:
        Low overhead..., Fowler, Gidney (2018).
    """

    def __init__(self):
        super().__init__(d=31)

    @property
    def quop_dimensions(self) -> Tuple[float, float, float]:
        return (8, 12, 6.5)

    def error_for_t(self, inp_err: float) -> float:
        l1 = 35 * inp_err**3
        l2 = 35 * l1**3
        return l2

    def rejection_prob(self, inp_err: float) -> float:
        l1 = 35 * inp_err**3
        return 15 * l1
