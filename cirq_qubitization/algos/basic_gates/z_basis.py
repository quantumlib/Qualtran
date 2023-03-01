from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side

_ZERO = np.array([1, 0], dtype=np.complex128)
_ONE = np.array([0, 1], dtype=np.complex128)


@frozen
class ZVector(Bloq):
    """The |0> or |1> state or effect.

    Args:
        bit: False chooses |0>, True chooses |1>
        side: True means this is a state with right registers; False means this is an
            effect with left registers.

    """

    bit: bool
    state: bool = True

    def pretty_name(self) -> str:
        s = self.short_name()
        return f'|{s}>' if self.state else f'<{s}|'

    def short_name(self) -> str:
        return '1' if self.bit else '0'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('q', bitsize=1, side=Side.RIGHT if self.state else Side.LEFT)]
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        side = outgoing if self.state else incoming
        tn.add(
            qtn.Tensor(
                data=_ONE if self.bit else _ZERO, inds=(side['q'],), tags=[self.short_name(), binst]
            )
        )


ZERO = ZVector(False)
ONE = ZVector(True)
ZERO_EFFECT = ZVector(False, state=False)
ONE_EFFECT = ZVector(True, state=True)
