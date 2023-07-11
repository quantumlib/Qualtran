from functools import cached_property
from typing import Dict, Tuple

import cirq
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.cirq_interop import CirqGateAsBloq


@frozen
class StatePrepChannel(Bloq):
    """Prepare a state using `cirq.StatePreparationChannel`.

    Attributes:
        bitsize: The number of (qu)bits in the state
        target_state: a tuple of amplitudes defining the state.
    """

    bitsize: int
    target_state: Tuple[float, ...]

    def __attrs_post_init__(self):
        if len(self.target_state) != 2**self.bitsize:
            raise ValueError(
                f"`target_state` size: wanted {2**self.bitsize}, got {len(self.target_state)}"
            )

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> Dict[str, 'SoquetT']:
        qubits = bb.split(x)
        bb.add_from(CirqGateAsBloq(cirq.StatePreparationChannel(self.target_state)), qubits=qubits)
        return {'x': bb.join(qubits)}
