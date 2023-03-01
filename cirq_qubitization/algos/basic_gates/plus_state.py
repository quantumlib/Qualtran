from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet

PLUS = np.ones(2, dtype=np.complex128) / np.sqrt(2)


@frozen
class PlusState(Bloq):
    """The |+> state.

    This is (1/sqrt(2))*(|0> + |1>).

    Registers:
     - (right) q: One qubit.
    """

    def short_name(self) -> str:
        return '+'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('q', bitsize=1, side=Side.RIGHT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == []
        assert list(outgoing.keys()) == ['q']
        out_soq = outgoing['q']
        assert isinstance(out_soq, Soquet)

        tn.add(qtn.Tensor(data=PLUS, inds=(outgoing['q'],), tags=['+', binst]))


@frozen
class PlusEffect(Bloq):
    """The <+| effect.

    This corresponds to projection into the X basis.

    Registers:
     - (left) q: One qubit.
    """

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('q', bitsize=1, side=Side.LEFT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == ['q']
        in_soq = incoming['q']
        assert isinstance(in_soq, Soquet)

        assert list(outgoing.keys()) == []

        tn.add(qtn.Tensor(data=PLUS, inds=(incoming['q'],), tags=['+', binst]))
