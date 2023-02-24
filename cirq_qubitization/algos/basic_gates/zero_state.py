from functools import cached_property
from typing import Any, Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side

ZERO = np.array([1, 0], dtype=np.complex128)
ONE = np.array([0, 1], dtype=np.complex128)


@frozen
class ZeroState(Bloq):
    """The |0> state."""

    def short_name(self) -> str:
        return '0'

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
        tn.add(qtn.Tensor(data=ZERO, inds=(outgoing['q'],), tags=['0', binst]))


@frozen
class OneState(Bloq):
    """The |0> state."""

    def short_name(self) -> str:
        return '1'

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
        tn.add(qtn.Tensor(data=ONE, inds=(outgoing['q'],), tags=['0', binst]))


@frozen
class ZeroEffect(Bloq):
    """The <0| effect.

    This corresponds to projection into the Z basis.

    Registers:
     - (left) q: One qubit.
    """

    def short_name(self) -> str:
        return '0'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('q', bitsize=1, side=Side.LEFT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        tn.add(qtn.Tensor(data=ZERO, inds=(incoming['q'],), tags=['0', tag]))


@frozen
class OneEffect(Bloq):
    """The <1| effect.

    This corresponds to projection into the Z basis.

    Registers:
     - (left) q: One qubit.
    """

    def short_name(self) -> str:
        return '1'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('q', bitsize=1, side=Side.LEFT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        tn.add(qtn.Tensor(data=ONE, inds=(incoming['q'],), tags=['1', tag]))
