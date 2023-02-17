from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class ZeroState(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qubit', bitsize=1, side=Side.RIGHT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == []
        assert list(outgoing.keys()) == ['qubit']
        out_soq = outgoing['qubit']
        assert isinstance(out_soq, Soquet)

        data = np.array([1, 0], dtype=np.complex128)
        tn.add(qtn.Tensor(data=data, inds=(out_soq,), tags=['0', binst]))
