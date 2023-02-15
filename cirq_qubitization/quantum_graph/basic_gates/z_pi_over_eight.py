import itertools
from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters

COPY = [1.0, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY).reshape((2, 2, 2))


@frozen
class ZPiOverEight(Bloq):
    n: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qubits', bitsize=1, wireshape=(self.n,))])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == ['qubits']
        in_soqs = incoming['qubits']
        assert in_soqs.shape == (self.n,)

        assert list(outgoing.keys()) == ['qubits']
        out_soqs = outgoing['qubits']
        assert out_soqs.shape == (self.n,)

        zdata = np.array(list(itertools.product([1, -1], repeat=self.n)))
        zdata = np.product(zdata, axis=1)
        zdata = zdata.reshape((2,) * self.n)
        zdata = np.exp(-1.0j * zdata * np.pi / (8 * 2))

        internal_edges = []
        for i in range(self.n):
            ie = qtn.rand_uuid()
            tn.add(qtn.Tensor(data=COPY, inds=(in_soqs[i], out_soqs[i], ie), tags=['COPY']))
            internal_edges.append(ie)

        tn.add(qtn.Tensor(data=zdata, inds=internal_edges, tags=['Z', binst]))
