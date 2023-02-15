import itertools
from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.basic_gates import PlusState, ZeroState
from cirq_qubitization.quantum_graph.basic_gates.cnot import COPY, XOR
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloqBuilder,
    get_soquets,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.quimb_sim import cbloq_to_dense


@frozen
class CNOT_wireshape(Bloq):
    """CNOT with two generic qubits."""

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qubits', bitsize=1, wireshape=(2,))])

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
        assert in_soqs.shape == (2,)

        assert list(outgoing.keys()) == ['qubits']
        out_soqs = outgoing['qubits']
        assert out_soqs.shape == (2,)

        internal = qtn.rand_uuid()
        tn.add(
            qtn.Tensor(data=COPY, inds=(in_soqs[0], out_soqs[0], internal), tags=['COPY', binst])
        )
        tn.add(qtn.Tensor(data=XOR, inds=(in_soqs[1], out_soqs[1], internal), tags=['XOR']))


def test_bell_wireshape():
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(PlusState())
    (q1,) = bb.add(ZeroState())

    (qubits,) = bb.add(CNOT_wireshape(), qubits=[q0, q1])

    cbloq = bb.fancy_finalize(qubits=qubits)
    vec = cbloq_to_dense(cbloq)

    should_be = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vec)


@frozen
class CNOT_bitsize(Bloq):
    """CNOT with one register of larger size."""

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(reg=2)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert sorted(incoming.keys()) == ['reg']
        in_soq = incoming['reg']
        assert isinstance(in_soq, Soquet)
        assert in_soq.reg.bitsize == 2

        assert sorted(outgoing.keys()) == ['reg']
        out_soq = outgoing['reg']
        assert isinstance(out_soq, Soquet)

        data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        tn.add(qtn.Tensor(data=data, inds=(out_soq, in_soq), tags=['CNOT', binst]))
