from functools import cached_property
from typing import Dict

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import DanglingT, RightDangle, Soquet
from cirq_qubitization.quantum_graph.quimb_sim import (
    _get_dangling_soquets,
    bloq_to_dense,
    cbloq_to_quimb,
)
from cirq_qubitization.quantum_graph.util_bloqs import Join


def test_get_soquets():
    soqs = _get_dangling_soquets(Join(10).registers, right=True)
    assert list(soqs.keys()) == ['join']
    soq = soqs['join']
    assert soq.binst == RightDangle
    assert soq.reg.bitsize == 10

    soqs = _get_dangling_soquets(Join(10).registers, right=False)
    assert list(soqs.keys()) == ['join']
    soq = soqs['join']
    assert soq.shape == (10,)
    assert soq[0].reg.bitsize == 1


@frozen
class TensorAdderTester(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [
                FancyRegister('x', bitsize=2, side=Side.LEFT),
                FancyRegister('qubits', bitsize=1, wireshape=(2,)),
                FancyRegister('y', bitsize=1, side=Side.RIGHT),
            ]
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == ['x', 'qubits']
        in_qubits = incoming['qubits']
        assert in_qubits.shape == (2,)
        assert incoming['x'].reg.bitsize == 2

        assert list(outgoing.keys()) == ['qubits', 'y']
        out_qubits = outgoing['qubits']
        assert out_qubits.shape == (2,)
        assert outgoing['y'].reg.bitsize == 1

        data = np.zeros((2**2, 2, 2, 2, 2, 2))
        data[3, 0, 1, 0, 1, 0] = 1
        tn.add(
            qtn.Tensor(
                data=data,
                inds=(
                    incoming['x'],
                    in_qubits[0],
                    in_qubits[1],
                    outgoing['y'],
                    out_qubits[0],
                    out_qubits[1],
                ),
                tags=[tag],
            )
        )


def test_bloq_to_dense():
    mat = bloq_to_dense(TensorAdderTester())
    # Right inds: qubits=(1,0), y=0
    right = 1 * 2**2 + 0 * 2**1 + 0 * 2**0

    # Left inds: x=3, qubits=(0,1)
    left = 3 * 2**2 + 0 * 2**1 + 1 * 2**0

    assert np.where(mat) == (right, left)


@frozen
class TensorAdderSimple(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(x=1)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == ['x']
        assert list(outgoing.keys()) == ['x']
        tn.add(qtn.Tensor(data=np.eye(2), inds=(incoming['x'], outgoing['x']), tags=[tag]))


def test_cbloq_to_quimb():
    bb = CompositeBloqBuilder()
    x = bb.add_register('x', 1)
    (x,) = bb.add(TensorAdderSimple(), x=x)
    (x,) = bb.add(TensorAdderSimple(), x=x)
    (x,) = bb.add(TensorAdderSimple(), x=x)
    (x,) = bb.add(TensorAdderSimple(), x=x)
    cbloq = bb.finalize(x=x)

    tn, _ = cbloq_to_quimb(cbloq)
    assert len(tn.tensors) == 4
    for oi in tn.outer_inds():
        assert isinstance(oi, Soquet)
        assert isinstance(oi.binst, DanglingT)
