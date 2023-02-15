import itertools
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import quimb
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _binst_to_cxns,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

COPY = [1.0, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2))


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


@frozen
class PlusState(Bloq):
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

        data = np.ones(2, dtype=np.complex64) / np.sqrt(2)
        tn.add(qtn.Tensor(data=data, inds=(out_soq,), tags=['+', binst]))

    def on_registers(self, **quregs):
        print('quregs', quregs)


@frozen
class MeasX(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qubit', bitsize=1, side=Side.LEFT)])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert list(incoming.keys()) == ['qubit']
        in_soq = incoming['qubit']
        assert isinstance(in_soq, Soquet)

        assert list(outgoing.keys()) == []

        data = np.ones(2, dtype=np.complex64) / np.sqrt(2)
        tn.add(qtn.Tensor(data=data, inds=(in_soq,), tags=['Mx', binst]))


class TIdentity(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qs', bitsize=1, wireshape=(5,))])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', qs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        for i in range(5):
            qs[i] = bb.add(ZPiOverEight(n=1), qubits=[qs[i]])[0][0]

        qs[[1, 2, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 2, 3]])[0]  # 5
        qs[[0, 1, 2]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 2]])[0]  # 6
        qs[[0, 1, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 3]])[0]  # 7
        qs[[0, 2, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 2, 3]])[0]  # 8
        qs[[0, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 3, 4]])[0]  # 9
        qs[[0, 1, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 4]])[0]  # 10
        qs[[0, 2, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 2, 4]])[0]  # 11
        qs = bb.add(ZPiOverEight(n=5), qubits=qs)[0]  # 12
        qs[[2, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[2, 3, 4]])[0]  # 13
        qs[[1, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 3, 4]])[0]  # 14
        qs[[1, 2, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 2, 4]])[0]  # 15

        return {'qs': qs}


class TFactory(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('magic', bitsize=1, side=Side.RIGHT)])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        qs = np.array([bb.add(PlusState())[0] for _ in range(5)])
        for i in range(1, 5):
            # 1 - 4
            qs[i] = bb.add(ZPiOverEight(n=1), qubits=[qs[i]])[0][0]

        qs[[1, 2, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 2, 3]])[0]  # 5
        qs[[0, 1, 2]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 2]])[0]  # 6
        qs[[0, 1, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 3]])[0]  # 7
        qs[[0, 2, 3]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 2, 3]])[0]  # 8
        qs[[0, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 3, 4]])[0]  # 9
        qs[[0, 1, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 1, 4]])[0]  # 10
        qs[[0, 2, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[0, 2, 4]])[0]  # 11
        qs = bb.add(ZPiOverEight(n=5), qubits=qs)[0]  # 12
        qs[[2, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[2, 3, 4]])[0]  # 13
        qs[[1, 3, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 3, 4]])[0]  # 14
        qs[[1, 2, 4]] = bb.add(ZPiOverEight(n=3), qubits=qs[[1, 2, 4]])[0]  # 15

        for i in range(1, 5):
            bb.add(MeasX(), qubit=qs[i])

        return {'magic': qs[0]}


def blow_up_soquets(regs: Iterable[FancyRegister], binst=LeftDangle):
    all_soqs: Dict[str, SoquetT] = {}
    soqs: SoquetT
    for reg in regs:
        if reg.wireshape:
            soqs = np.empty(reg.wireshape, dtype=object)
            for ri in reg.wire_idxs():
                soq = Soquet(binst, reg, idx=ri)
                soqs[ri] = soq
        else:
            # Annoyingly, this must be a special case.
            # Otherwise, x[i] = thing will nest *array* objects because our ndarray's type is
            # 'object'. This wouldn't happen--for example--with an integer array.
            soqs = Soquet(binst, reg)

        all_soqs[reg.name] = soqs
    return all_soqs


def binstgraph_to_quimb(bg: nx.DiGraph, pos=None):
    tn = qtn.TensorNetwork([])
    fix = {}

    for gen_i, gen in enumerate(nx.topological_generations(bg)):
        for binst in gen:
            incoming, outgoing = _binst_to_cxns(binst, binst_graph=bg)

            if isinstance(binst, DanglingT):
                continue

            bloq = binst.bloq
            assert isinstance(bloq, Bloq)

            inc_d: Dict[str, SoquetT] = {}
            for reg in bloq.registers.lefts():
                if reg.wireshape:
                    soqarr = np.empty(reg.wireshape, dtype=object)
                    inc_d[reg.name] = soqarr

            for cxn in incoming:
                if cxn.right.reg.wireshape:
                    inc_d[cxn.right.reg.name][cxn.right.idx] = cxn.left
                else:
                    inc_d[cxn.right.reg.name] = cxn.left

            out_d: Dict[str, SoquetT] = {}
            for reg in bloq.registers.rights():
                if reg.wireshape:
                    out_d[reg.name] = np.empty(reg.wireshape, dtype=object)

            for cxn in outgoing:
                if isinstance(cxn.right.binst, DanglingT):
                    # Usually, we use our own side of things to name our indices
                    # but our dangling indices we use the public API of the cbloq, therefore
                    # the right soquet.
                    assign = cxn.right
                else:
                    assign = cxn.left

                if cxn.left.reg.wireshape:
                    out_d[cxn.left.reg.name][cxn.left.idx] = assign
                else:
                    out_d[cxn.left.reg.name] = assign

            bloq.add_my_tensors(tn, binst, incoming=inc_d, outgoing=out_d)
            if pos is not None:
                fix[tuple([binst])] = pos[binst]

    return tn, fix


def binstgraph_to_musical_score(bg: nx.DiGraph):
    # list of generations, each is whether the line is participating
    score: List[Tuple[Optional[int], ...]] = []

    # ???
    soqs: List[Soquet] = []

    for gen_i, gen in enumerate(nx.topological_generations(bg)):
        moment = [None] * len(soqs)

        for binst in gen:
            incoming, outgoing = _binst_to_cxns(binst, binst_graph=bg)

            if isinstance(binst, DanglingT):
                for cxn in outgoing:
                    soqs.append(cxn.left)
                continue

            bloq = binst.bloq
            assert isinstance(bloq, Bloq)

            for cxn, outcxn in zip(incoming, outgoing):
                # TODO: can't just zip; need to correlate.
                lookup = cxn.left

                i = soqs.index(lookup)
                moment[i] = binst.i
                soqs[i] = outcxn.left

        score.append(tuple(moment))

    return score
