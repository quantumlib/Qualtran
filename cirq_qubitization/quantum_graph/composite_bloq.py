import itertools
from typing import Sequence, Tuple, Set, List, Dict, Optional, Iterable, Union

import networkx as nx
import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.fancy_registers import Soquets, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    Connection,
    Wire,
    LeftDangle,
    RightDangle,
    BloqInstance,
    DanglingT,
)
from cirq_qubitization.quantum_graph.util_bloqs import Split, Join


class CompositeBloq(Bloq):
    """A container type implementing the `Bloq` interface.

    Args:
        wires: A sequence of `Connection` encoding the quantum compute graph.
        registers: The registers defining the inputs and outputs of this Bloq. This
            should correspond to the dangling `Soquets` in the `wires`.
    """

    def __init__(self, wires: Sequence[Connection], soquets: Soquets):
        self._wires = tuple(wires)
        self._soquets = soquets

    @property
    def soquets(self) -> Soquets:
        return self._soquets

    @property
    def cxns(self) -> Tuple[Connection, ...]:
        return self._wires

    def to_cirq_circuit(self, **quregs: Sequence[cirq.Qid]):
        return _cbloq_to_cirq_circuit(quregs, self.cxns)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("Come back later.")


def _create_binst_graph(wires: Iterable[Connection]):
    """Helper function to create a NetworkX so we can topologically visit BloqInstances.

    `CompositeBloq` defines a directed acyclic graph, so we can iterate in (time) order.
    Here, we make two changes to our view of the graph:
        1. Our nodes are now BloqInstances because they are the objects to time-order. Register
           connections are added as edge attributes.
        2. We use networkx so we can use their algorithms for topological sorting.
    """
    binst_graph = nx.DiGraph()
    for wire in wires:
        binst_edge = (wire.left.binst, wire.right.binst)
        if binst_edge in binst_graph.edges:
            binst_graph.edges[binst_edge]['conns'].append((wire.left.reg_name, wire.right.reg_name))
        else:
            binst_graph.add_edge(*binst_edge, conns=[(wire.left.reg_name, wire.right.reg_name)])
    return binst_graph


def _process_binst(
    binst: BloqInstance, soqmap: Dict[Wire, Sequence[cirq.Qid]], binst_graph: nx.DiGraph
) -> Optional[cirq.Operation]:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soqmap: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    if not isinstance(binst, DanglingT):
        # TODO: dealing with dangling is annoying.

        # First: fix internal connections, i.e. differing in- and out- names
        # for reg in binst.bloq.soquets:
        #     if isinstance(reg, SplitRegister):
        #         for i in range(reg.bitsize):
        #             soqmap[Wire(binst, f'{reg.name}{i}')] = soqmap[Wire(binst, reg.name)][
        #                 i : i + 1
        #             ]
        #     elif isinstance(reg, JoinRegister):
        #         _qq = []
        #         for i in range(reg.bitsize):
        #             _qq.extend(soqmap[Wire(binst, f'{reg.name}{i}')])
        #         soqmap[Wire(binst, reg.name)] = _qq
        #     elif isinstance(reg, ApplyFRegister):
        #         soqmap[Wire(binst, reg.out_name)] = soqmap[Wire(binst, reg.name)]
        #     else:
        #         pass

        # Then add it using the current mapping of soqmap to regnames
        bloq = binst.bloq
        quregs = {reg.name: soqmap[Wire(binst, reg.name)] for reg in bloq.soquets}
        try:
            op = bloq.on_registers(**quregs)
        except NoCirqEquivalent:
            op = None
    else:
        op = None

    # Finally: track name updates for successors
    for suc in binst_graph.successors(binst):
        reg_conns = binst_graph.edges[binst, suc]['conns']
        for in_regname, out_regname in reg_conns:
            soqmap[Wire(suc, out_regname)] = soqmap[Wire(binst, in_regname)]

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[str, Sequence[cirq.Qid]], wires: Sequence[Connection]
) -> cirq.Circuit:
    """Transform CompositeBloq components into a cirq.Circuit.

    Args:
        quregs: Named registers of `cirq.Qid` to apply the quantum compute graph to.
        wires: A sequence of `Connection` objects that define the quantum compute graph.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    # Make a graph where we just connect binsts but note in the edges what the mappings are.
    binst_graph = _create_binst_graph(wires)

    # A mapping of soquet to qubits that we update as operations are appended to the circuit.
    soqmap = {Wire(LeftDangle, reg_name): qubits for reg_name, qubits in quregs.items()}

    moments: List[cirq.Moment] = []
    for i, binsts in enumerate(nx.topological_generations(binst_graph)):
        mom: List[cirq.Operation] = []
        for binst in binsts:
            op = _process_binst(binst, soqmap, binst_graph)
            if op:
                mom.append(op)
        if mom:
            moments.append(cirq.Moment(mom))

    return cirq.Circuit(moments)


class CompositeBloqBuilder:
    def __init__(self, parent_soqs: Soquets):
        # Our builder builds a sequence of Wires
        self._cxns: List[Connection] = []

        # Initialize our BloqInstance counter
        self._i = 0

        # TODO: enforce linearity
        self._used: Set[Wire] = set()

        self._parent_soqs = parent_soqs

    def initial_soquets(self) -> Dict[str, Wire]:
        ret = {}
        for soq in self._parent_soqs.lefts():
            if soq.wireshape == tuple():
                out = Wire(LeftDangle, soq)
            else:
                out = np.empty(soq.wireshape, dtype=object)
                for ri in itertools.product(*[range(sh) for sh in soq.wireshape]):
                    out[ri] = Wire(LeftDangle, soq, idx=ri)

            ret[soq.name] = out
        return ret

    def _new_binst(self, bloq: Bloq):
        # TODO: bloqinstance has reference to parent bloq to make equality work?
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def split(self, prev_wire: Wire, n: int) -> NDArray[Wire]:
        bloq = Split(n)
        (soq,) = bloq.soquets.lefts()
        (ret,) = self.add(bloq, **{soq.name: prev_wire})
        return ret

    def join(self, prev_wires: Sequence[Wire]) -> Wire:
        bloq = Join(
            tuple(w.soq.bitsize for w in prev_wires)
        )  # TODO: prev_soqs may be of differing size
        regs = bloq.soquets.lefts()
        (out_soq,) = self.add(bloq, **{reg.name: wire for reg, wire in zip(regs, prev_wires)})
        return out_soq

    def validate_cxn(self, cxn: Connection):
        assert isinstance(cxn.left, Wire)
        assert isinstance(cxn.right, Wire)
        pass

    def add(self, bloq: Bloq, **wire_map: ArrayLike) -> Tuple[NDArray[Wire], ...]:
        # TODO: rename method?
        binst = self._new_binst(bloq)

        for soq in bloq.soquets.lefts():
            # if we want fancy indexing (which we do), we need numpy
            # this also supports length-zero indexing natively, which is good too.
            in_wires = np.asarray(wire_map[soq.name])
            for li in soq.wire_idxs():
                cxn = Connection(in_wires[li], Wire(binst, soq, idx=li))
                self.validate_cxn(cxn)
                self._cxns.append(cxn)

        out_wires = []
        for soq in bloq.soquets.rights():
            out = np.empty(soq.wireshape, dtype=object)
            for ri in itertools.product(*[range(sh) for sh in soq.wireshape]):
                out[ri] = Wire(binst, soq, idx=ri)

            out_wires.append(out)

        return tuple(out_wires)

    def finalize(self, **wire_map: Wire) -> CompositeBloq:
        # TODO: rename method?

        for soq in self._parent_soqs.rights():
            in_wire = np.asarray(wire_map[soq.name])

            for li in soq.wire_idxs():
                self._cxns.append(Connection(in_wire[li], Wire(RightDangle, soq, idx=li)))

        # TODO: remove things from used
        # TODO: assert used() is empty

        return CompositeBloq(wires=self._cxns, soquets=self._parent_soqs)
