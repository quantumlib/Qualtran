from typing import Sequence, Tuple, List, Dict, Optional, Iterable

import cirq
import networkx as nx

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.quantum_graph import (
    Wire,
    Soquet,
    LeftDangle,
    BloqInstance,
    DanglingT,
)


class CompositeBloq(Bloq):
    """A container type implementing the `Bloq` interface.

    Args:
        wires: A sequence of `Wire` encoding the quantum compute graph.
        registers: The registers defining the inputs and outputs of this Bloq. This
            should correspond to the dangling `Soquets` in the `wires`.
    """

    def __init__(self, wires: Sequence[Wire], registers: Registers):
        self._wires = tuple(wires)
        self._registers = registers

    @property
    def registers(self) -> Registers:
        return self._registers

    @property
    def wires(self) -> Tuple[Wire, ...]:
        return self._wires

    def to_cirq_circuit(self, **quregs: Sequence[cirq.Qid]):
        return _cbloq_to_cirq_circuit(quregs, self.wires)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("Come back later.")


def _create_binst_graph(wires: Iterable[Wire]):
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
    binst: BloqInstance, soqmap: Dict[Soquet, Sequence[cirq.Qid]], binst_graph: nx.DiGraph
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
        # Add it using the current mapping of soqmap to regnames
        bloq = binst.bloq
        quregs = {reg.name: soqmap[Soquet(binst, reg.name)] for reg in bloq.registers}
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
            soqmap[Soquet(suc, out_regname)] = soqmap[Soquet(binst, in_regname)]

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[str, Sequence[cirq.Qid]], wires: Sequence[Wire]
) -> cirq.Circuit:
    """Transform CompositeBloq components into a cirq.Circuit.

    Args:
        quregs: Named registers of `cirq.Qid` to apply the quantum compute graph to.
        wires: A sequence of `Wire` objects that define the quantum compute graph.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    # Make a graph where we just connect binsts but note in the edges what the mappings are.
    binst_graph = _create_binst_graph(wires)

    # A mapping of soquet to qubits that we update as operations are appended to the circuit.
    soqmap = {Soquet(LeftDangle, reg_name): qubits for reg_name, qubits in quregs.items()}

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
