from functools import cached_property
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cirq
import networkx as nx

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.quantum_graph import (
    Connection,
    LeftDangle,
    RightDangle,
    BloqInstance,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


class CompositeBloq(Bloq):
    """A container type implementing the `Bloq` interface.

    Args:
        cxns: A sequence of `Connection` encoding the quantum compute graph.
        registers: The registers defining the inputs and outputs of this Bloq. This
            should correspond to the dangling `Soquets` in the `cxns`.
    """

    def __init__(self, cxns: Sequence[Connection], registers: Registers):
        self._cxns = tuple(cxns)
        self._registers = registers

    @property
    def registers(self) -> Registers:
        return self._registers

    @property
    def connections(self) -> Tuple[Connection, ...]:
        return self._cxns

    @cached_property
    def bloq_instances(self) -> Set[BloqInstance]:
        """The set of BloqInstances making up the nodes of the graph."""
        return {
            soq.binst
            for cxn in self._cxns
            for soq in [cxn.left, cxn.right]
            if not isinstance(soq.binst, DanglingT)
        }

    def to_cirq_circuit(self, **quregs: Sequence[cirq.Qid]):
        return _cbloq_to_cirq_circuit(quregs, self.connections)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("Come back later.")


def _create_binst_graph(cxns: Iterable[Connection]) -> nx.Graph:
    """Helper function to create a NetworkX so we can topologically visit BloqInstances.

    `CompositeBloq` defines a directed acyclic graph, so we can iterate in (time) order.
    Here, we make two changes to our view of the graph:
        1. Our nodes are now BloqInstances because they are the objects to time-order. Register
           connections are added as edge attributes.
        2. We use networkx so we can use their algorithms for topological sorting.
    """
    binst_graph = nx.DiGraph()
    for cxn in cxns:
        binst_edge = (cxn.left.binst, cxn.right.binst)
        if binst_edge in binst_graph.edges:
            binst_graph.edges[binst_edge]['cxns'].append((cxn.left.reg_name, cxn.right.reg_name))
        else:
            binst_graph.add_edge(*binst_edge, cxns=[(cxn.left.reg_name, cxn.right.reg_name)])
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
        reg_conns = binst_graph.edges[binst, suc]['cxns']
        for in_regname, out_regname in reg_conns:
            soqmap[Soquet(suc, out_regname)] = soqmap[Soquet(binst, in_regname)]

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[str, Sequence[cirq.Qid]], cxns: Sequence[Connection]
) -> cirq.Circuit:
    """Transform CompositeBloq components into a cirq.Circuit.

    Args:
        quregs: Named registers of `cirq.Qid` to apply the quantum compute graph to.
        cxns: A sequence of `Connection` objects that define the quantum compute graph.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    # Make a graph where we just connect binsts but note in the edges what the mappings are.
    binst_graph = _create_binst_graph(cxns)

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


class BloqBuilderError(ValueError):
    """A value error raised during composite bloq building."""


class CompositeBloqBuilder:
    """A builder class for constructing a `CompositeBloq`.

    Users should not instantiate a CompositeBloqBuilder directly. To build a composite bloq,
    override `Bloq.build_composite_bloq`. A properly-initialized builder instance will be
    provided as the first argument.

    Args:
        parent_regs: The `Registers` argument for the parent bloq.
    """

    def __init__(self, parent_regs: Registers):
        # To be appended to:
        self._cxns: List[Connection] = []

        # Initialize our BloqInstance counter
        self._i = 0

        # Linear types! Soquets must be used exactly once.
        self._initial_soquets = {reg.name: Soquet(LeftDangle, reg.name) for reg in parent_regs}
        self._available: Set[Soquet] = set(self._initial_soquets.values())

        self._parent_regs = parent_regs

    def initial_soquets(self) -> Dict[str, Soquet]:
        """Input soquets (by name) to start building a quantum compute graph."""
        return self._initial_soquets

    def _new_binst(self, bloq: Bloq) -> BloqInstance:
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def add(self, bloq: Bloq, **in_soqs: Soquet) -> Tuple[Soquet, ...]:
        """Add a new bloq instance to the compute graph.

        Args:
            bloq: The bloq representing the operation to add.
            **in_soqs: Keyword arguments mapping the new bloq's register names to input
                `Soquet`s, e.g. the output soquets from a prior operation.

        Returns:
            A `Soquet` for each output register ordered according to `bloq.registers`.
                Note: Analogous to a Python function call using kwargs and multiple return values,
                the ordering is irrespective of the order of `in_soqs` that have been passed in
                and depends only on the convention of the bloq's registers.
        """
        binst = self._new_binst(bloq)

        out_soqs = []
        for reg in bloq.registers:
            try:
                in_soq = in_soqs[reg.name]
            except KeyError:
                raise BloqBuilderError(
                    f"{bloq} requires an input Soquet named `{reg.name}`."
                ) from None

            try:
                self._available.remove(in_soq)
            except KeyError:
                raise BloqBuilderError(
                    f"{in_soq} is not an available input Soquet for {reg}."
                ) from None

            del in_soqs[reg.name]  # so we can check for surplus arguments.

            out_soq = Soquet(binst, reg.name)
            self._available.add(out_soq)

            self._cxns.append(Connection(in_soq, out_soq))
            out_soqs.append(out_soq)

        if in_soqs:
            raise BloqBuilderError(
                f"{bloq} does not accept input Soquets: {in_soqs.keys()}."
            ) from None

        return tuple(out_soqs)

    def finalize(self, **final_soqs: Soquet) -> CompositeBloq:
        """Finish building a CompositeBloq and return the immutable CompositeBloq.

        This method is similar to calling `add()` but instead of adding a new Bloq,
        it validates the final "dangling" soquets that serve as the outputs for
        the composite bloq as a whole.

        This method is called at the end of `Bloq.decompose_bloq`. Users overriding
        `Bloq.build_composite_bloq` should not call this method.

        Args:
            **final_soqs: Keyword arguments mapping the composite bloq's register names to
                final`Soquet`s, e.g. the output soquets from a prior, final operation.
        """
        for reg in self._parent_regs:
            try:
                in_soq = final_soqs[reg.name]
            except KeyError:
                raise BloqBuilderError(
                    f"Finalizing the build requires a final Soquet named `{reg.name}`."
                ) from None

            try:
                self._available.remove(in_soq)
            except KeyError:
                raise BloqBuilderError(
                    f"{in_soq} is not an available final Soquet for {reg}."
                ) from None

            del final_soqs[reg.name]  # so we can check for surplus arguments.

            out_soq = Soquet(RightDangle, reg.name)
            self._cxns.append(Connection(in_soq, out_soq))

        if final_soqs:
            raise BloqBuilderError(
                f"Finalizing the build does not accept final Soquets: {final_soqs.keys()}."
            ) from None

        if self._available:
            raise BloqBuilderError(
                f"During finalization, {self._available} Soquets were not used."
            ) from None

        return CompositeBloq(cxns=self._cxns, registers=self._parent_regs)
