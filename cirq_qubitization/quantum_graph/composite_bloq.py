from collections import defaultdict
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cirq
import networkx as nx
import numpy as np
from attrs import frozen
from numpy.typing import ArrayLike, NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

SoquetT = Union[Soquet, NDArray[Soquet]]


class CompositeBloq(Bloq):
    """A container type implementing the `Bloq` interface.

    Args:
        cxns: A sequence of `Connection` encoding the quantum compute graph.
        registers: The registers defining the inputs and outputs of this Bloq. This
            should correspond to the dangling `Soquets` in the `cxns`.
    """

    def __init__(self, cxns: Sequence[Connection], registers: FancyRegisters):
        self._cxns = tuple(cxns)
        self._registers = registers

    @property
    def registers(self) -> FancyRegisters:
        return self._registers

    @property
    def connections(self) -> Tuple[Connection, ...]:
        return self._cxns

    def internal_connections(self) -> Tuple[Connection, ...]:
        return tuple(
            cxn
            for cxn in self._cxns
            if not isinstance(cxn.left.binst, DanglingT)
            and not isinstance(cxn.right.binst, DanglingT)
        )

    @cached_property
    def bloq_instances(self) -> Set[BloqInstance]:
        """The set of BloqInstances making up the nodes of the graph."""
        return {
            soq.binst
            for cxn in self._cxns
            for soq in [cxn.left, cxn.right]
            if not isinstance(soq.binst, DanglingT)
        }

    @cached_property
    def all_soquets(self) -> Set[Soquet]:
        soquets = {cxn.left for cxn in self._cxns if not isinstance(cxn.left, DanglingT)}
        soquets |= {cxn.right for cxn in self._cxns if not isinstance(cxn.right, DanglingT)}
        return soquets

    def to_cirq_circuit(self, **quregs: NDArray[cirq.Qid]) -> cirq.Circuit:
        """Convert this CompositeBloq to a `cirq.Circuit`.

        Args:
            quregs: These keyword arguments map from register name to a sequence of `cirq.Qid`.
                Cirq operations operate on individual qubit objects.
                Consider using `**self.registers.get_named_qubits()` for this argument.
        """
        # First, convert register names to registers.
        quregs = {self.registers.get_left(reg_name): qubits for reg_name, qubits in quregs.items()}
        return _cbloq_to_cirq_circuit(quregs, self.connections)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("Come back later.")


def _create_binst_graph(cxns: Iterable[Connection]) -> nx.DiGraph:
    """Helper function to create a NetworkX graph so we can topologically visit BloqInstances.

    `CompositeBloq` defines a directed acyclic graph, so we can iterate in (time) order.
    Here, we make two changes to our view of the graph:
        1. Our nodes are now BloqInstances because they are the objects to time-order. Soquet
           connections are added as edge attributes.
        2. We use networkx so we can use their algorithms for topological sorting.
    """
    binst_graph = nx.DiGraph()
    for cxn in cxns:
        binst_edge = (cxn.left.binst, cxn.right.binst)
        if binst_edge in binst_graph.edges:
            binst_graph.edges[binst_edge]['cxns'].append(cxn)
        else:
            binst_graph.add_edge(*binst_edge, cxns=[cxn])
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
    if isinstance(binst, DanglingT):
        return None

    # Track inter-Bloq name changes
    for pred in binst_graph.predecessors(binst):
        for cxn in binst_graph.edges[pred, binst]['cxns']:
            soqmap[cxn.right] = soqmap[cxn.left]
            del soqmap[cxn.left]

    bloq = binst.bloq

    # Pull out the qubits from soqmap into qumap which has string keys.
    # This implicitly joins things with the same name.
    quregs: Dict[str, List[cirq.Qid]] = defaultdict(list)
    for reg in bloq.registers.lefts():
        for li in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=li)
            quregs[reg.name].extend(soqmap[soq])
            del soqmap[soq]

    op = bloq.on_registers(**quregs)

    # We pluck things back out from their collapsed by-name qumap into soqmap
    # This does implicit splitting.
    for reg in bloq.registers.rights():
        qarr = np.asarray(quregs[reg.name])
        for ri in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=ri)
            qs = qarr[ri]
            if isinstance(qs, np.ndarray):
                qs = qs.tolist()
            else:
                qs = [qs]
            soqmap[soq] = qs

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[FancyRegister, NDArray[cirq.Qid]], cxns: Sequence[Connection]
) -> cirq.Circuit:
    """Transform CompositeBloq components into a `cirq.Circuit`.

    Args:
        quregs: Assignment from each register to a sequence of `cirq.Qid` for the conversion
            to a `cirq.Circuit`.
        cxns: A sequence of `Connection` objects that define the quantum compute graph.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    # Make a graph where we just connect binsts but note in the edges what the mappings are.
    binst_graph = _create_binst_graph(cxns)

    # A mapping of soquet to qubits that we update as operations are appended to the circuit.
    soqmap = {}
    for reg in quregs.keys():
        qarr = np.asarray(quregs[reg])
        for ii in reg.wire_idxs():
            soqmap[Soquet(LeftDangle, reg, idx=ii)] = qarr[ii]

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


def _initialize_soquets(regs: FancyRegisters) -> Tuple[Dict[str, SoquetT], Set[Soquet]]:
    """Initialize input Soquets from left registers for bookkeeping in `CompositeBloqBuilder`.

    Returns:
        initial_soqs: A mapping from register name to a Soquet or Soquets. For multi-dimensional
            registers, the value will be an array of indexed Soquets. For 0-dimensional (normal)
            registers, the value will be a `Soquet` object.
        available: A flat set of all the `Soquet`s. During initialization, all Soquets are
            available to be consumed. `CompositeBloqBuilder` will keep the set of available
            Soquets up-to-date.
    """
    available: Set[Soquet] = set()
    initial_soqs: Dict[str, SoquetT] = {}
    soqs: SoquetT
    for reg in regs.lefts():
        if reg.wireshape:
            soqs = np.empty(reg.wireshape, dtype=object)
            for ri in reg.wire_idxs():
                soq = Soquet(LeftDangle, reg, idx=ri)
                soqs[ri] = soq
                available.add(soq)
        else:
            # Annoyingly, this must be a special case.
            # Otherwise, x[i] = thing will nest *array* objects because our ndarray's type is
            # 'object'. This wouldn't happen--for example--with an integer array.
            soqs = Soquet(LeftDangle, reg)
            available.add(soqs)

        initial_soqs[reg.name] = soqs
    return initial_soqs, available


@frozen
class CBBAddResult:
    binst: BloqInstance
    soquets: Tuple[Soquet, ...]


class CompositeBloqBuilder:
    """A builder class for constructing a `CompositeBloq`.

    Users should not instantiate a CompositeBloqBuilder directly. To build a composite bloq,
    override `Bloq.build_composite_bloq`. A properly-initialized builder instance will be
    provided as the first argument.

    Args:
        parent_regs: The `Registers` argument for the parent bloq.
    """

    def __init__(self, parent_regs: FancyRegisters):
        # To be appended to:
        self._cxns: List[Connection] = []

        # Initialize our BloqInstance counter
        self._i = 0

        self._parent_regs = parent_regs

        # Bookkeeping for linear types; Soquets must be used exactly once.
        self._initial_soquets, self._available = _initialize_soquets(parent_regs)

    def initial_soquets(self) -> Dict[str, SoquetT]:
        """Input soquets (by name) to start building a quantum compute graph."""
        return self._initial_soquets

    def _new_binst(self, bloq: Bloq) -> BloqInstance:
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def advanced_add(self, bloq: Bloq, **in_soqs: ArrayLike) -> CBBAddResult:
        """Add a new bloq instance to the compute graph.

        Args:
            bloq: The bloq representing the operation to add.
            **in_soqs: Keyword arguments mapping the new bloq's register names to input
                `Soquet`s or an array thereof. This is likely the output soquets from a prior
                operation.

        Returns:
            A `Soquet` or an array thereof for each output register ordered according to
                `bloq.registers`.
                Note: Analogous to a Python function call using kwargs and multiple return values,
                the ordering is irrespective of the order of `in_soqs` that have been passed in
                and depends only on the convention of the bloq's registers.
        """
        binst = self._new_binst(bloq)

        for reg in bloq.registers.lefts():
            try:
                # if we want fancy indexing (which we do), we need numpy
                # this also supports length-zero indexing natively, which is good too.
                in_soq = np.asarray(in_soqs[reg.name])
            except KeyError:
                raise BloqBuilderError(
                    f"{bloq} requires an input Soquet named `{reg.name}`."
                ) from None

            del in_soqs[reg.name]  # so we can check for surplus arguments.

            for li in reg.wire_idxs():
                idxed_soq = in_soq[li]
                assert isinstance(idxed_soq, Soquet), idxed_soq
                try:
                    self._available.remove(idxed_soq)
                except KeyError:
                    raise BloqBuilderError(
                        f"{idxed_soq} is not an available input Soquet for {reg}."
                    ) from None
                cxn = Connection(idxed_soq, Soquet(binst, reg, idx=li))
                self._cxns.append(cxn)

        if in_soqs:
            raise BloqBuilderError(
                f"{bloq} does not accept input Soquets: {in_soqs.keys()}."
            ) from None

        out_soqs: List[SoquetT] = []
        out: SoquetT
        for reg in bloq.registers.rights():
            if reg.wireshape:
                out = np.empty(reg.wireshape, dtype=object)
                for ri in reg.wire_idxs():
                    out_soq = Soquet(binst, reg, idx=ri)
                    out[ri] = out_soq
                    self._available.add(out_soq)
            else:
                # Annoyingly, the 0-dim case must be handled seprately.
                # Otherwise, x[i] = thing will nest *array* objects.
                out = Soquet(binst, reg)
                self._available.add(out)

            out_soqs.append(out)

        return CBBAddResult(binst=binst, soquets=tuple(out_soqs))

    def add(self, bloq: Bloq, **in_soqs: ArrayLike) -> Tuple[SoquetT, ...]:
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
        return self.advanced_add(bloq, **in_soqs).soquets

    def finalize(self, **final_soqs: SoquetT) -> CompositeBloq:
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
        for reg in self._parent_regs.rights():
            try:
                in_soq = np.asarray(final_soqs[reg.name])
            except KeyError:
                raise BloqBuilderError(
                    f"Finalizing the build requires a final Soquet named `{reg.name}`."
                ) from None

            del final_soqs[reg.name]  # so we can check for surplus arguments.

            for li in reg.wire_idxs():
                idxed_soq = in_soq[li]
                assert isinstance(idxed_soq, Soquet)
                try:
                    self._available.remove(idxed_soq)
                except KeyError:
                    raise BloqBuilderError(
                        f"{in_soq} is not an available final Soquet for {reg}."
                    ) from None
                self._cxns.append(Connection(idxed_soq, Soquet(RightDangle, reg, idx=li)))

        if final_soqs:
            raise BloqBuilderError(
                f"Finalizing the build does not accept final Soquets: {final_soqs.keys()}."
            ) from None

        if self._available:
            raise BloqBuilderError(
                f"During finalization, {self._available} Soquets were not used."
            ) from None

        return CompositeBloq(cxns=self._cxns, registers=self._parent_regs)


def my_rep(binst: BloqInstance) -> Optional[CompositeBloq]:
    from cirq_qubitization.quantum_graph.examples import MultiAnd

    if isinstance(binst.bloq, MultiAnd):
        return binst.bloq.decompose_bloq()

    return None


def _get_in_soqs(
    binst: BloqInstance,
    binst_graph: nx.DiGraph,
    parent_soqs: Dict[str, Soquet],
    binst_replacement: Dict[BloqInstance, BloqInstance],
) -> Dict[str, Soquet]:

    soqs: Dict[str, Soquet] = {}
    for pred in binst_graph.predecessors(binst):
        cxn: Connection
        for cxn in binst_graph.edges[pred, binst]['cxns']:
            if cxn.left.binst is LeftDangle and cxn.left.reg.name in parent_soqs:
                left = parent_soqs[cxn.left.reg.name]
            else:
                left = cxn.left

            if left.binst in binst_replacement:
                left = attrs.evolve(left, binst=binst_replacement[left.binst])

            soqs[cxn.right.reg.name] = left

    return soqs


def _process_binst_replace(
    bb: CompositeBloqBuilder,
    binst: BloqInstance,
    binst_graph: nx.DiGraph,
    parent_soqs: Dict[str, Soquet],
    binst_replacement: Dict[BloqInstance, BloqInstance],
):
    if isinstance(binst, DanglingT):
        return

    soqs = _get_in_soqs(binst, binst_graph, parent_soqs, binst_replacement)

    cbloq = my_rep(binst)
    if cbloq is None:
        print('adding', binst)
        res = bb.advanced_add(binst.bloq, **soqs)
        binst_replacement[binst] = res.binst
        return

    assert isinstance(cbloq, CompositeBloq)
    print("Recursing instead")
    _replace(cbloq, bb, soqs)
    print("pop")


def _replace(cbloq: CompositeBloq, bb: CompositeBloqBuilder, soqs: Dict[str, Soquet]):
    binst_graph = _create_binst_graph(cbloq.connections)
    binst_replacement = {}
    sorted_binsts = list(nx.topological_sort(binst_graph))
    for binst in sorted_binsts:
        # Modifies graph
        _process_binst_replace(bb, binst, binst_graph, soqs, binst_replacement)
    return binst_graph, soqs, binst_replacement


def replace(cbloq: CompositeBloq) -> CompositeBloq:
    bb = CompositeBloqBuilder(cbloq.registers)
    binst_graph, parent_soqs, binst_replacement = _replace(cbloq, bb, {})

    soqs = _get_in_soqs(RightDangle, binst_graph, parent_soqs, binst_replacement)
    return bb.finalize(**soqs)
