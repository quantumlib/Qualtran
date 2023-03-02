from collections import defaultdict
from functools import cached_property
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    Union,
)

import cirq
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
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
    def all_soquets(self) -> FrozenSet[Soquet]:
        soquets = {cxn.left for cxn in self._cxns}
        soquets |= {cxn.right for cxn in self._cxns}
        return frozenset(soquets)

    @cached_property
    def _binst_graph(self) -> nx.DiGraph:
        """Get a cached version of this composite bloq's BloqInstance graph.

        The BloqInstance graph (or binst_graph) records edges between bloq instances
        and stores the `Connection` (i.e. Soquet-Soquet) information on an edge attribute
        named `cxns`.

        NetworkX graphs are mutable. We require that any uses of this private property
        do not mutate the graph. It is cached for performance reasons. Use g.copy() to
        get a copy.
        """
        return _create_binst_graph(self.connections)

    def to_cirq_circuit(self, **quregs: NDArray[cirq.Qid]) -> cirq.Circuit:
        """Convert this CompositeBloq to a `cirq.Circuit`.

        Args:
            quregs: These keyword arguments map from register name to a sequence of `cirq.Qid`.
                Cirq operations operate on individual qubit objects.
                Consider using `**self.registers.get_named_qubits()` for this argument.
        """
        # First, convert register names to registers.
        quregs = {self.registers.get_left(reg_name): qubits for reg_name, qubits in quregs.items()}
        return _cbloq_to_cirq_circuit(quregs, self._binst_graph)

    def as_composite_bloq(self) -> 'CompositeBloq':
        """This override just returns the present composite bloq."""
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise NotImplementedError("Come back later.")

    def iter_bloqnections(
        self,
    ) -> Iterator[Tuple[BloqInstance, List[Connection], List[Connection]]]:
        """Iterate over Bloqs and their connections in topological order.

        Yields:
            A bloq instance, its predecessor connections, and its successor connections. The
            bloq instances are yielded in a topologically-sorted order. The predecessor
            and successor connections are lists of `Connection` objects feeding into or out of
            (respectively) the binst. Dangling nodes are not included as the binst (but
            connections to dangling nodes are included in predecessors and successors).
            Every connection that does not involve a dangling node will appear twice: once as
            a predecessor and again as a successor.
        """
        g = self._binst_graph
        for binst in nx.topological_sort(g):
            if isinstance(binst, DanglingT):
                continue
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=g)
            yield binst, pred_cxns, succ_cxns

    @staticmethod
    def _debug_binst(g: nx.DiGraph, binst: BloqInstance) -> List[str]:
        """Helper method used in `debug_text`"""
        lines = [f'{binst}']
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=g)
        for pred_cxn in pred_cxns:
            lines.append(
                f'  {pred_cxn.left.binst}.{pred_cxn.left.pretty()} -> {pred_cxn.right.pretty()}'
            )
        for succ_cxn in succ_cxns:
            lines.append(
                f'  {succ_cxn.left.pretty()} -> {succ_cxn.right.binst}.{succ_cxn.right.pretty()}'
            )
        return lines

    def debug_text(self) -> str:
        """Print connection information to assist in debugging.

        The output will be a topologically sorted list of BloqInstances with each
        topological generation separated by a horizontal line. Each bloq instance is followed
        by a list of its incoming and outgoing connections. Note that all non-dangling
        connections are represented twice: once as the output of a binst and again as the input
        to a subsequent binst.
        """
        g = self._binst_graph
        gen_texts = []
        for gen in nx.topological_generations(g):
            gen_lines = []
            for binst in gen:
                if isinstance(binst, DanglingT):
                    continue

                gen_lines.extend(self._debug_binst(g, binst))

            if gen_lines:
                gen_texts.append('\n'.join(gen_lines))

        delimited_gens = ('\n' + '-' * 20 + '\n').join(gen_texts)
        return delimited_gens


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


def _binst_to_cxns(
    binst: BloqInstance, binst_graph: nx.DiGraph
) -> Tuple[List[Connection], List[Connection]]:
    """Helper method to extract all predecessor and successor Connections for a binst."""
    pred_cxns: List[Connection] = []
    for pred in binst_graph.pred[binst]:
        pred_cxns.extend(binst_graph.edges[pred, binst]['cxns'])

    succ_cxns: List[Connection] = []
    for succ in binst_graph.succ[binst]:
        succ_cxns.extend(binst_graph.edges[binst, succ]['cxns'])

    return pred_cxns, succ_cxns


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

    pred_cxns, _ = _binst_to_cxns(binst, binst_graph)

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
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
    quregs: Dict[FancyRegister, NDArray[cirq.Qid]], binst_graph: nx.DiGraph
) -> cirq.Circuit:
    """Transform CompositeBloq components into a `cirq.Circuit`.

    Args:
        quregs: Assignment from each register to a sequence of `cirq.Qid` for the conversion
            to a `cirq.Circuit`.
        binst_graph: A graph connecting bloq instances with edge attributes containing the
            full list of `Connection`s, as returned by `CompositeBloq._get_binst_graph()`.
            This function does not mutate `binst_graph`.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
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


def _initialize_soquets(reg: FancyRegister, available: Set[Soquet]) -> SoquetT:
    """Initialize input Soquets from left registers.

    Args:
        reg: The register
        available: A set that we will add each individual soquet to (used for bookkeeping in
            `CompositeBloqBuilder`).

    Returns:
        A Soquet or Soquets. For multi-dimensional
        registers, the value will be an array of indexed Soquets. For 0-dimensional (normal)
        registers, the value will be a `Soquet` object.
    """
    if reg.wireshape:
        soqs = np.empty(reg.wireshape, dtype=object)
        for ri in reg.wire_idxs():
            soq = Soquet(LeftDangle, reg, idx=ri)
            soqs[ri] = soq
            available.add(soq)
        return soqs

    # Annoyingly, this must be a special case.
    # Otherwise, x[i] = thing will nest *array* objects because our ndarray's type is
    # 'object'. This wouldn't happen--for example--with an integer array.
    soqs = Soquet(LeftDangle, reg)
    available.add(soqs)
    return soqs


class CompositeBloqBuilder:
    """A builder class for constructing a `CompositeBloq`.

    Users may instantiate this class directly or use its methods by
    overriding `Bloq.build_composite_bloq`.

    When overriding `build_composite_bloq`, the Bloq class will ensure that the bloq under
    construction has the correct registers: namely, those of the decomposed bloq and parent
    bloq are the same. This affords some additional error checking (see `bb.finalize_strict()`).
    Initial soquets are passed as **kwargs (by register name) to the `build_composite_bloq` method.
    See `from_registers` for more details.

    To use this class directly, you must call `add_register` to set up the composite bloq's
    registers. When adding a LEFT or THRU register, the method will return soquets to be
    used when adding more bloqs. Adding a THRU or RIGHT register will enable more checks during
    `finalize()`. These checks can be made manditory with `finalize_strict()`.
    """

    def __init__(self):
        # To be appended to:
        self._cxns: List[Connection] = []
        self._regs: List[FancyRegister] = []

        # Initialize our BloqInstance counter
        self._i = 0

        # Bookkeeping for linear types; Soquets must be used exactly once.
        self._available: Set[Soquet] = set()

        # Whether we can call `add_register` and do non-strict `finalize`().
        self._add_register_allowed = True

    @overload
    def add_register(self, reg: FancyRegister, bitsize: None = None) -> Union[None, SoquetT]:
        ...

    @overload
    def add_register(self, reg: str, bitsize: int) -> SoquetT:
        ...

    def add_register(
        self, reg: Union[str, FancyRegister], bitsize: Optional[int] = None
    ) -> Union[None, SoquetT]:
        """Add a new register to the composite bloq being built.

        If this bloq builder was constructed with `from_registers`, this operation is not allowed.

        Args:
            reg: Either the register or a register name. If this is a register, then `bitsize`
                must also be provided and a default THRU register will be added.
            bitsize: If `reg` is a register name, this is the bitsize for the added register.
                Otherwise, this must not be provided.

        Returns:
            If `reg` is a LEFT or THRU register, return the soquet(s) corresponding to the
            initial, left-dangling soquets for the register. Otherwise, this is a RIGHT register
            and will be used for error checking in `finalize()` and nothing is returned.
        """
        if not self._add_register_allowed:
            raise ValueError(
                "This BloqBuilder was constructed from pre-specified registers. "
                "Ad hoc addition of more registers is not allowed."
            )

        if isinstance(reg, FancyRegister):
            if bitsize is not None:
                raise ValueError("`bitsize` must not be specified if `reg` is a Register.")
        else:
            if not isinstance(reg, str):
                raise ValueError("`reg` must be a string register name if not a Register.")
            if not isinstance(bitsize, int):
                raise ValueError(
                    "`bitsize` must be specified and must be an "
                    "integer if `reg` is a register name."
                )
            reg = FancyRegister(name=reg, bitsize=bitsize)

        self._regs.append(reg)
        if reg.side & Side.LEFT:
            return _initialize_soquets(reg, available=self._available)
        return None

    @classmethod
    def from_registers(cls, parent_regs: FancyRegisters):
        """Construct a CompositeBloqBuilder with pre-specified registers.

        This is safer if e.g. you're decomposing an existing Bloq and need the registers
        to match. This constructor is used by `Bloq.decompose_bloq()`.

        Using this constructor disables `add_register` and makes all `finalize` calls strict.
        """
        bb = cls()

        initial_soqs: Dict[str, SoquetT] = {}
        for reg in parent_regs:
            if reg.side & Side.LEFT:
                initial_soqs[reg.name] = bb.add_register(reg)
            else:
                bb.add_register(reg)

        return bb, initial_soqs

    def _new_binst(self, bloq: Bloq) -> BloqInstance:
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def add(self, bloq: Bloq, **in_soqs: SoquetT) -> Tuple[SoquetT, ...]:
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

        return tuple(out_soqs)

    def finalize(self, **final_soqs: SoquetT) -> CompositeBloq:
        """Finish building a CompositeBloq and return the immutable CompositeBloq.

        This method is similar to calling `add()` but instead of adding a new Bloq,
        it configures the final "dangling" soquets that serve as the outputs for
        the composite bloq as a whole.

        Args:
            **final_soqs: Keyword arguments mapping the composite bloq's register names to
                final`Soquet`s, e.g. the output soquets from a prior, final operation.
        """
        if not self._add_register_allowed:
            return self.finalize_strict(**final_soqs)

        # If items from `final_soqs` don't already exist in `_regs`, add RIGHT registers
        # for them. Then call `finalize_strict` where the actual dangling connections are added.

        def _infer_reg(name: str, soq: SoquetT) -> FancyRegister:
            """Go from Soquet -> register, but use a specific name for the register."""
            if isinstance(soq, Soquet):
                return FancyRegister(name=name, bitsize=soq.reg.bitsize, side=Side.RIGHT)

            # Get info from 0th soquet in an ndarray.
            return FancyRegister(
                name=name,
                bitsize=soq.reshape(-1)[0].reg.bitsize,
                wireshape=soq.shape,
                side=Side.RIGHT,
            )

        right_reg_names = [reg.name for reg in self._regs if reg.side & Side.RIGHT]
        for name, soq in final_soqs.items():
            if name not in right_reg_names:
                self._regs.append(_infer_reg(name, soq))

        return self.finalize_strict(**final_soqs)

    def finalize_strict(self, **final_soqs: SoquetT) -> CompositeBloq:
        """Finish building a CompositeBloq and return the immutable CompositeBloq.

        In contrast to `finalize()`, this method verifies the provided final_soqs
        against our list of RIGHT (and THRU) registers.

        Args:
            **final_soqs: Keyword arguments mapping the composite bloq's register names to
                final`Soquet`s, e.g. the output soquets from a prior, final operation.
        """
        registers = FancyRegisters(self._regs)
        for reg in registers.rights():
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

        return CompositeBloq(cxns=self._cxns, registers=registers)

    def allocate(self, n: int = 1) -> Soquet:
        from cirq_qubitization.quantum_graph.util_bloqs import Allocate

        (out_soq,) = self.add(Allocate(n=n))
        return out_soq

    def free(self, soq: Soquet) -> None:
        from cirq_qubitization.quantum_graph.util_bloqs import Free

        if not isinstance(soq, Soquet):
            raise ValueError("`free` expects a single Soquet to free.")

        self.add(Free(n=soq.reg.bitsize), free=soq)

    def split(self, soq: Soquet) -> SoquetT:
        """Add a Split bloq to split up a register."""
        from cirq_qubitization.quantum_graph.util_bloqs import Split

        if not isinstance(soq, Soquet):
            raise ValueError("`split` expects a single Soquet to split.")

        (out_soqs,) = self.add(Split(n=soq.reg.bitsize), split=soq)
        return out_soqs

    def join(self, soqs: NDArray[Soquet]) -> Soquet:
        from cirq_qubitization.quantum_graph.util_bloqs import Join

        try:
            (n,) = soqs.shape
        except AttributeError:
            raise ValueError("`join` expects a 1-d array of input soquets to join.") from None

        if not all(soq.reg.bitsize == 1 for soq in soqs):
            raise ValueError("`join` can only join equal-bitsized soquets, currently only size 1.")

        (out_soq,) = self.add(Join(n=n), join=soqs)
        return out_soq
