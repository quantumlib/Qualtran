#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Classes for building and manipulating `CompositeBloq`."""
from functools import cached_property
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import attrs
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .bloq import Bloq, DecomposeTypeError
from .data_types import QAny, QBit, QDType
from .quantum_graph import BloqInstance, Connection, DanglingT, LeftDangle, RightDangle, Soquet
from .registers import Register, Side, Signature

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregInT, CirqQuregT
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


SoquetT = Union[Soquet, NDArray[Soquet]]
"""A `Soquet` or array of soquets."""

SoquetInT = Union[Soquet, NDArray[Soquet], Sequence[Soquet]]
"""A soquet or array-like of soquets.

This type alias is used for input argument to parts of the library that are more
permissive about the types they accept. Under-the-hood, such functions will
canonicalize and return `SoquetT`.
"""


@attrs.frozen
class CompositeBloq(Bloq):
    """A bloq defined by a collection of sub-bloqs and dataflows between them

    CompositeBloq represents a quantum subroutine as a dataflow compute graph. The
    specific native representation is a list of `Connection` objects (i.e. a list of
    graph edges). This container should be considered immutable. Additional views
    of the graph are provided by methods and properties.

    Users should generally use `BloqBuilder` to construct a composite bloq either
    directly or by overriding `Bloq.build_composite_bloq`.

    Throughout this library we will often use the variable name `cbloq` to refer to a
    composite bloq.

    Args:
        cxns: A sequence of `Connection` encoding the quantum compute graph.
        signature: The registers defining the inputs and outputs of this Bloq. This
            should correspond to the dangling `Soquets` in the `cxns`.
    """

    connections: Tuple[Connection, ...] = attrs.field(converter=tuple)
    signature: Signature
    bloq_instances: FrozenSet[BloqInstance] = attrs.field(converter=frozenset)

    @bloq_instances.default
    def _default_bloq_instances(self):
        return {
            soq.binst
            for cxn in self.connections
            for soq in [cxn.left, cxn.right]
            if not isinstance(soq.binst, DanglingT)
        }

    @cached_property
    def all_soquets(self) -> FrozenSet[Soquet]:
        """A set of all `Soquet`s present in the compute graph."""
        soquets = {cxn.left for cxn in self.connections}
        soquets |= {cxn.right for cxn in self.connections}
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
        return _create_binst_graph(self.connections, self.bloq_instances)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        """Return a cirq.CircuitOperation containing a cirq-exported version of this cbloq."""
        import cirq

        circuit, out_quregs = self.to_cirq_circuit(qubit_manager=qubit_manager, **cirq_quregs)
        return cirq.CircuitOperation(circuit), out_quregs

    def to_cirq_circuit(
        self, qubit_manager: Optional['cirq.QubitManager'] = None, **cirq_quregs: 'CirqQuregInT'
    ) -> Tuple['cirq.FrozenCircuit', Dict[str, 'CirqQuregT']]:
        """Convert this CompositeBloq to a `cirq.Circuit`.

        Args:
            qubit_manager: A `cirq.QubitManager` to allocate new qubits.
            **cirq_quregs: Mapping from left register names to Cirq qubit arrays.

        Returns:
            circuit: The cirq.FrozenCircuit version of this composite bloq.
            cirq_quregs: The output mapping from right register names to Cirq qubit arrays.
        """
        import cirq

        from qualtran.cirq_interop._bloq_to_cirq import _cbloq_to_cirq_circuit

        if qubit_manager is None:
            qubit_manager = cirq.ops.SimpleQubitManager()

        return _cbloq_to_cirq_circuit(
            self.signature, cirq_quregs, self._binst_graph, qubit_manager=qubit_manager
        )

    @classmethod
    def from_cirq_circuit(cls, circuit: 'cirq.Circuit') -> 'CompositeBloq':
        """Construct a composite bloq from a Cirq circuit.

        Each `cirq.Operation` will be wrapped into a `CirqGate` wrapper bloq. The
        resultant composite bloq will represent a unitary with one thru-register
        named "qubits" of shape `(n_qubits,)`.
        """
        from qualtran.cirq_interop import cirq_optree_to_cbloq

        return cirq_optree_to_cbloq(circuit)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        """Support classical data by recursing into the composite bloq."""
        from qualtran.simulation.classical_sim import call_cbloq_classically

        out_vals, _ = call_cbloq_classically(self.signature, vals, self._binst_graph)
        return out_vals

    def call_classically(self, **vals: 'ClassicalValT') -> Tuple['ClassicalValT', ...]:
        """Support classical data by recursing into the composite bloq."""
        from qualtran.simulation.classical_sim import call_cbloq_classically

        out_vals, _ = call_cbloq_classically(self.signature, vals, self._binst_graph)
        return tuple(out_vals[reg.name] for reg in self.signature.rights())

    def as_composite_bloq(self) -> 'CompositeBloq':
        """This override just returns the present composite bloq."""
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError("CompositeBloq cannot be decomposed.")

    def build_call_graph(self, ssa: Optional['SympySymbolAllocator']) -> Set['BloqCountT']:
        """Return the bloq counts by counting up all the subbloqs."""
        from qualtran.resource_counting import build_cbloq_call_graph

        return build_cbloq_call_graph(self)

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

    def iter_bloqsoqs(
        self,
    ) -> Iterator[Tuple[BloqInstance, Dict[str, SoquetT], Tuple[SoquetT, ...]]]:
        """Iterate over bloq instances and their input soquets.

        This method is helpful for "adding from" this existing composite bloq. You must
        use `map_soqs` to map this cbloq's soquets to the correct ones for the
        new bloq.

        >>> bb, _ = BloqBuilder.from_signature(self.signature)
        >>> soq_map: List[Tuple[SoquetT, SoquetT]] = []
        >>> for binst, in_soqs, old_out_soqs in self.iter_bloqsoqs():
        >>>    in_soqs = bb.map_soqs(in_soqs, soq_map)
        >>>    new_out_soqs = bb.add_t(binst.bloq, **in_soqs)
        >>>    soq_map.extend(zip(old_out_soqs, new_out_soqs))
        >>> return bb.finalize(**bb.map_soqs(self.final_soqs(), soq_map))

        Yields:
            binst: The current bloq instance
            in_soqs: A dictionary mapping the binst's register names to predecessor soquets.
                This is suitable for `bb.add(binst.bloq, **in_soqs)`
            out_soqs: A tuple of the output soquets of `binst`. This can be used to update
                the mapping from this cbloq's soquets to a modified copy, see the example code.
        """

        for binst, preds, succs in self.iter_bloqnections():
            in_soqs = _cxn_to_soq_dict(
                binst.bloq.signature.lefts(),
                preds,
                get_me=lambda x: x.right,
                get_assign=lambda x: x.left,
            )
            out_soqs = tuple(_reg_to_soq(binst, reg) for reg in binst.bloq.signature.rights())
            yield binst, in_soqs, out_soqs

    def final_soqs(self) -> Dict[str, SoquetT]:
        """Return the final output soquets.

        This method is helpful for finalizing an "add from" operation, see `iter_bloqsoqs`.
        """
        if RightDangle not in self._binst_graph:
            return {}
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=self._binst_graph)
        return _cxn_to_soq_dict(
            self.signature.rights(),
            final_preds,
            get_me=lambda x: x.right,
            get_assign=lambda x: x.left,
        )

    def copy(self) -> 'CompositeBloq':
        """Create a copy of this composite bloq by re-building it."""
        bb, _ = BloqBuilder.from_signature(self.signature)
        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        for binst, in_soqs, old_out_soqs in self.iter_bloqsoqs():
            in_soqs = _map_soqs(in_soqs, soq_map)
            new_out_soqs = bb.add_t(binst.bloq, **in_soqs)
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = _map_soqs(self.final_soqs(), soq_map)
        return bb.finalize(**fsoqs)

    def flatten_once(self, pred: Callable[[BloqInstance], bool]) -> 'CompositeBloq':
        """Decompose and flatten each subbloq that satisfies `pred`.

        This will only flatten "once". That is, we will go through the bloq instances
        contained in this composite bloq and (optionally) flatten each one but will not
        recursively flatten the results. For a recursive version see `flatten`.

        Args:
            pred: A predicate that takes a bloq instance and returns True if it should
                be decomposed and flattened or False if it should remain undecomposed.
                All bloqs for which this callable returns True must support decomposition.

        Returns:
            A new composite bloq where subbloqs matching `pred` have been decomposed and
            flattened.

        Raises:
            NotImplementedError: If `pred` returns True but the underlying bloq does not
                support `decompose_bloq()`.
            DidNotFlattenAnythingError: If none of the bloq instances satisfied `pred`.

        """
        bb, _ = BloqBuilder.from_signature(self.signature)

        # We take particular care during flattening to preserve the `binst.i` of bloq instances
        # that are not flattened. We do this by initializing the bloq builder's `i` counter
        # to one greater than the existing maximum value, so all calls to `add_from` will result
        # in new, higher `binst.i` values.
        # pylint: disable=protected-access
        bb._i = max(binst.i for binst in self.bloq_instances) + 1

        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        did_work = False
        for binst, in_soqs, old_out_soqs in self.iter_bloqsoqs():
            in_soqs = _map_soqs(in_soqs, soq_map)  # update `in_soqs` from old to new.

            if pred(binst):
                new_out_soqs = bb.add_from(binst.bloq.decompose_bloq(), **in_soqs)
                did_work = True
            else:
                # Since we took care to not re-use existing `binst.i` values for flattened
                # bloqs, it is safe to call `bb._add_binst` with the old `binst` (and in
                # particular with the old `binst.i`) to preserve the `binst.i` of unflattened
                # bloqs.
                # pylint: disable=protected-access
                new_out_soqs = tuple(soq for _, soq in bb._add_binst(binst, in_soqs=in_soqs))

            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        if not did_work:
            raise DidNotFlattenAnythingError()

        fsoqs = _map_soqs(self.final_soqs(), soq_map)
        return bb.finalize(**fsoqs)

    def adjoint(self) -> 'CompositeBloq':
        """Get a composite bloq which is the adjoint of this composite bloq.

        The adjoint of a composite bloq is another composite bloq where the order of
        operations is reversed and each subbloq is replaced with its adjoint.
        """
        from .adjoint import _adjoint_cbloq

        return _adjoint_cbloq(self)

    def flatten(
        self, pred: Callable[[BloqInstance], bool], max_depth: int = 1_000
    ) -> 'CompositeBloq':
        """Recursively decompose and flatten subbloqs until none satisfy `pred`.

        This will continue flattening the results of subbloq.decompose_bloq() until
        all bloqs which would satisfy `pred` have been flattened.

        Args:
            pred: A predicate that takes a bloq instance and returns True if it should
                be decomposed and flattened or False if it should remain undecomposed.
                All bloqs for which this callable returns True must support decomposition.
            max_depth: To avoid infinite recursion, give up after this many recursive steps.

        Returns:
            A new composite bloq where all recursive subbloqs matching `pred` have been
            decomposed and flattened.

        Raises:
            NotImplementedError: If `pred` returns True but the underlying bloq does not
                support `decompose_bloq()`.
        """
        cbloq = self
        for _ in range(max_depth):
            try:
                cbloq = cbloq.flatten_once(pred)
            except DidNotFlattenAnythingError:
                break
        else:
            raise ValueError("Max recursion depth exceeded in `flatten`.")

        return cbloq

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


def _create_binst_graph(
    cxns: Iterable[Connection], nodes: Iterable[BloqInstance] = ()
) -> nx.DiGraph:
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
    binst_graph.add_nodes_from(nodes)
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


def _cxn_to_soq_dict(
    regs: Iterable[Register],
    cxns: Iterable[Connection],
    get_me: Callable[[Connection], Soquet],
    get_assign: Callable[[Connection], Soquet],
) -> Dict[str, SoquetT]:
    """Helper function to get a dictionary of incoming or outgoing soquets from a connection.

    Args:
        regs: Left or right registers (used as a reference to initialize multidimensional
            registers correctly).
        cxns: Predecessor or successor connections from which we get the soquets of interest.
        get_me: A function that says which soquet is used to derive keys for the returned
            dictionary. Generally: if `cxns` is predecessor connections, this will return the
            `right` element of the connection and opposite of successor connections.
        get_assign: A function that says which soquet is used to dervice the values for the
            returned dictionary. Generally, this is the opposite side vs. `get_me`, but we
            do something fancier in `cbloq_to_quimb`.
    """
    soqdict: Dict[str, SoquetT] = {}

    # Initialize multi-dimensional dictionary values.
    for reg in regs:
        if reg.shape:
            soqdict[reg.name] = np.empty(reg.shape, dtype=object)

    # In the abstract: set `soqdict[me] = assign`. Specifically: use the register name as
    # keys and handle multi-dimensional registers.
    for cxn in cxns:
        me = get_me(cxn)
        assign = get_assign(cxn)

        if me.reg.shape:
            soqdict[me.reg.name][me.idx] = assign
        else:
            soqdict[me.reg.name] = assign

    return soqdict


def _get_dangling_soquets(signature: Signature, right=True) -> Dict[str, SoquetT]:
    """Get instantiated dangling soquets from a `Signature`.

    Args:
        signature: The registers
        right: If True, return soquets corresponding to right registers; otherwise left.

    Returns:
        all_soqs: A mapping from register name to a Soquet or Soquets. For multi-dimensional
            registers, the value will be an array of indexed Soquets. For 0-dimensional (normal)
            registers, the value will be a `Soquet` object.
    """

    if right:
        regs = signature.rights()
        dang = RightDangle
    else:
        regs = signature.lefts()
        dang = LeftDangle

    all_soqs: Dict[str, SoquetT] = {}
    soqs: SoquetT
    for reg in regs:
        all_soqs[reg.name] = _reg_to_soq(dang, reg)
    return all_soqs


def _flatten_soquet_collection(vals: Iterable[SoquetT]) -> List[Soquet]:
    """Flatten SoquetT into a flat list of Soquet.

    SoquetT is either a unit Soquet or an ndarray thereof.
    """
    soqvals = []
    for soq_or_arr in vals:
        if isinstance(soq_or_arr, Soquet):
            soqvals.append(soq_or_arr)
        else:
            soqvals.extend(soq_or_arr.reshape(-1))
    return soqvals


def _get_flat_dangling_soqs(signature: Signature, right: bool) -> List[Soquet]:
    """Flatten out the values of the soquet dictionaries from `_get_dangling_soquets`."""
    soqdict = _get_dangling_soquets(signature, right=right)
    return _flatten_soquet_collection(soqdict.values())


class BloqError(ValueError):
    """A value error raised when CompositeBloq conditions are violated.

    This error is raised during bloq building using `BloqBuilder`, which checks
    for the validity of registers and connections during the building process. This error is
    also raised by the validity assertion functions provided in this module.
    """


class DidNotFlattenAnythingError(ValueError):
    """An exception raised if `flatten_once()` did not find anything to flatten."""


class _IgnoreAvailable:
    """Used as an argument in `_reg_to_soq` to ignore any `available.add()` tracking."""

    def add(self, x: Hashable):
        pass


def _reg_to_soq(
    binst: Union[BloqInstance, DanglingT],
    reg: Register,
    available: Union[Set[Soquet], _IgnoreAvailable] = _IgnoreAvailable(),
) -> SoquetT:
    """Create the soquet or array of soquets for a register.

    Args:
        binst: The output soquet's bloq instance.
        reg: The register
        available: By default, don't track the soquets. If a set is provided, we will add
            each individual, indexed soquet to it. This is used for bookkeeping
            in `BloqBuilder`.

    Returns:
        A Soquet or Soquets. For multi-dimensional
        registers, the value will be an array of indexed Soquets. For 0-dimensional (normal)
        registers, the value will be a `Soquet` object.
    """
    if reg.shape:
        soqs = np.empty(reg.shape, dtype=object)
        for ri in reg.all_idxs():
            soq = Soquet(binst, reg, idx=ri)
            soqs[ri] = soq
            available.add(soq)
        return soqs

    # Annoyingly, this must be a special case.
    # Otherwise, x[i] = thing will nest *array* objects because our ndarray's type is
    # 'object'. This wouldn't happen--for example--with an integer array.
    soqs = Soquet(binst, reg)
    available.add(soqs)
    return soqs


def _process_soquets(
    registers: Iterable[Register],
    in_soqs: Dict[str, SoquetT],
    debug_str: str,
    func: Callable[[Soquet, Register, Tuple[int, ...]], None],
) -> None:
    """Process and validate `in_soqs` in the context of `registers`.

    This implements the following "outer loop" and calls
    `func(indexed_soquet, register, index)` for every `register` and
    corresponding soquets (from `in_soqs`) in the input.

    >>> for reg in registers:
    >>>     for idx in reg.all_idxs():
    >>>        func(in_soqs[reg.name][idx], reg, idx)

    We also perform input validation to make sure that the set of register names
    used as keys for `in_soqs` is identical to set of registers passed in `registers`.

    Args:
        registers: The registers to use for expected keys of `in_soqs`.
        in_soqs: A dictionary from register name to input soquets.
        debug_str: A string to use in error messages identifying what's being processed.
        func: A callable for operating on an individual (indexed) soquet. Must accept
            the incoming, indexed soquet as well as the register and (left-)index it
            has been mapped to.
    """

    for reg in registers:
        try:
            # if we want fancy indexing (which we do), we need numpy
            # this also supports length-zero indexing natively, which is good too.
            in_soq = np.asarray(in_soqs[reg.name])
        except KeyError:
            raise BloqError(f"{debug_str} requires a Soquet named `{reg.name}`.") from None

        del in_soqs[reg.name]  # so we can check for surplus arguments.

        for li in reg.all_idxs():
            idxed_soq = in_soq[li]
            assert isinstance(idxed_soq, Soquet), idxed_soq
            func(idxed_soq, reg, li)

    if in_soqs:
        raise BloqError(f"{debug_str} does not accept Soquets: {in_soqs.keys()}.") from None


def _map_soqs(
    soqs: Dict[str, SoquetT], soq_map: Iterable[Tuple[SoquetT, SoquetT]]
) -> Dict[str, SoquetT]:
    """Map `soqs` according to `soq_map`.

    See `CompositeBloq.iter_bloqsoqs` for example code. The public entry-point
    for this function is the `BloqBuilder.map_soqs` static function.

    Args:
        soqs: A soquet dictionary mapping register names to Soquets or arrays
            of Soquets. The values of this dictionary will be mapped.
        soq_map: An iterable of (old_soq, new_soq) tuples that inform how to
            perform the mapping. Note that this is a list of tuples (not a dictionary)
            because `old_soq` may be an unhashable numpy array of Soquet.

    Returns:
        A mapped version of `soqs`.
    """

    # First: flatten out any numpy arrays
    flat_soq_map: Dict[Soquet, Soquet] = {}
    for old_soqs, new_soqs in soq_map:
        if isinstance(old_soqs, Soquet):
            assert isinstance(new_soqs, Soquet), new_soqs
            flat_soq_map[old_soqs] = new_soqs
            continue

        assert isinstance(old_soqs, np.ndarray), old_soqs
        assert isinstance(new_soqs, np.ndarray), new_soqs
        assert old_soqs.shape == new_soqs.shape, (old_soqs.shape, new_soqs.shape)
        for o, n in zip(old_soqs.reshape(-1), new_soqs.reshape(-1)):
            flat_soq_map[o] = n

    # Then use vectorize to use the flat mapping.
    def _map_soq(soq: Soquet) -> Soquet:
        # Helper function to map an individual soquet.
        return flat_soq_map.get(soq, soq)

    # Use `vectorize` to call `_map_soq` on each element of the array.
    vmap = np.vectorize(_map_soq, otypes=[object])

    def _map_soqs(soqs: SoquetT) -> SoquetT:
        if isinstance(soqs, Soquet):
            return _map_soq(soqs)
        return vmap(soqs)

    return {name: _map_soqs(soqs) for name, soqs in soqs.items()}


class BloqBuilder:
    """A builder class for constructing a `CompositeBloq`.

    Users may instantiate this class directly or use its methods by
    overriding `Bloq.build_composite_bloq`.

    When overriding `build_composite_bloq`, the Bloq class will ensure that the bloq under
    construction has the correct registers: namely, those of the decomposed bloq and parent
    bloq are the same. This affords some additional error checking.
    Initial soquets are passed as **kwargs (by register name) to the `build_composite_bloq` method.

    When using this class directly, you must call `add_register` to set up the composite bloq's
    registers. When adding a LEFT or THRU register, the method will return soquets to be
    used when adding more bloqs. Adding a THRU or RIGHT register can enable more checks during
    `finalize()`.

    Args:
        add_registers_allowed: Whether we allow the addition of registers during bloq building.
        This affords some additional error checking if set to `False` but you must specify
        all registers ahead-of-time.
    """

    def __init__(self, add_registers_allowed: bool = True):
        # To be appended to:
        self._cxns: List[Connection] = []
        self._regs: List[Register] = []
        self._binsts: Set[BloqInstance] = set()

        # Initialize our BloqInstance counter
        self._i = 0

        # Bookkeeping for linear types; Soquets must be used exactly once.
        self._available: Set[Soquet] = set()

        # Whether we can call `add_register` and do non-strict `finalize()`.
        self.add_register_allowed = add_registers_allowed

    @overload
    def add_register(self, reg: Register, bitsize: None = None) -> Union[None, SoquetT]:
        ...

    @overload
    def add_register(self, reg: str, bitsize: int) -> SoquetT:
        ...

    def add_register(
        self, reg: Union[str, Register], bitsize: Optional[int] = None
    ) -> Union[None, SoquetT]:
        """Add a new register to the composite bloq being built.

        If this bloq builder was constructed with `add_registers_allowed=False`,
        this operation is not allowed.

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
        if not self.add_register_allowed:
            raise ValueError(
                "This BloqBuilder was constructed from pre-specified registers. "
                "Ad hoc addition of more registers is not allowed."
            )

        if isinstance(reg, Register):
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
            reg = Register(name=reg, dtype=QBit() if bitsize == 1 else QAny(bitsize))

        self._regs.append(reg)
        if reg.side & Side.LEFT:
            return _reg_to_soq(LeftDangle, reg, available=self._available)
        return None

    @classmethod
    def from_signature(
        cls, signature: Signature, add_registers_allowed: bool = False
    ) -> Tuple['BloqBuilder', Dict[str, SoquetT]]:
        """Construct a BloqBuilder with a pre-specified signature.

        This is safer if e.g. you're decomposing an existing Bloq and need the signatures
        to match. This constructor is used by `Bloq.decompose_bloq()`.
        """
        # Initial construction: allow register addition for the following loop.
        bb = cls(add_registers_allowed=True)

        initial_soqs: Dict[str, SoquetT] = {}
        for reg in signature:
            if reg.side & Side.LEFT:
                initial_soqs[reg.name] = bb.add_register(reg)
            else:
                bb.add_register(reg)

        # Now we can set it to the desired value.
        bb.add_register_allowed = add_registers_allowed

        return bb, initial_soqs

    @staticmethod
    def map_soqs(
        soqs: Dict[str, SoquetT], soq_map: Iterable[Tuple[SoquetT, SoquetT]]
    ) -> Dict[str, SoquetT]:
        """Map `soqs` according to `soq_map`.

        See `CompositeBloq.iter_bloqsoqs` for example code.

        Args:
            soqs: A soquet dictionary mapping register names to Soquets or arrays
                of Soquets. The values of this dictionary will be mapped.
            soq_map: An iterable of (old_soq, new_soq) tuples that inform how to
                perform the mapping. Note that this is a list of tuples (not a dictionary)
                because `old_soq` may be an unhashable numpy array of Soquet.

        Returns:
            A mapped version of `soqs`.
        """
        return _map_soqs(soqs=soqs, soq_map=soq_map)

    def _new_binst_i(self) -> int:
        i = self._i
        self._i += 1
        return i

    def _add_cxn(
        self, binst: BloqInstance, idxed_soq: Soquet, reg: Register, idx: Tuple[int, ...]
    ) -> None:
        """Helper function to be used as the base for the `func` argument of `_process_soquets`.

        This creates a connection between the provided input `idxed_soq` to the current binst's
        `(reg, idx)`.
        """
        try:
            self._available.remove(idxed_soq)
        except KeyError:
            bloq = binst if isinstance(binst, DanglingT) else binst.bloq
            raise BloqError(
                f"{idxed_soq} is not an available Soquet for `{bloq}.{reg.name}`."
            ) from None
        cxn = Connection(idxed_soq, Soquet(binst, reg, idx))
        self._cxns.append(cxn)

    def add_t(self, bloq: Bloq, **in_soqs: SoquetInT) -> Tuple[SoquetT, ...]:
        """Add a new bloq instance to the compute graph and always return a tuple of soquets.

        This method will always return a tuple of soquets. See `BloqBuilder.add_d(..)` for a
        method that returns a dictionary of soquets. See `BloqBuilder.add(..)` for a return
        type that depends on the arity of the bloq.

        Args:
            bloq: The bloq representing the operation to add.
            **in_soqs: Keyword arguments mapping the new bloq's register names to input
                `Soquet`s or an array thereof. This is likely the output soquets from a prior
                operation.

        Returns:
            A `Soquet` or an array thereof for each output register ordered according to
                `bloq.signature`. If a bloq has one or zero output registers, we will return
                a tuple of size one or zero, respectively. Note that the ordering is according
                to `bloq.signature` and irrespective of the order of `**in_soqs`.
        """
        binst = BloqInstance(bloq, i=self._new_binst_i())
        return tuple(soq for _, soq in self._add_binst(binst, in_soqs=in_soqs))

    def add_d(self, bloq: Bloq, **in_soqs: SoquetInT) -> Dict[str, SoquetT]:
        """Add a new bloq instance to the compute graph and return new soquets as a dictionary.

        This method returns a dictionary of soquets. See `BloqBuilder.add_t(..)` for a method
        that returns an ordered tuple of soquets. See `BloqBuilder.add(..)` for a return
        type that depends on the arity of the bloq.

        Args:
            bloq: The bloq representing the operation to add.
            **in_soqs: Keyword arguments mapping the new bloq's register names to input
                `Soquet`s or an array thereof. This is likely the output soquets from a prior
                operation.

        Returns:
            A dictionary mapping right (output) register names to SoquetT.
        """
        binst = BloqInstance(bloq, i=self._new_binst_i())
        return dict(self._add_binst(binst, in_soqs=in_soqs))

    def add(self, bloq: Bloq, **in_soqs: SoquetInT) -> Union[None, SoquetT, Tuple[SoquetT, ...]]:
        """Add a new bloq instance to the compute graph.

        This is the primary method for building a composite bloq. Each call to `add` adds a
        new bloq instance to the compute graph, wires up the soquets from prior operations
        into the new bloq, and returns new soquets to be used for subsequent bloqs.

        This method will raise a `BloqError` if the addition is invalid. Soquets must be
        used exactly once and soquets must match the `Register` specifications of the bloq.

        See also `add_t` or `add_d` for versions of this function that return output soquets
        in a structured way that may be more appropriate for programmatic adding of bloqs.

        Args:
            bloq: The bloq representing the operation to add.
            **in_soqs: Keyword arguments mapping the new bloq's register names to input
                `Soquet`s or an array thereof. This is likely the output soquets from a prior
                operation.

        Returns:
            A `Soquet` or an array thereof for each right (output) register ordered according to
                `bloq.signature`. If `bloq` has no right registers, this will return `None`;
                If there is one right register, we return one `SoquetT`; If there are multiple
                right registers we return a tuple of `SoquetT` that can be unpacked with tuple
                unpacking. In this final case, the ordering is according to `bloq.signature`
                and irrespective of the order of `**in_soqs`.
        """
        outs = self.add_t(bloq, **in_soqs)
        if len(outs) == 0:
            return None
        if len(outs) == 1:
            return outs[0]
        return outs

    def _add_binst(
        self, binst: BloqInstance, in_soqs: Dict[str, SoquetInT]
    ) -> Iterator[Tuple[str, SoquetT]]:
        """Add a bloq instance.

        Warning! Do not use this function externally! Untold bad things will happen if
        the provided `binst.i` is not unique.
        """
        self._binsts.add(binst)

        bloq = binst.bloq

        def _add(idxed_soq: Soquet, reg: Register, idx: Tuple[int, ...]):
            # close over `binst`
            return self._add_cxn(binst, idxed_soq, reg, idx)

        _process_soquets(
            registers=bloq.signature.lefts(), in_soqs=in_soqs, debug_str=str(bloq), func=_add
        )
        yield from (
            (reg.name, _reg_to_soq(binst, reg, available=self._available))
            for reg in bloq.signature.rights()
        )

    def add_from(self, bloq: Bloq, **in_soqs: SoquetInT) -> Tuple[SoquetT, ...]:
        """Add all the sub-bloqs from `bloq` to the composite bloq under construction.

        Args:
            bloq: Where to add from. If this is a composite bloq, use its contents directly.
                Otherwise, we call `decompose_bloq()` first.
            in_soqs: Input soquets for `bloq`; used to connect its left-dangling soquets.

        Returns:
            The output soquets from `cbloq`.
        """
        if isinstance(bloq, CompositeBloq):
            cbloq = bloq
        else:
            cbloq = bloq.decompose_bloq()

        for k, v in in_soqs.items():
            if not isinstance(v, Soquet):
                in_soqs[k] = np.asarray(v)

        # Initial mapping of LeftDangle according to user-provided in_soqs.
        soq_map: List[Tuple[SoquetT, SoquetT]] = [
            (_reg_to_soq(LeftDangle, reg), in_soqs[reg.name]) for reg in cbloq.signature.lefts()
        ]

        for binst, in_soqs, old_out_soqs in cbloq.iter_bloqsoqs():
            in_soqs = _map_soqs(in_soqs, soq_map)
            new_out_soqs = self.add_t(binst.bloq, **in_soqs)
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = _map_soqs(cbloq.final_soqs(), soq_map)
        return tuple(fsoqs[reg.name] for reg in cbloq.signature.rights())

    def finalize(self, **final_soqs: SoquetT) -> CompositeBloq:
        """Finish building a CompositeBloq and return the immutable CompositeBloq.

        This method is similar to calling `add()` but instead of adding a new Bloq,
        it configures the final "dangling" soquets that serve as the outputs for
        the composite bloq as a whole.

        If `self.add_registers_allowed` is set to `True`, additional register
        names passed to this function will be added as RIGHT registers. Otherwise,
        this method validates the provided `final_soqs` against our list of RIGHT
        (and THRU) registers.

        Args:
            **final_soqs: Keyword arguments mapping the composite bloq's register names to
                final`Soquet`s, e.g. the output soquets from a prior, final operation.
        """
        if not self.add_register_allowed:
            return self._finalize_strict(**final_soqs)

        # If items from `final_soqs` don't already exist in `_regs`, add RIGHT registers
        # for them. Then call `_finalize_strict` where the actual dangling connections are added.

        def _infer_reg(name: str, soq: SoquetT) -> Register:
            """Go from Soquet -> register, but use a specific name for the register."""
            if isinstance(soq, Soquet):
                return Register(name=name, dtype=soq.reg.dtype, side=Side.RIGHT)

            # Get info from 0th soquet in an ndarray.
            return Register(
                name=name, dtype=soq.reshape(-1)[0].reg.dtype, shape=soq.shape, side=Side.RIGHT
            )

        right_reg_names = [reg.name for reg in self._regs if reg.side & Side.RIGHT]
        for name, soq in final_soqs.items():
            if name not in right_reg_names:
                self._regs.append(_infer_reg(name, soq))

        return self._finalize_strict(**final_soqs)

    def _finalize_strict(self, **final_soqs: SoquetT) -> CompositeBloq:
        """Finish building a CompositeBloq and return the immutable CompositeBloq.

        Args:
            **final_soqs: Keyword arguments mapping the composite bloq's register names to
                final`Soquet`s, e.g. the output soquets from a prior, final operation.
        """
        signature = Signature(self._regs)

        def _fin(idxed_soq: Soquet, reg: Register, idx: Tuple[int, ...]):
            # close over `RightDangle`
            return self._add_cxn(RightDangle, idxed_soq, reg, idx)

        _process_soquets(
            registers=signature.rights(), debug_str='Finalizing', in_soqs=final_soqs, func=_fin
        )
        if self._available:
            raise BloqError(
                f"During finalization, {self._available} Soquets were not used."
            ) from None

        return CompositeBloq(
            connections=self._cxns, signature=signature, bloq_instances=self._binsts
        )

    def allocate(self, n: int = 1, dtype: Optional[QDType] = None) -> Soquet:
        from qualtran.bloqs.util_bloqs import Allocate

        if dtype is not None:
            return self.add(Allocate(dtype=dtype))
        return self.add(Allocate(dtype=(QAny(n))))

    def free(self, soq: Soquet) -> None:
        from qualtran.bloqs.util_bloqs import Free

        if not isinstance(soq, Soquet):
            raise ValueError("`free` expects a single Soquet to free.")

        self.add(Free(dtype=soq.reg.dtype), reg=soq)

    def split(self, soq: Soquet) -> NDArray[Soquet]:
        """Add a Split bloq to split up a register."""
        from qualtran.bloqs.util_bloqs import Split

        if not isinstance(soq, Soquet):
            raise ValueError("`split` expects a single Soquet to split.")

        return self.add(Split(dtype=soq.reg.dtype), reg=soq)

    def join(self, soqs: NDArray[Soquet], dtype: Optional[QDType] = None) -> Soquet:
        from qualtran.bloqs.util_bloqs import Join

        try:
            (n,) = soqs.shape
        except AttributeError:
            raise ValueError("`join` expects a 1-d array of input soquets to join.") from None

        if not all(soq.reg.bitsize == 1 for soq in soqs):
            raise ValueError("`join` can only join equal-bitsized soquets, currently only size 1.")
        if dtype is None:
            dtype = QAny(n)

        return self.add(Join(dtype=dtype), reg=soqs)
