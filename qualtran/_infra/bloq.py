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


"""Contains the main interface for defining `Bloq`s."""

import abc
from typing import Any, Dict, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import cirq
    import networkx as nx
    import quimb.tensor as qtn
    import sympy
    from numpy.typing import NDArray

    from qualtran import (
        AddControlledT,
        BloqBuilder,
        CompositeBloq,
        CtrlSpec,
        Signature,
        Soquet,
        SoquetT,
    )
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, GeneralizerT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _decompose_from_build_composite_bloq(bloq: 'Bloq') -> 'CompositeBloq':
    from qualtran import BloqBuilder

    bb, initial_soqs = BloqBuilder.from_signature(bloq.signature, add_registers_allowed=False)
    out_soqs = bloq.build_composite_bloq(bb=bb, **initial_soqs)
    return bb.finalize(**out_soqs)


class DecomposeNotImplementedError(NotImplementedError):
    """Raised if a decomposition is not yet provided.

    In contrast to `DecomposeTypeError`, a decomposition is theoretically possible; just not
    implemented yet.
    """


class DecomposeTypeError(TypeError):
    """Raised if a decomposition does not make sense in this context.

    In contrast to `DecomposeNotImplementedError`, a decomposition does not make sense
    in this context. This can be raised if the bloq is "atomic" -- that is, considered part
    of the compilation target gateset. This can be raised if certain bloq attributes do not
    permit a decomposition, most commonly if an attribute is symbolic.
    """


class Bloq(metaclass=abc.ABCMeta):
    """Bloq is the primary abstract base class for all operations.

    Bloqs let you represent high-level quantum programs and subroutines as a hierarchical
    collection of Python objects. The main interface is this abstract base class.

    There are two important flavors of implementations of the `Bloq` interface. The first flavor
    consists of bloqs implemented by you, the user-developer to express quantum operations of
    interest. For example:

    >>> class ShorsAlgorithm(Bloq):
    >>>     ...

    The other important `Bloq` subclass is `CompositeBloq`, which is a container type for a
    collection of sub-bloqs.

    There is only one mandatory method you must implement to have a well-formed `Bloq`,
    namely `Bloq.registers`. There are many other methods you can optionally implement to
    encode more information about the bloq.
    """

    @property
    @abc.abstractmethod
    def signature(self) -> 'Signature':
        """The input and output names and types for this bloq.

        This property can be thought of as analogous to the function signature in ordinary
        programming. For example, it is analogous to function declarations in a
        C header (`*.h`) file.

        This is the only mandatory method (property) you must implement to inherit from
        `Bloq`. You can optionally implement additional methods to encode more information
        about this bloq.
        """

    def pretty_name(self) -> str:
        return self.__class__.__name__

    def short_name(self) -> str:
        name = self.pretty_name()
        if len(name) <= 10:
            return name

        return name[:8] + '..'

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        """Override this method to define a Bloq in terms of its constituent parts.

        Bloq authors should override this method. If you already have an instance of a `Bloq`,
        consider calling `decompose_bloq()` which will set up the correct context for
        calling this function.

        Args:
            bb: A `BloqBuilder` to append sub-Bloq to.
            **soqs: The initial soquets corresponding to the inputs to the Bloq.

        Returns:
            The soquets corresponding to the outputs of the Bloq (keyed by name) or
            `NotImplemented` if there is no decomposition.
        """
        raise DecomposeNotImplementedError(f"{self} does not declare a decomposition.")

    def decompose_bloq(self) -> 'CompositeBloq':
        """Decompose this Bloq into its constituent parts contained in a CompositeBloq.

        Bloq users can call this function to delve into the definition of a Bloq. If you're
        trying to define a bloq's decomposition, consider overriding `build_composite_bloq`
        which provides helpful arguments for implementers.

        Returns:
            A CompositeBloq containing the decomposition of this Bloq.

        Raises:
            NotImplementedError: If there is no decomposition defined; namely: if
                `build_composite_bloq` returns `NotImplemented`.
        """
        return _decompose_from_build_composite_bloq(self)

    def supports_decompose_bloq(self) -> bool:
        """Whether this bloq supports `.decompose_bloq()`.

        By default, we check that the method `build_composite_bloq` is overriden. For
        extraordinary circumstances, you may need to override this method directly to
        return an accurate value.
        """
        return not self.build_composite_bloq.__qualname__.startswith('Bloq.')

    def as_composite_bloq(self) -> 'CompositeBloq':
        """Wrap this Bloq into a size-1 CompositeBloq.

        This method is overriden so if this Bloq is already a CompositeBloq, it will
        be returned.
        """
        from qualtran import BloqBuilder

        bb, initial_soqs = BloqBuilder.from_signature(self.signature, add_registers_allowed=False)
        return bb.finalize(**bb.add_d(self, **initial_soqs))

    def adjoint(self) -> 'Bloq':
        """The adjoint of this bloq.

        Bloq authors can override this method in certain circumstances. Otherwise, the default
        fallback wraps this bloq in `Adjoint`.

        Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
        details.
        """

        from qualtran import Adjoint

        return Adjoint(self)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        """How this bloq operates on classical data.

        Override this method if your bloq represents classical, reversible logic. For example:
        quantum circuits composed of X and C^nNOT gates are classically simulable.

        Bloq definers should override this method. If you already have an instance of a `Bloq`,
        consider calling `call_clasically(**vals)` which will do input validation before
        calling this function.

        Args:
            **vals: The input classical values for each left (or thru) register. The data
                types are guaranteed to match `self.registers`. Values for registers
                with bitsize `n` will be integers of that bitsize. Values for registers with
                `shape` will be an ndarray of integers of the given bitsize. Note: integers
                can be either Numpy or Python integers. If they are Python integers, they
                are unsigned.

        Returns:
            A dictionary mapping right (or thru) register name to output classical values.
        """
        try:
            return self.decompose_bloq().on_classical_vals(**vals)
        except NotImplementedError as e:
            raise NotImplementedError(f"{self} does not support classical simulation: {e}") from e

    def call_classically(self, **vals: 'ClassicalValT') -> Tuple['ClassicalValT', ...]:
        """Call this bloq on classical data.

        Bloq users can call this function to apply bloqs to classical data. If you're
        trying to define a bloq's action on classical values, consider overriding
        `on_classical_vals` which promises type checking for arguments.

        Args:
            **vals: The input classical values for each left (or thru) register. The data
                types must match `self.registers`. Values for registers
                with bitsize `n` should be integers of that bitsize or less. Values for registers
                with `shape` should be an ndarray of integers of the given bitsize.
                Note: integers can be either Numpy or Python integers, but should be positive
                and unsigned.

        Returns:
            A tuple of output classical values ordered according to this bloqs right (or thru)
            registers.
        """
        res = self.as_composite_bloq().on_classical_vals(**vals)
        return tuple(res[reg.name] for reg in self.signature.rights())

    def tensor_contract(self) -> 'NDArray':
        """Return a contracted, dense ndarray representing this bloq.

        This constructs a tensor network and then contracts it according to our registers,
        i.e. the dangling indices. The returned array will be 0-, 1- or 2-dimensional. If it is
        a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
        of (right, left) indices.
        """
        from qualtran.simulation.tensor import bloq_to_dense

        return bloq_to_dense(self)

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        """Override this method to support native quimb simulation of this Bloq.

        This method is responsible for adding a tensor corresponding to the unitary, state, or
        effect of the bloq to the provided tensor network `tn`. Often, this method will add
        one tensor for a given Bloq, but some bloqs can be represented in a factorized form
        requiring the addition of more than one tensor.

        If this method is not overriden, the default implementation will try to use the bloq's
        decomposition to find a dense representation for this bloq.

        Args:
            tn: The tensor network to which we add our tensor(s)
            tag: An arbitrary tag that must be forwarded to `qtn.Tensor`'s `tag` attribute.
            incoming: A mapping from register name to SoquetT to order left indices for
                the tensor network.
            outgoing: A mapping from register name to SoquetT to order right indices for
                the tensor network.
        """
        from qualtran.simulation.tensor import cbloq_as_contracted_tensor

        cbloq = self.decompose_bloq()
        tn.add(cbloq_as_contracted_tensor(cbloq, incoming, outgoing, tags=[self.short_name(), tag]))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        """Override this method to build the bloq call graph.

        This method must return a set of `(bloq, n)` tuples where `bloq` is called `n` times in
        the decomposition. This method defines one level of the call graph, specifically the
        edges from this bloq to its immediate children. To get the full graph,
        call `Bloq.call_graph()`.

        By default, this method will use `self.decompose_bloq()` to count the bloqs called
        in the decomposition. By overriding this method, you can provide explicit call counts.
        This is appropriate if: 1) you can't or won't provide a complete decomposition, 2) you
        know symbolic expressions for the counts, or 3) you need to "generalize" the subbloqs
        by overwriting bloq attributes that do not affect its cost with generic sympy symbols using
        the provided `SympySymbolAllocator`.
        """
        return self.decompose_bloq().build_call_graph(ssa)

    def call_graph(
        self,
        generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
        keep: Optional[Sequence['Bloq']] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple['nx.DiGraph', Dict['Bloq', Union[int, 'sympy.Expr']]]:
        """Get the bloq call graph and call totals.

        The call graph has edges from a parent bloq to each of the bloqs that it calls in
        its decomposition. The number of times it is called is stored as an edge attribute.
        To specify the bloq call counts for a specific node, override `Bloq.build_call_graph()`.

        Args:
            generalizer: If provided, run this function on each (sub)bloq to replace attributes
                that do not affect resource estimates with generic sympy symbols. If the function
                returns `None`, the bloq is omitted from the counts graph. If a sequence of
                generalizers is provided, each generalizer will be run in order.
            keep: If this function evaluates to True for the current bloq, keep the bloq as a leaf
                node in the call graph instead of recursing into it.
            max_depth: If provided, build a call graph with at most this many layers.

        Returns:
            g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
                the number of times successor bloq is called via its predecessor.
            sigma: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
                according to `keep` and `max_depth` (if provided) or if a bloq cannot be
                decomposed.
        """
        from qualtran.resource_counting.bloq_counts import get_bloq_call_graph

        return get_bloq_call_graph(self, generalizer=generalizer, keep=keep, max_depth=max_depth)

    def bloq_counts(
        self, generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None
    ) -> Dict['Bloq', Union[int, 'sympy.Expr']]:
        """The number of subbloqs directly called by this bloq.

        This corresponds to one level of the call graph, see `Bloq.call_graph()`.
        To specify explicit values for a bloq, override `Bloq.build_call_graph(...)`, not this
        method.

        Args:
            generalizer: If provided, run this function on each (sub)bloq to replace attributes
                that do not affect resource estimates with generic sympy symbols. If the function
                returns `None`, the bloq is omitted from the counts graph. If a sequence of
                generalizers is provided, each generalizer will be run in order.

        Returns:
            A dictionary mapping subbloq to the number of times they appear in the decomposition.
        """
        _, sigma = self.call_graph(generalizer=generalizer, max_depth=1)
        return sigma

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> Tuple['Bloq', 'AddControlledT']:
        """Get a controlled version of this bloq and a function to wire it up correctly.

        Users should likely call `Bloq.controlled(...)` which uses this method behind-the-scenes.
        Intrepid bloq authors can override this method to provide a custom controlled version of
        this bloq. By default, this will use the `qualtran.Controlled` meta-bloq to control any
        bloq.

        This method must return both a controlled version of this bloq and a callable that
        'wires up' soquets correctly.

        A controlled version of this bloq has all the registers from the original bloq plus
        any additional control registers to support the activation function specified by
        the `ctrl_spec`. In the simplest case, this could be one additional 1-qubit register
        that activates the bloq if the input is in the |1> state, but additional logic is possible.
        See the documentation for `CtrlSpec` for more information.

        The second return value ensures we can accurately wire up soquets into the added registers.
        It must have the following signature:

            def _my_add_controlled(
                bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
            ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:

        Which takes a bloq builder (for adding the controlled bloq), the new control soquets,
        input soquets for the existing registers; and returns a sequence of the output control
        soquets and a sequence of the output soquets for the existing registers. This complexity
        is sadly unavoidable due to the variety of ways of wiring up custom controlled bloqs.

        Returns:
            controlled_bloq: A controlled version of this bloq
            add_controlled: A function with the signature documented above that the system
                can use to automatically wire up the new control registers.
        """
        from qualtran import Controlled, CtrlSpec

        if ctrl_spec is None:
            ctrl_spec = CtrlSpec()

        return Controlled.make_ctrl_system(self, ctrl_spec=ctrl_spec)

    def controlled(self, ctrl_spec: Optional['CtrlSpec'] = None) -> 'Bloq':
        """Return a controlled version of this bloq.

        By default, the system will use the `qualtran.Controlled` meta-bloq to wrap this
        bloq. Bloqs authors can declare their own, custom controlled versions by overriding
        `Bloq.get_ctrl_system` in the bloq.

        Args:
            ctrl_spec: an optional `CtrlSpec`, which specifies how to control the bloq. The
                default spec means the bloq will be active when one control qubit is in the |1>
                state. See the CtrlSpec documentation for more possibilities including
                negative controls, integer-equality control, and ndarrays of control values.

        Returns:
            A controlled version of the bloq.
        """
        controlled_bloq, _ = self.get_ctrl_system(ctrl_spec=ctrl_spec)
        return controlled_bloq

    def t_complexity(self) -> 'TComplexity':
        """The `TComplexity` for this bloq.

        By default, this will recurse into this bloq's decomposition but this
        method can be overriden with a known value.
        """
        from qualtran.cirq_interop.t_complexity_protocol import t_complexity

        return t_complexity(self)

    def _t_complexity_(self) -> 'TComplexity':
        return NotImplemented

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        """Override this method to support conversion to a Cirq operation.

        If this method is not overriden, the default implementation will wrap this bloq
        in a `BloqAsCirqGate` shim.

        Args:
            qubit_manager: A `cirq.QubitManager` for allocating `cirq.Qid`s.
            **cirq_quregs: kwargs mapping from this bloq's left register names to an ndarray of
                `cirq.Qid`. The final dimension of this array corresponds to the registers
                `bitsize` size. Any additional dimensions come first and correspond to the
                register `shape` sizes.

        Returns:
            op: A cirq operation corresponding to this bloq acting on the provided cirq qubits or
                None. This method should return None if and only if the bloq instance truly should
                not be included in the Cirq circuit (e.g. for reshaping bloqs). A bloq with no cirq
                equivalent should raise an exception instead.
            cirq_quregs: A mapping from this bloq's right register of the same format as the
                `cirq_quregs` argument. The returned dictionary corresponds to the output qubits.
        """
        from qualtran.cirq_interop import BloqAsCirqGate

        return BloqAsCirqGate.bloq_on(
            bloq=self, cirq_quregs=cirq_quregs, qubit_manager=qubit_manager
        )

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.Operation':
        """A `cirq.Operation` of this bloq operating on the given qubits.

        This method supports an alternative decomposition backend that follows a 'Cirq-style'
        association of gates with qubits to form operations. Instead of wiring up `Soquet`s,
        each gate operates on qubit addresses (`cirq.Qid`s), which are reused by multiple
        gates. This method lets you operate this bloq on qubits and returns a `cirq.Operation`.

        The primary, bloq-native way of writing decompositions is to override
        `build_composite_bloq`. If this is what you're doing, do not use this method.

        To provide a Cirq-style decomposition for this bloq, implement a method (typically named
        `decompose_from_registers` for historical reasons) that yields a list of `cirq.Operation`s
        using `cirq.Gate.on(...)`, `Bloq.on(...)`, `GateWithRegisters.on_registers(...)`, or
        `Bloq.on_registers(...)`.

        See Also:
            `Bloq.on_registers`: Provides the same functionality, but with named registers
                instead of a flat list of qubits.
            `decompose_from_cirq_style_method`: More details on how to write a cirq-style
                decomposition.
        """
        import cirq

        from qualtran.cirq_interop import BloqAsCirqGate

        return cirq.Gate.on(BloqAsCirqGate(bloq=self), *qubits)

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid'], 'NDArray[cirq.Qid]']
    ) -> 'cirq.Operation':
        """A `cirq.Operation` of this bloq operating on the given qubit registers.

        This method supports an alternative decomposition backend that follows a 'Cirq-style'
        association of gates with qubit registers to form operations. See `Bloq.on()` for
        more details.

        Args:
            **qubit_regs: A mapping of register name to the qubits comprising that register.

        See Also:
            `Bloq.on`: Provides the same functionality, but with a flat list of qubits.
                instead of named registers.
            `decompose_from_cirq_style_method`: More details on how to write a cirq-style
                decomposition.
        """
        from qualtran._infra.gate_with_registers import merge_qubits

        return self.on(*merge_qubits(self.signature, **qubit_regs))

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        """On a musical score visualization, use this `WireSymbol` to represent `soq`.

        By default, we use a "directional text box", which is a text box that is either
        rectangular for thru-registers or facing to the left or right for non-thru-registers.

        Override this method to provide a more relevant `WireSymbol` for the provided soquet.
        This method can access bloq attributes. For example: you may want to draw either
        a filled or empty circle for a control register depending on a control value bloq
        attribute.
        """
        from qualtran.drawing import directional_text_box

        return directional_text_box(text=soq.pretty(), side=soq.reg.side)
