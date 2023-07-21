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
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn
    from cirq_ft import TComplexity
    from numpy.typing import NDArray

    from qualtran import BloqBuilder, CompositeBloq, Signature, Soquet, SoquetT
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _decompose_from_build_composite_bloq(bloq: 'Bloq') -> 'CompositeBloq':
    from qualtran import BloqBuilder

    bb, initial_soqs = BloqBuilder.from_signature(bloq.signature, add_registers_allowed=False)
    out_soqs = bloq.build_composite_bloq(bb=bb, **initial_soqs)
    return bb.finalize(**out_soqs)


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

        This is the only manditory method (property) you must implement to inherit from
        `Bloq`. You can optionally implement additional methods to encode more information
        about this bloq.
        """

    def pretty_name(self) -> str:
        return self.__class__.__name__

    def short_name(self) -> str:
        name = self.pretty_name()
        if len(name) <= 8:
            return name

        return name[:6] + '..'

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        """Override this method to define a Bloq in terms of its constituent parts.

        Bloq definers should override this method. If you already have an instance of a `Bloq`,
        consider calling `decompose_bloq()` which will set up the correct context for
        calling this function.

        Args:
            bb: A `BloqBuilder` to append sub-Bloq to.
            **soqs: The initial soquets corresponding to the inputs to the Bloq.

        Returns:
            The soquets corresponding to the outputs of the Bloq (keyed by name) or
            `NotImplemented` if there is no decomposition.
        """
        raise NotImplementedError(f"{self} does not support decomposition.")

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
            raise NotImplementedError(f"{self} does not support classical simulation: {e}")

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
        return self.as_composite_bloq().tensor_contract()

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
        import quimb.tensor as qtn

        from qualtran.simulation.quimb_sim import _cbloq_as_contracted_tensor_data_and_inds

        cbloq = self.decompose_bloq()
        data, inds = _cbloq_as_contracted_tensor_data_and_inds(
            cbloq=cbloq, signature=self.signature, incoming=incoming, outgoing=outgoing
        )
        tn.add(qtn.Tensor(data=data, inds=inds, tags=[self.short_name(), tag]))

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        """Return a set of `(n, bloq)` tuples where bloq is used `n` times in the decomposition.

        By default, this method will use `self.decompose_bloq()` to count up bloqs.
        However, you can override this if you don't want to provide a complete decomposition,
        if you know symbolic expressions for the counts, or if you need to "generalize"
        the subbloqs by overwriting bloq attributes that do not affect its cost with generic
        sympy symbols (perhaps with the aid of the provided `SympySymbolAllocator`).
        """
        return self.decompose_bloq().bloq_counts(ssa)

    def t_complexity(self) -> 'TComplexity':
        """The `TComplexity` for this bloq.

        By default, this will recurse into this bloq's decomposition but this
        method can be overriden with a known value.
        """
        return self.decompose_bloq().t_complexity()

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
