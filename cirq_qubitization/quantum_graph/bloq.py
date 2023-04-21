import abc
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn
    from numpy.typing import NDArray

    from cirq_qubitization import TComplexity
    from cirq_qubitization.quantum_graph.composite_bloq import (
        CompositeBloq,
        CompositeBloqBuilder,
        SoquetT,
    )
    from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


class Bloq(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> 'FancyRegisters':
        ...

    def pretty_name(self) -> str:
        return self.__class__.__name__

    def short_name(self) -> str:
        name = self.pretty_name()
        if len(name) <= 8:
            return name

        return name[:6] + '..'

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        """Override this method to define a Bloq in terms of its constituent parts.

        Bloq definers should override this method. If you already have an instance of a `Bloq`,
        consider calling `decompose_bloq()` which will set up the correct context for
        calling this function.

        Args:
            bb: A `CompositeBloqBuilder` to append sub-Bloq to.
            **soqs: The initial soquets corresponding to the inputs to the Bloq.

        Returns:
            The soquets corresponding to the outputs of the Bloq (keyed by name) or
            `NotImplemented` if there is no decomposition.
        """
        return NotImplemented

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
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb, initial_soqs = CompositeBloqBuilder.from_registers(
            self.registers, add_registers_allowed=False
        )
        out_soqs = self.build_composite_bloq(bb=bb, **initial_soqs)
        if out_soqs is NotImplemented:
            raise NotImplementedError(f"Cannot decompose {self}.")

        return bb.finalize(**out_soqs)

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
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb, initial_soqs = CompositeBloqBuilder.from_registers(
            self.registers, add_registers_allowed=False
        )
        ret_soqs_tuple = bb.add(self, **initial_soqs)
        assert len(list(self.registers.rights())) == len(ret_soqs_tuple)
        ret_soqs = {reg.name: v for reg, v in zip(self.registers.rights(), ret_soqs_tuple)}
        return bb.finalize(**ret_soqs)

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

        from cirq_qubitization.quantum_graph.quimb_sim import (
            _cbloq_as_contracted_tensor_data_and_inds,
        )

        cbloq = self.decompose_bloq()
        data, inds = _cbloq_as_contracted_tensor_data_and_inds(
            cbloq=cbloq, registers=self.registers, incoming=incoming, outgoing=outgoing
        )
        tn.add(qtn.Tensor(data=data, inds=inds, tags=[self.short_name(), tag]))

    def t_complexity(self) -> 'TComplexity':
        """The `TComplexity` for this bloq.

        By default, this will recurse into this bloq's decomposition but this
        method can be overriden with a known value.
        """
        return self.decompose_bloq().t_complexity()

    def on_registers(self, **qubit_regs: 'NDArray[cirq.Qid]') -> 'cirq.OP_TREE':
        """Support for conversion to a Cirq circuit."""
        raise NotImplementedError("This bloq does not support Cirq conversion.")


class NoCirqEquivalent(NotImplementedError):
    """Raise this in `Bloq.on_registers` to signify that it should be omitted from Cirq circuits.

    For example, this would apply for qubit bookkeeping operations.
    """
