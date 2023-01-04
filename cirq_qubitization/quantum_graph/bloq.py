import abc
from typing import Dict, Sequence, TYPE_CHECKING

import cirq

from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
    from cirq_qubitization.quantum_graph.quantum_graph import Soquet


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
        self, bb: 'CompositeBloqBuilder', **soqs: 'Soquet'
    ) -> Dict[str, 'Soquet']:
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
            NotImplementedError if there is no decomposition defined; namely: if
            `build_composite_bloq` returns `NotImplemented`.
        """
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb = CompositeBloqBuilder(self.registers)
        out_soqs = self.build_composite_bloq(bb=bb, **bb.initial_soquets())
        if out_soqs is NotImplemented:
            raise NotImplementedError(f"Cannot decompose {self}.")

        return bb.finalize(**out_soqs)

    def as_composite_bloq(self) -> 'CompositeBloq':
        """Wrap this Bloq into a size-1 CompositeBloq."""
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb = CompositeBloqBuilder(self.registers)
        ret_soqs_tuple = bb.add(self, **bb.initial_soquets())
        assert len(list(self.registers.rights())) == len(ret_soqs_tuple)
        ret_soqs = {reg.name: v for reg, v in zip(self.registers.rights(), ret_soqs_tuple)}
        return bb.finalize(**ret_soqs)

    # ----- cirq stuff -----

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)

    def decompose_from_registers(self, **qubit_regs: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)


class NoCirqEquivalent(NotImplementedError):
    """Raise this in `Bloq.on_registers` to signify that it should be omitted from Cirq circuits.

    For example, this would apply for qubit bookkeeping operations.
    """
