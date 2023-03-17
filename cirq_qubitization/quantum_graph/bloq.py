import abc
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING

import cirq
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import (
        CompositeBloq,
        CompositeBloqBuilder,
        SoquetT,
    )
    from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, Side

QUBITS = set()


def new_qubit() -> cirq.LineQubit:
    global QUBITS
    i = 0
    while True:
        q = cirq.LineQubit(i)
        if q not in QUBITS:
            QUBITS.add(q)
            return q
        i += 1


def free_qubit(q: cirq.Qid):
    global QUBITS
    QUBITS.remove(q)


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
            NotImplementedError if there is no decomposition defined; namely: if
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

    def tensor_contract(self) -> NDArray:
        """Return a contracted, dense ndarray representing this bloq.

        This constructs a tensor network and then contracts it according to our registers,
        i.e. the dangling indices. The returned array will be 0-, 1- or 2- dimensional. If it is
        a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
        of (right, left) indices.
        """
        return self.as_composite_bloq().tensor_contract()

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        raise NotImplementedError("This bloq does not support tensor contraction.")

    # ----- cirq stuff -----

    def decompose_from_registers(self, **qubit_regs: NDArray['cirq.Qid']) -> 'cirq.OP_TREE':
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)

    def on_registers(self, quregs: Dict[str, 'cirq.Qid']) -> 'cirq.Operation':
        from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, Side

        big_ol_qubits = []
        names = []
        for reg in self.registers:
            if reg.side is Side.THRU:
                for i, q in enumerate(quregs[reg.name]):
                    big_ol_qubits.append(q)
                    names.append(f'{reg.name}{i}')
            elif reg.side is Side.LEFT:
                for i, q in enumerate(quregs[reg.name]):
                    big_ol_qubits.append(q)
                    names.append(f'{reg.name}{i}')
                    free_qubit(q)
            elif reg.side is Side.RIGHT:
                qs = []
                for i in range(reg.total_bits()):
                    q = new_qubit()
                    qs.append(q)
                    big_ol_qubits.append(q)
                    names.append(f'{reg.name}{i}')
                quregs[reg.name] = qs

        return BloqToCirq(self, len(big_ol_qubits), tuple(names)).on(*big_ol_qubits)


@frozen
class BloqToCirq(cirq.Gate):
    bloq: Bloq
    nq: int
    names: Tuple[str, ...]

    def _num_qubits_(self):
        return self.nq

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        syms = [self.bloq.short_name()]
        syms.extend(self.names[i] for i in range(1, self.nq))
        return cirq.CircuitDiagramInfo(wire_symbols=syms)


class NoCirqEquivalent(NotImplementedError):
    """Raise this in `Bloq.on_registers` to signify that it should be omitted from Cirq circuits.

    For example, this would apply for qubit bookkeeping operations.
    """
