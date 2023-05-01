from functools import cached_property

import numpy as np
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


@frozen
class Rz(Bloq):
    """Single-qubit Rz gate.

        Registers:
         - qubit: One-bit register.

        References:
            (Efficient synthesis of universal Repeat-Until-Success circuits
    )[https://arxiv.org/abs/1404.5320].
    """

    eps: float = 1e-11

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_t = 1.149 * np.log2(1.0 / self.eps) + 9.2
        return t_complexity_protocol.TComplexity(t=num_t)

    def as_cirq_op(
        self, q: 'CirqQuregT', angle: float
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rz(angle).on(q), {'q': [q]}


@frozen
class Rx(Rz):
    def as_cirq_op(
        self, q: 'CirqQuregT', angle: float
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rx(angle).on(q), {'q': [q]}


@frozen
class Ry(Rz):
    def as_cirq_op(
        self, q: 'CirqQuregT', angle: float
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.ry(angle).on(q), {'q': [q]}
