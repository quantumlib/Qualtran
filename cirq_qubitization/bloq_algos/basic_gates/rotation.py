from functools import cached_property
from typing import Dict, Tuple

import numpy as np
from attrs import frozen

from cirq_qubitization import t_complexity_protocol
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


@frozen
class Rz(Bloq):
    """Single-qubit Rz gate.

    Args:
        angle: Rotation angle.
        eps: precision for implementation of rotation.

    Registers:
        - q: One-bit register.

    References:
        (Efficient synthesis of universal Repeat-Until-Success circuits)
        [https://arxiv.org/abs/1404.5320], which offers a small improvement
        (Optimal ancilla-free Clifford+T approximation of z-rotations)
        [https://arxiv.org/pdf/1403.2975.pdf].
    """

    angle: float
    eps: float = 1e-11

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # This is an improvement over Ref. 2 from the docstring which provides
        # a bound of 3 log(1/eps).
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_t = int(np.ceil(1.149 * np.log2(1.0 / self.eps) + 9.2))
        return t_complexity_protocol.TComplexity(t=num_t)

    def as_cirq_op(
        self, q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rz(self.angle).on(q), {'q': [q]}


@frozen
class Rx(Rz):
    def as_cirq_op(
        self, q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rx(self.angle).on(q), {'q': [q]}


@frozen
class Ry(Rz):
    def as_cirq_op(
        self, q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.ry(self.angle).on(q), {'q': [q]}
