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

import abc
from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
from cirq_ft import TComplexity

from qualtran import Bloq, Signature

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT


@frozen
class RotationBloq(Bloq, metaclass=abc.ABCMeta):
    angle: float
    eps: float = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # This is an improvement over Ref. 2 from the docstring which provides
        # a bound of 3 log(1/eps).
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_t = int(np.ceil(1.149 * np.log2(1.0 / self.eps) + 9.2))
        return TComplexity(t=num_t)


@frozen
class Rz(RotationBloq):
    """Single-qubit Rz gate.

    Args:
        angle: Rotation angle.
        eps: precision for implementation of rotation.

    Registers:
     - q: One-bit register.

    References:
        [Efficient synthesis of universal Repeat-Until-Success
        circuits](https://arxiv.org/abs/1404.5320), which offers a small improvement
        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rz(self.angle).on(q), {'q': np.array([q])}


@frozen
class Rx(RotationBloq):
    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.rx(self.angle).on(q), {'q': np.array([q])}


@frozen
class Ry(RotationBloq):
    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.ry(self.angle).on(q), {'q': np.array([q])}
