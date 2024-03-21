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
from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING, Union

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import Ry, ZPowGate
from qualtran.drawing import TextBox

if TYPE_CHECKING:
    import cirq

    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.drawing import WireSymbol


@frozen
class SU2RotationGate(GateWithRegisters):
    r"""Implements an arbitrary SU(2) rotation.

    The rotation is represented by the matrix:

        $$
        \begin{matrix}
        e^{i(\lambda + \phi)} \cos(\theta) & e^{i\phi} \sin(\theta) \\
        e^{i\lambda} \sin(\theta) & - \cos(\theta)
        \end{matrix}
        $$

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Equation 7.
    """

    theta: float
    phi: float
    lambd: float

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    @cached_property
    def rotation_matrix(self) -> NDArray[np.complex_]:
        return np.array(
            [
                [
                    np.exp(1j * (self.lambd + self.phi)) * np.cos(self.theta),
                    np.exp(1j * self.phi) * np.sin(self.theta),
                ],
                [np.exp(1j * self.lambd) * np.sin(self.theta), -np.cos(self.theta)],
            ]
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        import cirq

        (qubit,) = q
        return cirq.PhasedXZGate(
            x_exponent=2 * self.theta / np.pi,
            z_exponent=1 - (self.lambd + self.phi) / np.pi,
            axis_phase_exponent=self.lambd / np.pi - 0.5,
        ).on(qubit), {'q': q}

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> Dict[str, 'SoquetT']:
        q = bb.add(ZPowGate(exponent=2, global_shift=0.5), q=q)
        q = bb.add(ZPowGate(exponent=1 - self.lambd / np.pi, global_shift=-1), q=q)
        q = bb.add(Ry(angle=2 * self.theta), q=q)
        q = bb.add(ZPowGate(exponent=-self.phi / np.pi, global_shift=-1), q=q)
        return {'q': q}

    def pretty_name(self) -> str:
        return f'SU_2({self.theta}, {self.phi}, {self.lambd})'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        return TextBox(self.pretty_name())
