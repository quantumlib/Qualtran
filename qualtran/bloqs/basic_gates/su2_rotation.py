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
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import Ry, ZPowGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import TextBox

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.drawing import WireSymbol


@frozen
class SU2RotationGate(GateWithRegisters):
    r"""Implements an arbitrary SU(2) rotation.

    The rotation is represented by the matrix:

    $$
        e^{i \alpha}
        \begin{pmatrix}
        e^{i(\lambda + \phi)} \cos(\theta) & e^{i\phi} \sin(\theta) \\
        e^{i\lambda} \sin(\theta) & - \cos(\theta)
        \end{pmatrix}
    $$

    where $s$ is the global phase shift.

    Args:
        theta: rotation angle $\theta$ in the above matrix.
        phi: phase angle $\theta$ in the above matrix.
        lambd: phase angle $\lambda$ in the above matrix.
        global_shift: phase angle $\alpha$, i.e. apply a global phase shift of $e^{i \alpha}$.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
        Motlagh and Wiebe. (2023). Equation 7.
    """

    theta: float
    phi: float
    lambd: float  # cannot use `lambda` as it is a python keyword
    global_shift: float = 0

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    @cached_property
    def rotation_matrix(self) -> NDArray[np.complex_]:
        return np.exp(1j * self.global_shift) * np.array(
            [
                [
                    np.exp(1j * (self.lambd + self.phi)) * np.cos(self.theta),
                    np.exp(1j * self.phi) * np.sin(self.theta),
                ],
                [np.exp(1j * self.lambd) * np.sin(self.theta), -np.cos(self.theta)],
            ]
        )

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        tn.add(
            qtn.Tensor(
                data=self.rotation_matrix,
                inds=(outgoing['q'], incoming['q']),
                tags=[self.short_name(), tag],
            )
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> Dict[str, 'SoquetT']:
        q = bb.add(ZPowGate(exponent=2, global_shift=0.5 + self.global_shift / (2 * np.pi)), q=q)
        q = bb.add(ZPowGate(exponent=1 - self.lambd / np.pi, global_shift=-1), q=q)
        q = bb.add(Ry(angle=2 * self.theta), q=q)
        q = bb.add(ZPowGate(exponent=-self.phi / np.pi, global_shift=-1), q=q)
        return {'q': q}

    def pretty_name(self) -> str:
        return 'SU_2'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        return TextBox(
            f"{self.pretty_name()}({self.theta}, {self.phi}, {self.lambd}, {self.global_shift})"
        )

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(rotations=1)


@bloq_example
def _su2_rotation_gate() -> SU2RotationGate:
    su2_rotation_gate = SU2RotationGate(np.pi / 4, np.pi / 2, np.pi / 2)
    return su2_rotation_gate


@bloq_example
def _hadamard() -> SU2RotationGate:
    hadamard = SU2RotationGate(np.pi / 4, 0, 0)
    return hadamard


@bloq_example
def _t_gate() -> SU2RotationGate:
    t_gate = SU2RotationGate(0, 3 * np.pi / 4, 0, -3 * np.pi / 4)
    return t_gate


_SU2_ROTATION_GATE_DOC = BloqDocSpec(
    bloq_cls=SU2RotationGate,
    import_line='from qualtran.bloqs.basic_gates import SU2RotationGate',
    examples=[_su2_rotation_gate, _hadamard, _t_gate],
)
