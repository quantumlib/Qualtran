#  Copyright 2024 Google LLC
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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, ConnectionT, GateWithRegisters, Register, Signature
from qualtran.bloqs.basic_gates import GlobalPhase, Rx, Rz
from qualtran.drawing import Text, TextBox
from qualtran.symbolics import is_symbolic, pi, SymbolicFloat

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import BloqBuilder, SoquetT
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import SympySymbolAllocator


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

    Args:
        theta: rotation angle $\theta$ in the above matrix.
        phi: phase angle $\phi$ in the above matrix.
        lambd: phase angle $\lambda$ in the above matrix.
        global_shift: phase angle $\alpha$, i.e. apply a global phase shift of $e^{i \alpha}$.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
        Motlagh and Wiebe. (2023). Equation 7.
    """

    theta: SymbolicFloat
    phi: SymbolicFloat
    lambd: SymbolicFloat  # cannot use `lambda` as it is a python keyword
    global_shift: SymbolicFloat = 0
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    @cached_property
    def rotation_matrix(self) -> NDArray[np.complex128]:
        if isinstance(self.lambd, sympy.Expr):
            raise ValueError(f'Symbolic lambda not allowed: {self.lambd}')
        if isinstance(self.phi, sympy.Expr):
            raise ValueError(f'Symbolic phi not allowed: {self.phi}')
        if isinstance(self.theta, sympy.Expr):
            raise ValueError(f'Symbolic theta not allowed: {self.theta}')
        if isinstance(self.global_shift, sympy.Expr):
            raise ValueError(f'Symbolic global_shift not allowed: {self.global_shift}')
        return np.exp(1j * self.global_shift) * np.array(
            [
                [
                    np.exp(1j * (self.lambd + self.phi)) * np.cos(self.theta),
                    np.exp(1j * self.phi) * np.sin(self.theta),
                ],
                [np.exp(1j * self.lambd) * np.sin(self.theta), -np.cos(self.theta)],
            ]
        )

    @classmethod
    def from_matrix(cls, mat: NDArray[np.complex128]) -> 'SU2RotationGate':

        theta = np.arctan2(np.abs(mat[1, 0]), np.abs(mat[0, 0]))
        if np.isclose(np.cos(theta), 0):
            alpha = 0
            phi = np.angle(mat[0, 1] / np.sin(theta))
            lambd = np.angle(mat[1, 0] / np.sin(theta))
        else:
            alpha = np.angle(-mat[1, 1] / np.cos(theta))
            if np.isclose(np.sin(theta), 0):
                phi = np.angle(mat[0, 0] / np.cos(theta) * np.exp(-1j * alpha))
                lambd = 0
            else:
                phi = np.angle(mat[0, 1] / np.sin(theta) * np.exp(-1j * alpha))
                lambd = np.angle(mat[1, 0] / np.sin(theta) * np.exp(-1j * alpha))

        return cls(theta, phi, lambd, alpha)

    @classmethod
    def from_euler_zxz_angles(
        cls,
        z_1: SymbolicFloat,
        x: SymbolicFloat,
        z_2: SymbolicFloat,
        *,
        global_shift: SymbolicFloat = 0,
        eps: Optional[SymbolicFloat],
    ) -> 'SU2RotationGate':
        r"""SU(2) rotation from Z-X-Z Euler angles.

        Corresponds to the matrix $e^{i \delta} Rz(\phi) Rx(\theta) Rz(\psi)$, i.e.

        $$
            e^{i \delta}
            \begin{pmatrix}
            \cos(\beta / 2)
            \end{pmatrix}
        $$

        All angles are in radians.

        Args:
            z_1: angle $\psi$ of the first Rz.
            x: angle $\theta$ of the middle Rx.
            z_2: angle $\phi$ of the second Rz.
            global_shift: global phase shift exponent $\delta$.
            eps: precision of rotation.
        """
        su2_lambd = pi(z_1) / 2 - z_1
        su2_theta = x / 2
        su2_phi = pi(z_2) / 2 - z_2
        su2_global_shift = global_shift + (z_1 + z_2) / 2 - pi(global_shift, z_1, z_2)

        if eps is not None:
            return cls(su2_theta, su2_phi, su2_lambd, su2_global_shift, eps=eps)
        else:
            return cls(su2_theta, su2_phi, su2_lambd, su2_global_shift)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=self.rotation_matrix,
                inds=[(outgoing['q'], 0), (incoming['q'], 0)],
                tags=[str(self)],
            )
        ]

    def _unitary_(self):
        if self.is_symbolic():
            return None
        return self.rotation_matrix

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> Dict[str, 'SoquetT']:
        # TODO implement controlled version, and pass eps/4 to each rotation (incl. global phase)
        #      https://github.com/quantumlib/Qualtran/issues/1330
        bb.add(
            GlobalPhase(
                exponent=(
                    0.5
                    + self.global_shift / pi(self.global_shift)
                    + self.lambd / (2 * pi(self.lambd))
                    + self.phi / (2 * pi(self.phi))
                )
            )
        )
        q = bb.add(Rz(pi(self.lambd) / 2 - self.lambd, eps=self.eps / 3), q=q)
        q = bb.add(Rx(2 * self.theta, eps=self.eps / 3), q=q)
        q = bb.add(Rz(pi(self.phi) / 2 - self.phi, eps=self.eps / 3), q=q)
        return {'q': q}

    def adjoint(self) -> 'SU2RotationGate':
        return SU2RotationGate(
            theta=self.theta,
            phi=-self.lambd,
            lambd=-self.phi,
            global_shift=-self.global_shift,
            eps=self.eps,
        )

    def is_symbolic(self) -> bool:
        return is_symbolic(self.theta, self.phi, self.lambd, self.global_shift)

    @classmethod
    def arbitrary(cls, ssa: 'SympySymbolAllocator') -> 'SU2RotationGate':
        """Return a parametrized arbitrary rotation for resource counting"""
        theta = ssa.new_symbol("theta")
        phi = ssa.new_symbol("phi")
        lambd = ssa.new_symbol("lambda")
        alpha = ssa.new_symbol("alpha")
        eps = ssa.new_symbol("eps")
        return cls(theta, phi, lambd, alpha, eps)

    def __str__(self):
        if self.is_symbolic():
            return f'SU_2({self.theta},{self.phi},{self.lambd},{self.global_shift})'
        return f'SU_2({self.theta:.2f},{self.phi:.2f},{self.lambd:.2f},{self.global_shift:.2f})'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            if self.is_symbolic():
                return Text(f'({self.theta},{self.phi},{self.lambd},{self.global_shift})')
            return Text(
                f'({self.theta:.2f},{self.phi:.2f},{self.lambd:.2f},{self.global_shift:.2f})'
            )

        return TextBox("SU2")


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
    bloq_cls=SU2RotationGate, examples=[_su2_rotation_gate, _hadamard, _t_gate]
)
