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
from typing import Optional, Tuple, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import frozen

from qualtran import bloq_example, BloqDocSpec, CompositeBloq, DecomposeTypeError, Register
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.symbolics import SymbolicFloat


@frozen
class ZPowGate(CirqGateAsBloqBase):
    r"""A gate that rotates around the Z axis of the Bloch sphere.

    The unitary matrix of `ZPowGate(exponent=t, global_shift=s)` is:
    $$
        e^{i \pi s t}
        \begin{bmatrix}
            1 & 0 \\
            0 & e^{i \pi t}
        \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i\pi t/2}$ vs the traditionally defined rotation matrices
    about the Pauli Z axis. See `Rz` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    Args:
        exponent: The t in gate**t. Determines how much the eigenvalues of
            the gate are phased by. For example, eigenvectors phased by -1
            when `gate**1` is applied will gain a relative phase of
            e^{i pi exponent} when `gate**exponent` is applied (relative to
            eigenvectors unaffected by `gate**1`).
        global_shift: Offsets the eigenvalues of the gate at exponent=1.
            In effect, this controls a global phase factor on the gate's
            unitary matrix. The factor for global_shift=s is:
                exp(i * pi * s * t)
        eps: precision for implementation of rotation.

    Registers:
        qubits: One-bit register.

    References:
        [Efficient synthesis of universal Repeat-Until-Success
        circuits](https://arxiv.org/abs/1404.5320). Offers a small improvement

        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """

    exponent: SymbolicFloat = 1.0
    global_shift: SymbolicFloat = 0.0
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        if isinstance(self.global_shift, sympy.Expr):
            raise TypeError(f"cirq.ZPowGate does not support symbolic {self.global_shift=}")
        return cirq.ZPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def __pow__(self, power):
        g = self.cirq_gate**power
        return ZPowGate(g.exponent, g.global_shift, self.eps)

    def adjoint(self) -> 'ZPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Z**{self.exponent}'


@bloq_example
def _z_pow() -> ZPowGate:
    z_pow = ZPowGate(exponent=0.123, eps=1e-8)
    return z_pow


_Z_POW_DOC = BloqDocSpec(bloq_cls=ZPowGate, examples=[_z_pow])


@frozen
class CZPowGate(CirqGateAsBloqBase):
    exponent: float = 1.0
    global_shift: float = 0.0
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.CZPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def _t_complexity_(self) -> 'TComplexity':
        if cirq.has_stabilizer_effect(self.cirq_gate):
            return TComplexity(clifford=1)
        return TComplexity(rotations=1)

    def __pow__(self, power):
        g = self.cirq_gate**power
        return CZPowGate(g.exponent, g.global_shift, self.eps)

    def adjoint(self) -> 'CZPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def __str__(self):
        return f'CZ**{self.exponent}'


@frozen
class XPowGate(CirqGateAsBloqBase):
    r"""A gate that rotates around the X axis of the Bloch sphere.

    The unitary matrix of `XPowGate(exponent=t, global_shift=s)` is:
    $$
    e^{i \pi t (s + 1/2)}
    \begin{bmatrix}
      \cos(\pi t /2) & -i \sin(\pi t /2) \\
      -i \sin(\pi t /2) & \cos(\pi t /2)
    \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli X axis. See `Rx` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    Args:
        exponent: The t in gate**t. Determines how much the eigenvalues of
            the gate are phased by. For example, eigenvectors phased by -1
            when `gate**1` is applied will gain a relative phase of
            e^{i pi exponent} when `gate**exponent` is applied (relative to
            eigenvectors unaffected by `gate**1`).
        global_shift: Offsets the eigenvalues of the gate at exponent=1.
            In effect, this controls a global phase factor on the gate's
            unitary matrix. The factor for global_shift=s is:
                exp(i * pi * s * t)
        eps: precision for implementation of rotation.

    Registers:
        q: One-bit register.

    References:
        [Efficient synthesis of universal Repeat-Until-Success
        circuits](https://arxiv.org/abs/1404.5320). Offers a small improvement

        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """
    exponent: Union[sympy.Expr, float] = 1.0
    global_shift: float = 0.0
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.XPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def adjoint(self) -> 'XPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'X**{self.exponent}'


@bloq_example
def _x_pow() -> XPowGate:
    x_pow = XPowGate(exponent=0.123, eps=1e-8)
    return x_pow


_X_POW_DOC = BloqDocSpec(bloq_cls=XPowGate, examples=[_x_pow])


@frozen
class YPowGate(CirqGateAsBloqBase):
    r"""A gate that rotates around the Y axis of the Bloch sphere.

    The unitary matrix of `YPowGate(exponent=t)` is:
    $$
        \begin{bmatrix}
            e^{i \pi t /2} \cos(\pi t /2) & - e^{i \pi t /2} \sin(\pi t /2) \\
            e^{i \pi t /2} \sin(\pi t /2) & e^{i \pi t /2} \cos(\pi t /2)
        \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli Y axis. See `Ry` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    Args:
        exponent: The t in gate**t. Determines how much the eigenvalues of
            the gate are phased by. For example, eigenvectors phased by -1
            when `gate**1` is applied will gain a relative phase of
            e^{i pi exponent} when `gate**exponent` is applied (relative to
            eigenvectors unaffected by `gate**1`).

        global_shift: Offsets the eigenvalues of the gate at exponent=1.
            In effect, this controls a global phase factor on the gate's
            unitary matrix. The factor for global_shift=s is:
                exp(i * pi * s * t)
        eps: precision for implementation of rotation.

    Registers:
        q: One-bit register.

    References:
        [Efficient synthesis of universal Repeat-Until-Success
        circuits](https://arxiv.org/abs/1404.5320). Offers a small improvement

        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """
    exponent: Union[sympy.Expr, float] = 1.0
    global_shift: float = 0.0
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.YPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def adjoint(self) -> 'YPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Y**{self.exponent}'


@bloq_example
def _y_pow() -> YPowGate:
    y_pow = YPowGate(exponent=0.123, eps=1e-8)
    return y_pow


_Y_POW_DOC = BloqDocSpec(bloq_cls=YPowGate, examples=[_y_pow])


@frozen
class Rz(CirqGateAsBloqBase):
    """Single-qubit Rz gate.

    Args:
        angle: Rotation angle in radians.
        eps: precision for implementation of rotation.

    Registers:
        q: One-bit register.

    References:
        [Efficient synthesis of universal Repeat-Until-Success
        circuits](https://arxiv.org/abs/1404.5320). Offers a small improvement

        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """

    angle: Union[sympy.Expr, float]
    eps: Union[sympy.Expr, float] = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rz(self.angle)

    def adjoint(self) -> 'Rz':
        return attrs.evolve(self, angle=-self.angle)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Rz({self.angle})'


@frozen
class Rx(CirqGateAsBloqBase):
    angle: Union[sympy.Expr, float]
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rx(self.angle)

    def adjoint(self) -> 'Rx':
        return attrs.evolve(self, angle=-self.angle)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Rx({self.angle})'


@frozen
class Ry(CirqGateAsBloqBase):
    angle: Union[sympy.Expr, float]
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.ry(self.angle)

    def adjoint(self) -> 'Ry':
        return attrs.evolve(self, angle=-self.angle)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Ry({self.angle})'


@bloq_example
def _rx() -> Rx:
    rx = Rx(angle=np.pi / 4.0)
    return rx


@bloq_example
def _ry() -> Ry:
    ry = Ry(angle=np.pi / 4.0)
    return ry


@bloq_example
def _rz() -> Rz:
    rz = Rz(angle=np.pi / 4.0)
    return rz
