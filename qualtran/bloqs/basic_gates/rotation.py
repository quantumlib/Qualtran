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
from typing import Protocol

import cirq
import numpy as np
from attrs import frozen

from qualtran import bloq_example
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


class _HasEps(Protocol):
    """Protocol for typing `RotationBloq` base class mixin that has accuracy specified as eps."""

    eps: float


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
        circuits](https://arxiv.org/abs/1404.5320), which offers a small improvement
        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """

    exponent: float = 1.0
    global_shift: float = 0.0
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.ZPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def __pow__(self, power):
        g = self.cirq_gate**power
        return ZPowGate(g.exponent, g.global_shift, self.eps)


@frozen
class CZPowGate(CirqGateAsBloqBase):
    exponent: float = 1.0
    global_shift: float = 0.0
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.CZPowGate(exponent=self.exponent, global_shift=self.global_shift)

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(rotations=1)

    def __pow__(self, power):
        g = self.cirq_gate**power
        return CZPowGate(g.exponent, g.global_shift, self.eps)


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
        circuits](https://arxiv.org/abs/1404.5320), which offers a small improvement
        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """
    exponent: float = 1.0
    global_shift: float = 0.0
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.XPowGate(exponent=self.exponent, global_shift=self.global_shift)


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
        circuits](https://arxiv.org/abs/1404.5320), which offers a small improvement
        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """
    exponent: float = 1.0
    global_shift: float = 0.0
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.YPowGate(exponent=self.exponent, global_shift=self.global_shift)


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
        circuits](https://arxiv.org/abs/1404.5320), which offers a small improvement
        [Optimal ancilla-free Clifford+T approximation
        of z-rotations](https://arxiv.org/pdf/1403.2975.pdf).
    """

    angle: float
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rz(self.angle)


@frozen
class Rx(CirqGateAsBloqBase):
    angle: float
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rx(self.angle)


@frozen
class Ry(CirqGateAsBloqBase):
    angle: float
    eps: float = 1e-11

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.ry(self.angle)


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
