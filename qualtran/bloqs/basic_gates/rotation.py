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
r"""Single-qubit rotation gates.

A single qubit's state can be mapped to the Bloch sphere. Rotations around the three
axes are generated by the pauli matrices:
$$
R_a(\theta) = \exp(-i \theta / 2 \sigma_a)
$$
where $\sigma_a$ is one of the Pauli $X, Y, Z$ operators.

Since global phase is often irrelevant, practitioners can use an alternative phase convention
and define single qubit gates as the real-valued power of a Pauli operator:
$$
(\sigma_a)^t
$$
which can be multiplied by $e^{-i \pi t / 2}$ to recover the $R_a(t \pi)$ matrix.

In Qualtran, we provide `ZPowGate` and `Rz` for the two respective phase conventions, as well
as analogues for the X and Y axes.

Global phase becomes a relevant, relative phase when forming controlled gates. Indeed, the
`ZPowGate` is a controlled `GlobalPhase` operation. Whereas `ZPowGate` and `Rz` are the
same up to global phase, their controlled versions `CZPowGate` and `CRz` are different operations
with different costs.

#### General References
 - [Quantum Computation and Quantum Information](https://doi.org/10.1017/CBO9780511976667).
   Nielsen and Chuang. 2010. Section 4.2
 - [Elementary gates for quantum computation](https://arxiv.org/abs/quant-ph/9503016).
   Barenco et. al. 1995.
"""


from functools import cached_property
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    CtrlSpec,
    DecomposeTypeError,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.symbolics import SymbolicFloat

if TYPE_CHECKING:
    from pennylane.operation import Operation
    from pennylane.wires import Wires


@frozen
class ZPowGate(CirqGateAsBloqBase):
    r"""Apply a power of the Pauli Z operator to a single qubit.

    Given  `exponent` $t$, the unitary matrix of this gate is:
    $$
    Z^t =
    \begin{bmatrix}
        1 & 0 \\
        0 & e^{i \pi t}
    \end{bmatrix}
    $$

    This is an atomic bloq in Qualtran. For many architectures, you will likely need to
    synthesize an arbitrary-angle rotation from a discrete gateset like Clifford+T. Please
    see the references for more information.

    #### Relationships
    This gate differs by a global phase from the $R_Z$ gate. `ZPowGate(t)` equals
    `Rz(angle=t*np.pi)` plus `GlobalPhase(t/2)`.

    This gate is the controlled version of a global phase gate. `ZPowGate(t)` equals
    `GlobalPhase(t).controlled()`.

    `exponent=1` corresponds to `ZGate`, `exponent=0.5` to `SGate`, and `exponent=0.25` to
    `TGate`.

    Args:
        exponent: The exponent t in Z^t.
        eps: The precision of the rotation. This parameter is for bookkeeping and does
            not affect e.g. the tensor representation of this gate. When synthesizing
            a rotation from a discrete gate set, you must fix a precision `eps`.

    Registers:
        q: The qubit.

    References:
        [Optimal ancilla-free Clifford+T approximation of z-rotations](https://arxiv.org/abs/1403.2975).
        Ross and Selinger. 2014.

        [Efficient synthesis of universal Repeat-Until-Success circuits](https://arxiv.org/abs/1404.5320).
        Bocharov et. al. 2014.
        Offers a small improvement in Cliffod+T synthesis.
    """

    exponent: SymbolicFloat = 1.0
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit())])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.ZPowGate(exponent=self.exponent, global_shift=0)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec)

        ctrl_bloq = CZPowGate(exponent=self.exponent, eps=self.eps)

        def add_ctrled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl_soq,) = ctrl_soqs
            ctrl_soq, q = bb.add(ctrl_bloq, q=np.array([ctrl_soq, in_soqs['q']]))
            return (ctrl_soq,), (q,)

        return ctrl_bloq, add_ctrled

    def __pow__(self, power):
        g = self.cirq_gate**power
        return ZPowGate(exponent=g.exponent, eps=self.eps)

    def adjoint(self) -> 'ZPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(f'Z^{self.exponent}')

    def __str__(self):
        return f'Z**{self.exponent}'


@bloq_example
def _z_pow() -> ZPowGate:
    z_pow = ZPowGate(exponent=0.123, eps=1e-8)
    return z_pow


_Z_POW_DOC = BloqDocSpec(bloq_cls=ZPowGate, examples=[_z_pow], call_graph_example=None)


@frozen
class CZPowGate(Bloq):
    r"""The controlled `ZPowGate`

    The unitary matrix of `CZPowGate(exponent=t)` is:
    $$
    C[Z^t] =
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i \pi t} \\
    \end{bmatrix}
    $$

    #### Relationships
    This gate has the same unitary as `Controlled(ZPowGate)`. CZPowGate(exponent=1) corresponds
    to a `CZ` gate.

    Args:
        exponent: The exponent t in Z^t.
        eps: The precision of the controlled rotation. This parameter is for bookkeeping and does
            not affect e.g. the tensor representation of this gate. When synthesizing
            a rotation from a discrete gate set, you must fix a precision `eps`.

    Registers:
        q: A shape=(2,) register of two qubits ordered. This is a symmetric gate.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567).
        Jones et. al. 2012. Figure 8.
    """

    exponent: SymbolicFloat = 1.0
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit(), shape=(2,))])

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> Dict[str, 'SoquetT']:
        from qualtran.bloqs.mcmt import And

        q1, q2 = q  # type: ignore
        (q1, q2), anc = bb.add(And(), ctrl=[q1, q2])
        anc = bb.add(ZPowGate(self.exponent, eps=self.eps), q=anc)
        (q1, q2) = bb.add(And().adjoint(), ctrl=[q1, q2], target=anc)
        return {'q': np.array([q1, q2])}

    def __pow__(self, power):
        return attrs.evolve(self, exponent=self.exponent * power)

    def adjoint(self) -> 'CZPowGate':
        return attrs.evolve(self, exponent=-self.exponent)

    def __str__(self):
        return f'CZ**{self.exponent}'


@bloq_example
def _cz_pow() -> CZPowGate:
    cz_pow = CZPowGate(exponent=0.123)
    return cz_pow


_CZ_POW_DOC = BloqDocSpec(bloq_cls=CZPowGate, examples=[_cz_pow])


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
        return TextBox(f'X^{self.exponent}')

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
        return TextBox(f'Y^{self.exponent}')

    def __str__(self):
        return f'Y**{self.exponent}'


@bloq_example
def _y_pow() -> YPowGate:
    y_pow = YPowGate(exponent=0.123, eps=1e-8)
    return y_pow


_Y_POW_DOC = BloqDocSpec(bloq_cls=YPowGate, examples=[_y_pow])


@frozen
class Rz(CirqGateAsBloqBase):
    r"""Apply a single-qubit Z rotation.

    Given `angle` $\theta$, the unitary matrix of this gate is:
    $$
    R_Z(\theta) = \exp(-i \frac{\theta}{2} Z) =
    \begin{bmatrix}
        e^{-i \theta/2} & 0 \\
        0 & e^{i \theta/2}
    \end{bmatrix}
    $$

    This is an atomic bloq in Qualtran. For many architectures, you will likely need to
    synthesize an arbitrary-angle rotation from a discrete gateset like Clifford+T. Please
    see the references for more information.

    #### Relationships
    This gate differs by a global phase from the `Z^t` gate. `Rz(a)` equals
    `ZPowGate(exponent=a/np.pi)` plus `GlobalPhase(-a/(2*np.pi))`.

    Args:
        angle: The rotation angle in radians.
        eps: The precision of the rotation. This parameter is for bookkeeping and does
            not affect e.g. the tensor representation of this gate. When synthesizing
            a rotation from a discrete gate set, you must fix a precision `eps`.

    Registers:
        q: One-bit register.

    References:
        [Optimal ancilla-free Clifford+T approximation of z-rotations](https://arxiv.org/abs/1403.2975).
        Ross and Selinger. 2014.

        [Efficient synthesis of universal Repeat-Until-Success circuits](https://arxiv.org/abs/1404.5320).
        Bocharov et. al. 2014.
        Offers a small improvement in Cliffod+T synthesis.
    """

    angle: SymbolicFloat
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit())])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rz(self.angle)

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import pennylane as qml

        return qml.RZ(phi=self.angle, wires=wires)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec)

        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            bloq=self,
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=CRz(self.angle, eps=self.eps),
            ctrl_reg_name='ctrl',
        )

    def adjoint(self) -> 'Rz':
        return attrs.evolve(self, angle=-self.angle)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def __str__(self):
        return f'Rz({self.angle})'


@bloq_example
def _rz() -> Rz:
    a = sympy.Symbol('a')
    rz = Rz(a)
    return rz


_RZ_DOC = BloqDocSpec(bloq_cls=Rz, examples=[_rz], call_graph_example=None)


@frozen
class CRz(Bloq):
    r"""A controlled Rz rotation.

    Given `angle` $\theta$, the unitary matrix of this gate is:
    $$
    C[R_Z(\theta)] =
    \begin{bmatrix}
        1 & & &  \\
        & 1 & &  \\
        & & e^{-i \theta/2} &  \\
        & & &  e^{i \theta/2}
    \end{bmatrix}
    $$

    Args:
        angle: The rotation angle in radians.
        eps: The precision of the rotation. This parameter is for bookkeeping and does
            not affect e.g. the tensor representation of this gate. When synthesizing
            a rotation from a discrete gate set, you must fix a precision `eps`.

    Registers:
        ctrl: Whether the rotation is active.
        q: The qubit on which we optionally perform the rotation.

    References:
        [Elementary gates for quantum computation](https://arxiv.org/abs/quant-ph/9503016).
        Barenco et al. 1995. Special case of Lemma 5.4.

        [Is Controlled(Rz(theta)) more expensive than Controlled(Z^t) on the surface code?](https://quantumcomputing.stackexchange.com/a/40012).
        Adam Zalcman. 2024.
    """

    angle: SymbolicFloat
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, q=1)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', q: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        from qualtran.bloqs.basic_gates import CNOT

        t = self.angle / np.pi
        q = bb.add(ZPowGate(t / 2, eps=self.eps / 2), q=q)
        ctrl, q = bb.add(CNOT(), ctrl=ctrl, target=q)
        q = bb.add(ZPowGate(-t / 2, eps=self.eps / 2), q=q)
        ctrl, q = bb.add(CNOT(), ctrl=ctrl, target=q)

        return {'ctrl': ctrl, 'q': q}

    def __str__(self):
        return f'CRz({self.angle})'


@bloq_example
def _crz() -> CRz:
    theta = sympy.Symbol(r'\theta')
    crz = CRz(angle=theta)
    return crz


_CRZ_DOC = BloqDocSpec(bloq_cls=CRz, examples=[_crz])


@frozen
class Rx(CirqGateAsBloqBase):
    angle: Union[sympy.Expr, float]
    eps: SymbolicFloat = 1e-11

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.rx(self.angle)

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import pennylane as qml

        return qml.RX(phi=self.angle, wires=wires)

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

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import pennylane as qml

        return qml.RY(phi=self.angle, wires=wires)

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
