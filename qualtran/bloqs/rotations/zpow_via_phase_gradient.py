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
from typing import cast

import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    DecomposeTypeError,
    QBit,
    QFxp,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_alloc_free
from qualtran.symbolics import ceil, is_symbolic, log2, pi, SymbolicFloat, SymbolicInt


@frozen
class ZPowConstViaPhaseGradient(Bloq):
    r"""Apply an $Z**t$ on a qubit using a phase gradient state.

    This bloq implements a `Z**t` by conditionally loading `t/2` into a quantum
    register, conditioned on the qubit `q` (rotation target), and then adding
    this value to the phase gradient to get a phase kickback, and uncomputes the load.
    This controlled-load trick is taken from Ref. [2], Fig 2a.

    See :class:`PhaseGradientState` for details on phase gradients.

    It loads an approximation of `t/2` to `phase_grad_bitsize` bits,
    which is loaded using `phase_grad_bitsize` clean ancilla.

    The total Tofolli cost is `phase_grad_bitsize - 2`.


    Args:
        exponent: value of `t` to apply `Z**t`
        phase_grad_bitsize: number of qubits of the phase gradient state.

    Registers:
        q: qubit to apply rotation on.
        phase_grad: phase gradient state of type `QFxp` with `phase_grad_bitsize` fractional bits.

    References:
        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2001.09580).
        Haner et. al. 2020. Section 3: Components. "Integer addition" and Fig 2a.
    """
    exponent: SymbolicFloat
    phase_grad_bitsize: SymbolicInt

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(q=QBit(), phase_grad=self.phase_grad_dtype)

    @classmethod
    def from_precision(
        cls, exponent: SymbolicFloat, *, eps: SymbolicFloat
    ) -> 'ZPowConstViaPhaseGradient':
        r"""Apply a ZPow(t) with precision `eps`.

        Uses a phase gradient of size $\ceil(\log(2\pi / \epsilon)$.

        Args:
            exponent: value of `t` to apply `Z**t`
            eps: precision to approximate the unitary to.
        """
        b_grad = ceil(log2(2 * pi(eps) / eps))
        return cls(exponent, b_grad)

    @cached_property
    def phase_grad_dtype(self) -> QFxp:
        return QFxp(self.phase_grad_bitsize, self.phase_grad_bitsize)

    @property
    def _load_bloq(self) -> XorK:
        if is_symbolic(self.exponent) or is_symbolic(self.phase_grad_bitsize):
            return XorK(self.phase_grad_dtype, cast(sympy.Expr, self.exponent / 2))

        k_int = self.phase_grad_dtype.to_fixed_width_int(self.exponent / 2)
        return XorK(self.phase_grad_dtype, k_int)

    def build_composite_bloq(
        self, bb: BloqBuilder, q: Soquet, phase_grad: Soquet
    ) -> dict[str, SoquetT]:
        if is_symbolic(self.exponent):
            raise DecomposeTypeError(f"cannot decompose {self} with symbolic {self.exponent=}")

        # load the angle
        t = bb.allocate(dtype=self.phase_grad_dtype)
        q, t = bb.add(self._load_bloq.controlled(), ctrl=q, x=t)

        # add
        t, phase_grad = bb.add(
            AddIntoPhaseGrad(
                x_bitsize=self.phase_grad_bitsize, phase_bitsize=self.phase_grad_bitsize
            ),
            x=t,
            phase_grad=phase_grad,
        )

        # unload the angle
        q, t = bb.add(self._load_bloq.controlled(), ctrl=q, x=t)
        bb.free(t)

        return {'q': q, 'phase_grad': phase_grad}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            self._load_bloq.controlled(): 2,
            AddIntoPhaseGrad(self.phase_grad_bitsize, self.phase_grad_bitsize): 1,
        }

    def __str__(self) -> str:
        return f'ZPow({self.exponent})'


@bloq_example(generalizer=ignore_alloc_free)
def _zpow_const_via_phase_grad() -> ZPowConstViaPhaseGradient:
    zpow_const_via_phase_grad = ZPowConstViaPhaseGradient.from_precision(3 / 8, eps=1e-11)
    return zpow_const_via_phase_grad


@bloq_example(generalizer=ignore_alloc_free)
def _zpow_const_via_phase_grad_symb_prec() -> ZPowConstViaPhaseGradient:
    eps = sympy.symbols(r"\epsilon")
    zpow_const_via_phase_grad_symb_prec = ZPowConstViaPhaseGradient.from_precision(3 / 8, eps=eps)
    return zpow_const_via_phase_grad_symb_prec


@bloq_example(generalizer=ignore_alloc_free)
def _zpow_const_via_phase_grad_symb_angle() -> ZPowConstViaPhaseGradient:
    t = sympy.symbols(r"t")
    zpow_const_via_phase_grad_symb_angle = ZPowConstViaPhaseGradient.from_precision(t, eps=1e-11)
    return zpow_const_via_phase_grad_symb_angle
