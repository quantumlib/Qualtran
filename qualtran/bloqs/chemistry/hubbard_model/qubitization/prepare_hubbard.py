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
from typing import Iterator, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, BQUInt, QAny, Register, Signature
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.mcmt.and_bloq import MultiAnd
from qualtran.bloqs.mod_arithmetic import ModAddK
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.symbolics.math_funcs import acos, ssqrt

if TYPE_CHECKING:
    from qualtran.symbolics import SymbolicFloat


@attrs.frozen
class PrepareHubbard(PrepareOracle):
    r"""The PREPARE operation optimized for the 2D Hubbard model.

    In contrast to PREPARE for an arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` index for phases.

    `PrepareHubbard` uses $O(\log(N))$ T gates and $O(1)$ single-qubit rotations.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        t: coefficient for hopping terms in the Hubbard model hamiltonian.
        u: coefficient for single body Z term and two-body ZZ terms in the Hubbard model
            hamiltonian.

    Registers:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        junk: Temporary Work space.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 20.
    """

    x_dim: int
    y_dim: int
    t: float
    u: float

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('U', BQUInt(1, 2)),
            Register('V', BQUInt(1, 2)),
            Register('p_x', BQUInt((self.x_dim - 1).bit_length(), self.x_dim)),
            Register('p_y', BQUInt((self.y_dim - 1).bit_length(), self.y_dim)),
            Register('alpha', BQUInt(1, 2)),
            Register('q_x', BQUInt((self.x_dim - 1).bit_length(), self.x_dim)),
            Register('q_y', BQUInt((self.y_dim - 1).bit_length(), self.y_dim)),
            Register('beta', BQUInt(1, 2)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register('temp', QAny(2)),)

    @cached_property
    def l1_norm_of_coeffs(self) -> 'SymbolicFloat':
        # https://arxiv.org/abs/1805.03662v2 equation 60
        N = self.x_dim * self.y_dim * 2
        qlambda = 2 * N * self.t + (N * self.u) // 2
        return qlambda

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        temp = quregs['temp']

        N = self.x_dim * self.y_dim * 2
        yield cirq.Ry(rads=2 * acos(ssqrt(self.t * N / self.l1_norm_of_coeffs))).on(*V)
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)
        yield PrepareUniformSuperposition(self.x_dim).on_registers(controls=[], target=p_x)
        yield PrepareUniformSuperposition(self.y_dim).on_registers(controls=[], target=p_y)
        yield cirq.H.on_each(*temp)
        yield cirq.CNOT(*U, *V)
        yield cirq.X(*beta)
        yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]
        yield cirq.Circuit(cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])]))
        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)
        yield ModAddK(len(q_x), self.x_dim, add_val=1, cvs=[0, 0]).on(*U, *V, *q_x)
        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)

        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield MultiAnd(cvs=(0, 0, 1)).on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        yield CSwap.make_on(ctrl=and_target, x=[*p_x, *p_y, *alpha], y=[*q_x, *q_y, *beta])
        yield MultiAnd(cvs=(0, 0, 1)).adjoint().on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        context.qubit_manager.qfree([*and_anc, *and_target])


@bloq_example
def _prep_hubb() -> PrepareHubbard:
    x_dim = 4
    y_dim = 4
    t = 1.0
    u = 4.0 / t
    prep_hubb = PrepareHubbard(x_dim, y_dim, t=t, u=u)
    return prep_hubb


_PREPARE_HUBBARD = BloqDocSpec(bloq_cls=PrepareHubbard, examples=(_prep_hubb,))
