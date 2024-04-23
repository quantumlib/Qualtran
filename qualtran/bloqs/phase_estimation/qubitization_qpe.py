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
from typing import Set, Tuple, TYPE_CHECKING

import attrs
import cirq

from qualtran import Bloq, bloq_example, GateWithRegisters, QFxp, Register, Signature
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.resource_counting.symbolic_counting_utils import (
    ceil,
    is_symbolic,
    log2,
    pi,
    SymbolicFloat,
    SymbolicInt,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class QubitizationQPE(GateWithRegisters):
    """Heisenberg limited phase estimation circuit for learning eigenphase of `walk`.

    The Bloq yields an OPTREE to construct Heisenberg limited phase estimation circuit
    for learning eigenphases of the `walk` operator with `m` bits of accuracy. The
    circuit is implemented as given in Fig.2 of Ref-1.

        ```
           ┌─────────┐                                     ┌─────────┐
      |0> -│         │-------------------------(0)---(0)---│         │---M--- [m1]:highest bit
           │         │                          |     |    │         │
      |0> -│         │----------------(0)---(0)-+-----+----│         │---M--- [m2]
           │CtrlState│                 |     |  |     |    │ QFT_inv │
      |0> -│  Prep   │-------(0)---(0)-+-----+--+-----+----│         │---M--- [m3]
           │         │        |     |  |     |  |     |    │         │
      |0> -│         │---@----+-----+--+-----+--+-----+----│         │---M--- [m4]:lowest bit
           └─────────┘   |    |     |  |     |  |     |    └─────────┘
    |Psi> ---------------W----R-W^2-R--R-W^4-R--R-W^8-R---------------------- |Psi>
        ```

    TODO: Note that there are slight differences between the Fig2 of the Ref[1] and the circuit
          implemented here. Further investigation is required to reconcile the difference.

    Args:
        walk: Bloq representing the Qubitization walk operator to run the phase estimation protocol
            on.
        m_bits: Bitsize of the phase register to be used during phase estimation.
        ctrl_state_prep: Bloq to prepare the control state on the phase register. Defaults to
            `OnEach(self.m_bits, Hadamard())`.
        qft_inv: Bloq to apply inverse QFT on the phase register. Defaults to
            `QFTTextBook(self.m_bits).adjoint()`


    Registers:
        qpe_reg: Control register of type `QFxp(self.m_bits, self.m_bits)` for phase estimation.
        target registers: All registers used in `self.unitary.signature`

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T
        Complexity](https://arxiv.org/abs/1805.03662) Fig. 2
    """

    walk: QubitizationWalkOperator
    m_bits: SymbolicInt
    ctrl_state_prep: Bloq = attrs.field()
    qft_inv: Bloq = attrs.field()

    @ctrl_state_prep.default
    def _default_state_prep(self):
        return LPResourceState(self.m_bits)

    @qft_inv.default
    def _default_inverse_qft(self):
        return QFTTextBook(self.m_bits, with_reverse=True).adjoint()

    def __attrs_post_init__(self):
        assert is_symbolic(self.m_bits) or (
            self.ctrl_state_prep.signature.n_qubits() == self.m_bits
        )

    @classmethod
    def from_standard_deviation_eps(cls, walk: QubitizationWalkOperator, eps: SymbolicFloat):
        r"""Estimate the phase $\phi$ with uncertainty in standard deviation bounded by $\epsilon$.

        The standard deviation of phase estimation using optimal resource states scales as the
        square of Holevo variance $\tan{\frac{\pi}{2^m}}$.
        This bound can be used to estimate the size of the phase register s.t. the estimated phase
        has a standard deviation of at-most $\epsilon$. See the class docstring for more details.

        ```
            m = ceil(log2(pi/eps))
        ```

        Args:
            walk: Walk operator to obtain phase estimate of.
            eps: Maximum standard deviation of the estimated phase.
        """
        m_bits = ceil(log2(pi(eps) / eps))
        return QubitizationQPE(walk=walk, m_bits=m_bits)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.walk.signature)

    @cached_property
    def phase_registers(self) -> Tuple[Register, ...]:
        return (Register('qpe_reg', QFxp(self.m_bits, self.m_bits)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.phase_registers, *self.target_registers])

    def pretty_name(self) -> str:
        return f'QubitizationQPE[{self.m_bits}]'

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        walk_regs = {reg.name: quregs[reg.name] for reg in self.walk.signature}
        reflect_regs = {reg.name: walk_regs[reg.name] for reg in self.walk.reflect.signature}

        reflect_controlled = self.walk.reflect.controlled(control_values=[0])
        walk_controlled = self.walk.controlled(control_values=[1])

        qpre_reg = quregs['qpe_reg']

        yield self.ctrl_state_prep.on(*qpre_reg)
        yield walk_controlled.on_registers(**walk_regs, control=qpre_reg[-1])
        walk = self.walk**2
        for i in range(self.m_bits - 2, -1, -1):
            yield reflect_controlled.on_registers(control=qpre_reg[i], **reflect_regs)
            yield walk.on_registers(**walk_regs)
            walk = walk**2
            yield reflect_controlled.on_registers(control=qpre_reg[i], **reflect_regs)
        yield self.qft_inv.on(*qpre_reg)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Assumes self.unitary is not fast forwardable.
        M = 2**self.m_bits
        return {
            (self.ctrl_state_prep, 1),
            (self.walk.controlled(control_values=[1]), 1),
            (self.walk.reflect.controlled(control_values=[0]), 2 * (self.m_bits - 1)),
            (self.walk, M - 2),
            (self.qft_inv, 1),
        }


@bloq_example
def _qubitization_qpe_hubbard_model_small() -> QubitizationQPE:
    import numpy as np

    from qualtran.bloqs.hubbard_model import get_walk_operator_for_hubbard_model
    from qualtran.bloqs.phase_estimation import QubitizationQPE

    x_dim, y_dim, t = 2, 2, 2
    mu = 4 * t
    walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, mu)

    algo_eps = t / 100
    N = x_dim * y_dim * 2
    qlambda = 2 * N * t + (N * mu) // 2
    qpe_eps = algo_eps / (qlambda * np.sqrt(2))
    qubitization_qpe_hubbard_model = QubitizationQPE.from_standard_deviation_eps(walk, qpe_eps)
    return qubitization_qpe_hubbard_model


@bloq_example
def _qubitization_qpe_hubbard_model_large() -> QubitizationQPE:
    import numpy as np

    from qualtran.bloqs.hubbard_model import get_walk_operator_for_hubbard_model
    from qualtran.bloqs.phase_estimation import QubitizationQPE

    x_dim, y_dim, t = 20, 20, 20
    mu = 4 * t
    walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, mu)

    algo_eps = t / 100
    N = x_dim * y_dim * 2
    qlambda = 2 * N * t + (N * mu) // 2
    qpe_eps = algo_eps / (qlambda * np.sqrt(2))
    qubitization_qpe_hubbard_model = QubitizationQPE.from_standard_deviation_eps(walk, qpe_eps)
    return qubitization_qpe_hubbard_model
