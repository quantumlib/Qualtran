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
from typing import Iterator, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CtrlSpec,
    GateWithRegisters,
    Register,
    Signature,
)
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


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
    ctrl_state_prep: QPEWindowStateBase
    qft_inv: Bloq = attrs.field()

    @qft_inv.default
    def _default_inverse_qft(self):
        return QFTTextBook(self.m_bits, with_reverse=True).adjoint()

    def __attrs_post_init__(self):
        assert is_symbolic(self.m_bits) or (self.qft_inv.signature.n_qubits() == self.m_bits)

    @cached_property
    def m_bits(self) -> SymbolicInt:
        return self.ctrl_state_prep.m_bits

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.walk.signature)

    @cached_property
    def phase_registers(self) -> Tuple[Register, ...]:
        return tuple(self.ctrl_state_prep.signature)

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.phase_registers, *self.target_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> Iterator[cirq.OP_TREE]:
        walk_regs = {reg.name: quregs[reg.name] for reg in self.walk.signature}
        reflect_regs = {reg.name: walk_regs[reg.name] for reg in self.walk.reflect.signature}

        reflect_controlled = self.walk.reflect.controlled(ctrl_spec=CtrlSpec(cvs=0))
        walk_controlled = self.walk.controlled()

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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Assumes self.unitary is not fast forwardable.
        M = 2**self.m_bits
        return {
            self.ctrl_state_prep: 1,
            self.walk.controlled(): 1,
            self.walk.reflect.controlled(ctrl_spec=CtrlSpec(cvs=0)): 2 * (self.m_bits - 1),
            self.walk: M - 2,
            self.qft_inv: 1,
        }

    def __str__(self) -> str:
        return f'QubitizationQPE[{self.m_bits}]'


@bloq_example
def _qubitization_qpe_hubbard_model_small() -> QubitizationQPE:
    import numpy as np

    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

    x_dim, y_dim, t = 2, 2, 2
    u = 4 * t
    walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, u)

    algo_eps = t / 100
    N = x_dim * y_dim * 2
    qlambda = 2 * N * t + (N * u) // 2
    qpe_eps = algo_eps / (qlambda * np.sqrt(2))
    qubitization_qpe_hubbard_model_small = QubitizationQPE(
        walk, LPResourceState.from_standard_deviation_eps(qpe_eps)
    )
    return qubitization_qpe_hubbard_model_small


@bloq_example
def _qubitization_qpe_hubbard_model_large() -> QubitizationQPE:
    import numpy as np

    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

    x_dim, y_dim, t = 20, 20, 20
    u = 4 * t
    walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, u)

    algo_eps = t / 100
    N = x_dim * y_dim * 2
    qlambda = 2 * N * t + (N * u) // 2
    qpe_eps = algo_eps / (qlambda * np.sqrt(2))
    qubitization_qpe_hubbard_model_large = QubitizationQPE(
        walk, LPResourceState.from_standard_deviation_eps(qpe_eps)
    )
    return qubitization_qpe_hubbard_model_large


@bloq_example
def _qubitization_qpe_chem_thc() -> QubitizationQPE:
    from openfermion.resource_estimates.utils import QI

    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.thc.walk_operator import get_walk_operator_for_thc_ham
    from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

    # Li et al parameters from openfermion.resource_estimates.thc.compute_cost_thc_test
    num_spinorb = 152
    num_bits_state_prep = 10
    num_bits_rot = 20
    thc_dim = 450
    num_spat = num_spinorb // 2
    qroam_blocking_factor = np.power(2, QI(thc_dim + num_spat)[0])
    t_l, eta, zeta = build_random_test_integrals(thc_dim, num_spinorb // 2, seed=7)
    walk = get_walk_operator_for_thc_ham(
        t_l,
        eta,
        zeta,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_theta=num_bits_rot,
        kr1=qroam_blocking_factor,
        kr2=qroam_blocking_factor,
    )

    algo_eps = 0.0016
    qpe_eps = algo_eps / (walk.block_encoding.alpha * 2**0.5)
    qubitization_qpe_chem_thc = QubitizationQPE(
        walk, LPResourceState.from_standard_deviation_eps(qpe_eps)
    )
    return qubitization_qpe_chem_thc


@bloq_example
def _qubitization_qpe_sparse_chem() -> QubitizationQPE:
    import numpy as np

    from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.sparse.walk_operator import get_walk_operator_for_sparse_chem_ham
    from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

    num_spatial = 6
    tpq, eris = build_random_test_integrals(num_spatial // 2, seed=7)
    walk = get_walk_operator_for_sparse_chem_ham(
        tpq, eris, num_bits_rot_aa=8, num_bits_state_prep=16
    )

    algo_eps = 0.0016
    qlambda = np.sum(np.abs(tpq)) + 0.5 * np.sum(np.abs(eris))
    qpe_eps = algo_eps / (qlambda * np.sqrt(2))
    qubitization_qpe_sparse_chem = QubitizationQPE(
        walk, LPResourceState.from_standard_deviation_eps(qpe_eps)
    )
    return qubitization_qpe_sparse_chem


_QUBITIZATION_QPE_DOC = BloqDocSpec(
    bloq_cls=QubitizationQPE,
    examples=(
        _qubitization_qpe_hubbard_model_small,
        _qubitization_qpe_sparse_chem,
        _qubitization_qpe_chem_thc,
    ),
)
