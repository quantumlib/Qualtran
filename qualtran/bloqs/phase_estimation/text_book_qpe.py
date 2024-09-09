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

from qualtran import Bloq, bloq_example, BloqDocSpec, GateWithRegisters, Register, Signature
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class TextbookQPE(GateWithRegisters):
    r"""Phase Estimation algorithm as presented in Chapter 5.2 of Neilson & Chuang

    The bloq implements the following phase estimation circuit, where `ctrl_state_prep` and
    `qft_inv` are configurable parameters.

    ```
           ┌─────────┐                              ┌─────────┐
      |0> -│         │-----------------------@------│         │---M--- [m1]:highest bit
           │         │                       |      │         │
      |0> -│         │-----------------@-----+------│         │---M--- [m2]
           │CtrlState│                 |     |      │ QFT_inv │
      |0> -│  Prep   │-----------@-----+-----+------│         │---M--- [m3]
           │         │           |     |     |      │         │
      |0> -│         │-----@-----+-----+-----+------│         │---M--- [m4]:lowest bit
           └─────────┘     |     |     |     |      └─────────┘
    |Psi> -----------------U-----U^2---U^4---U^8---------------------- |Psi>
    ```

    Note that the circuit measures $\varphi$ (in fixed point representation,
    so $0 \leq \varphi \leq 1$) s.t. $e^{i\phi}$ is an eigenvalue of $U$ where $\phi = 2\pi\varphi$
    is the estimated phase.

    The standard textbook version, as described in Ref[1], uses

    1. A uniform state preparation via hadamard on all control qubits for `CtrlStatePrep`
    2. A textbook QFT inverse algorithm, implemented in `QFTTextBook`, for `QFT_inv`

    Some useful properties of the phase estimation algorithm are given as follows -

    ## Cost of TextbookQPE
    The cost of textbook QPE on `m` control qubits is a sum of costs of

    1. **CtrlStatePrep** - This typically scales as $\mathcal{O}(m)$. For uniform state preparation,
            the cost is simply $m$ clifford gates.
    2. **Controlled-Us** - There are two cases:
        1. If the unitary is fast forwardable; i.e. cost of $U^n$ is independent of $n$, the cost
            of this step is simply $\mathcal{O}(m \text{ cost(C-U)})$
        2. If the unitary is not fast forwardable; the cost of this step is
            $\mathcal{O}((2 ^ {m} - 1) \text{cost(C-U)})$.
    4. **QFT_inv** - The textbook version of QFT uses $\mathcal{O}(m^2)$ rotations but this can be
            improved to $\mathcal{O}(m \log{m})$ using approximate QFT constructions.

    As seen above, in most cases the dominant cost of phase estimation comes from step 2.B, which
    depends exponentially on the number of control bits $m$.

    ## Choosing number of control bits - $m$.
    In the analysis below, we assume the textbook version of phase estimation where `CtrlStatePrep`
    is a uniform state preparation. One can obtain smaller values for $m$ when using different
    initial states for the control register, like then `LPResourceState` implemented in Qualtran.

    ### Dependence of $m$ using precision $n$ and success probability $\delta$ as the measure of uncertainty
    One way to think about the uncertainty in the obtained phase is to consider the problem where
    you wish to estimate the phase $\varphi$ upto $n$ bits of precision (i.e. with accuracy
    $2^{-n}$) with probability of success $1 - \delta$. In this setup, the expression of $m$ can be
    written as (following Eq 5.35 of Ref[1])

    $$
        m = n + \left\lceil\log_2\left({2 + \frac{2}{\delta}}\right)\right\rceil
    $$

    Setting the number of bits $m$ as per the expression above, we get

    $$
        Pr\left[|\tilde{\varphi} - \varphi| < \frac{1}{2^n}\right] \geq 1 - \delta
    $$

    Here $\varphi$ is the true phase and $\tilde{\varphi}$ is the estimated phase.

    `TextbookQPE(unitary, RectangularWindow.from_precision_and_delta(precision, delta))` method
    can be used to instantiate the Bloq with parameters $m$ and $\delta$ as described above.

    ### Dependence of $m$ using standard deviation $\epsilon$ as the measure of uncertainty
    A stronger way to bound the uncertainty in the obtained phase is to bound the variance of the
    estimator $\tilde{\varphi}$ by a given parameter $\epsilon$. Following the analysis in Ref[1,2],
    we can show that the variance for textbook phase estimation follows the Standard Quantum
    Limit(SQL) of

    $$
        \sigma[\phi] = 2\pi \sigma[\tilde{\varphi}] = 2\pi\sqrt{\text{Var}[\tilde{\varphi}]}
                                                    \leq\frac{\pi}{\sqrt{M}}=\frac{\pi}{\sqrt{2^m}}
    $$

    Therefore, to bound the standard deviation of the phase estimator $\tilde{\phi}$ by given parameter
    $\epsilon$, we set

    $$
        m = \left\lceil2\log_2 \left(\frac{\pi}{\epsilon}\right)\right\rceil
    $$

    `TextbookQPE(unitary, RectangularWindow.from_standard_deviation_eps(eps))` method can be
    used to instantiate the Bloq with parameter $\epsilon$ as described above.


    Args:
        unitary: Bloq representing the unitary to run the phase estimation protocol on.
        ctrl_state_prep: Bloq to prepare the control state on the phase register.
        qft_inv: Bloq to apply inverse QFT on the phase register. Defaults to
            `QFTTextBook(self.m_bits).adjoint()`


    Registers:
        qpe_reg: Control register of type `QFxp(self.m_bits, self.m_bits)` for phase estimation.
        target registers: All registers used in `self.unitary.signature`


    References:
        [Quantum Computation and Quantum Information: 10th Anniversary Edition,
        Nielsen & Chuang](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)
        Chapter 5.2

        [Entanglement-free Heisenberg-limited phase estimation](https://arxiv.org/abs/0709.2996)
    """

    unitary: Bloq
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
        return tuple(self.unitary.signature)

    @cached_property
    def phase_registers(self) -> Tuple[Register, ...]:
        return tuple(self.ctrl_state_prep.signature)

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.phase_registers, *self.target_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> Iterator[cirq.OP_TREE]:
        target_quregs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        unitary_op = self.unitary.on_registers(**target_quregs)
        phase_qubits = quregs['qpe_reg']

        yield self.ctrl_state_prep.on(*phase_qubits)
        for i, qbit in enumerate(phase_qubits[::-1]):
            yield cirq.pow(unitary_op.controlled_by(qbit), 2**i)
        yield self.qft_inv.on(*phase_qubits)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Assumes self.unitary is not fast forwardable.
        return {
            self.ctrl_state_prep: 1,
            self.unitary.controlled(): (2**self.m_bits) - 1,
            self.qft_inv: 1,
        }


@bloq_example
def _textbook_qpe_small() -> TextbookQPE:
    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE

    textbook_qpe_small = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))
    return textbook_qpe_small


@bloq_example
def _textbook_qpe_using_m_bits() -> TextbookQPE:
    import sympy

    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE

    theta = sympy.Symbol('theta')
    m_bits = sympy.Symbol('m')
    textbook_qpe_using_m_bits = TextbookQPE(
        ZPowGate(exponent=2 * theta), RectangularWindowState(m_bits)
    )
    return textbook_qpe_using_m_bits


@bloq_example
def _textbook_qpe_from_precision_and_delta() -> TextbookQPE:
    import sympy

    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE

    theta = sympy.Symbol('theta')
    precision, delta = sympy.symbols('n, delta')
    textbook_qpe_from_precision_and_delta = TextbookQPE(
        ZPowGate(exponent=2 * theta),
        RectangularWindowState.from_precision_and_delta(precision, delta),
    )
    return textbook_qpe_from_precision_and_delta


@bloq_example
def _textbook_qpe_from_standard_deviation_eps() -> TextbookQPE:
    import sympy

    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE

    theta = sympy.Symbol('theta')
    epsilon = sympy.symbols('epsilon')
    textbook_qpe_from_standard_deviation_eps = TextbookQPE(
        ZPowGate(exponent=2 * theta), RectangularWindowState.from_standard_deviation_eps(epsilon)
    )
    return textbook_qpe_from_standard_deviation_eps


_CC_TEXTBOOK_PHASE_ESTIMATION_DOC = BloqDocSpec(
    bloq_cls=TextbookQPE,
    examples=(
        _textbook_qpe_small,
        _textbook_qpe_using_m_bits,
        _textbook_qpe_from_standard_deviation_eps,
        _textbook_qpe_from_precision_and_delta,
    ),
)
