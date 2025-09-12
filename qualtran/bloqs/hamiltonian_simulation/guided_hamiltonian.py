#  Copyright 2025 Google LLC
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
from collections import Counter
from functools import cached_property

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, Signature
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.phase_estimation import KaiserWindowState, QubitizationQPE
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.bloqs.qubitization import QubitizationWalkOperator
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import ceil, ln, log2, pi, SymbolicFloat, SymbolicInt


@frozen
class GuidedHamiltonianPhaseEstimation(Bloq):
    r"""Phase estimation subroutine for the Guided Hamiltonian Problem.

    Implements the algorithm $U_\text{PE}$ used in the Guided Hamiltonian Problem.
    See the bloq `GuidedHamiltonian` for the full problem definition.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Section 4.2 "Guided Sparse Hamiltonian Problem".
    """

    hamiltonian: BlockEncoding
    guiding_state: BlackBoxPrepare
    precision: SymbolicFloat
    fail_prob: SymbolicFloat

    def __attrs_post_init__(self):
        assert (
            self.hamiltonian.resource_bitsize == 0
        ), "block encoding with resource regs not supported"

        assert self.hamiltonian.system_bitsize == self.guiding_state.selection_bitsize

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            phase_estimate=self.qpe_window_state.m_register.dtype,
            system=QAny(self.hamiltonian.system_bitsize),
            walk_ancilla=QAny(self.hamiltonian.ancilla_bitsize),
            guide_ancilla=QAny(self.guiding_state.junk_bitsize),
        )

    @cached_property
    def walk_operator(self) -> QubitizationWalkOperator:
        # TODO https://github.com/quantumlib/Qualtran/issues/1266
        #      QubitizationWalkOperator currently accepts only specific
        #      block-encodings, but should be generalized.
        #      For now we ignore the type here.
        return QubitizationWalkOperator(self.hamiltonian)  # type: ignore

    @cached_property
    def qpe_window_state(self) -> QPEWindowStateBase:
        """Kaiser Window state.
        Computes a slightly larger value for a simpler expression.
        https://arxiv.org/abs/2209.13581, Eq D14, D15
        """
        eps, delta = self.precision, self.fail_prob

        alpha = ln(1 / delta) / pi(delta)

        N = (1 / eps) * ln(1 / delta)
        m_bits = ceil(log2(N))
        return KaiserWindowState(bitsize=m_bits, alpha=alpha)

    @cached_property
    def qpe_bloq(self) -> QubitizationQPE:
        return QubitizationQPE(self.walk_operator, self.qpe_window_state)  # type: ignore

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self.guiding_state: 1, self.qpe_bloq: 1}


@bloq_example
def _guided_phase_estimate_symb() -> GuidedHamiltonianPhaseEstimation:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat import GuidingState, KikuchiHamiltonian, KXorInstance
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.symbolics import ceil, log2

    n, k, c = sympy.symbols("n k c", positive=True, integer=True)
    m_guide, m_solve = sympy.symbols("m_1 m_2", positive=True, integer=True)

    inst_guide = KXorInstance.symbolic(n, m_guide, k, max_rhs=2)
    inst_solve = KXorInstance.symbolic(n, m_solve, k, max_rhs=2)
    l = c * k
    s = l * ceil(log2(n))

    Psi = GuidingState(inst_guide, l)
    H = KikuchiHamiltonian(inst_solve, c * k, s)

    eps, delta = sympy.symbols(r"\epsilon_{PE} \delta_{PE}", positive=True, real=True)
    guided_phase_estimate_symb = GuidedHamiltonianPhaseEstimation(
        H, BlackBoxPrepare(Psi), eps, delta
    )

    return guided_phase_estimate_symb


_GUIDED_HAMILTONIAN_PHASE_ESTIMATION_DOC = BloqDocSpec(
    bloq_cls=GuidedHamiltonianPhaseEstimation, examples=[_guided_phase_estimate_symb]
)


@frozen
class GuidedHamiltonian(Bloq):
    r"""Solve the Guided (Sparse) Hamiltonian Problem.

    Definition 4.8 (modified to accept any block-encoding):
    In the Guided Hamiltonian problem we are given the following as input:

    1. A $(\sqrt{2} s, \cdot, 0)$-block-encoding of a Hamiltonian $H$ such that $\|H\|_\max \le s$.
    2. A unitary program that prepares $|\Psi\rangle|0^A\rangle$.
    3. Parameters $\lambda \in [-\Lambda, \Lambda]$, $\alpha \in (0, 1)$, $\gamma \in (0, 1]$.

    and we should output

    - YES (1) if $\| \Pi_{\ge \lambda} (H) |\Psi\rangle \| \ge \gamma$
    - NO (0) if $\|H\| \le (1 - \alpha) \lambda$

    Note that the above drops the sparsity requirement, and accepts any
    $(\alpha_H, \cdot, \cdot)$-block-encoding of $H$.
    In the sparse Hamiltonian case, $\alpha_H = s$ (where $s$ is the sparsity).

    Algorithm (Theorem 4.9):
        This uses phase estimation on the block-encoding of $e^{iHt}$, and then uses
        amplitude amplification to increase the success probability to $1 - o(1)$.

    We instead directly do phase-estimation on the qubitized walk operator for $H$.

    Args:
        hamiltonian: the block-encoding of $H$
        guiding_state: the unitary that prepares $|\Psi\rangle$
        lambd: parameter $\lambda$
        alpha: parameter $\alpha$
        gamma: parameter $\gamma$

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Section 4.2 "Guided Sparse Hamiltonian Problem".
    """

    hamiltonian: BlockEncoding
    guiding_state: BlackBoxPrepare
    lambd: SymbolicFloat
    alpha: SymbolicFloat
    gamma: SymbolicFloat

    def __attrs_post_init__(self):
        assert self.hamiltonian.resource_bitsize == 0, "resource not supported"
        assert (
            self.hamiltonian.system_bitsize == self.guiding_state.selection_bitsize
        ), "system registers must match"

        assert self.signature == self.qpe_bloq.signature

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            phase_estimate=self.qpe_bloq.qpe_window_state.m_register.dtype,
            system=QAny(self.hamiltonian.system_bitsize),
            walk_ancilla=QAny(self.hamiltonian.ancilla_bitsize),
            guide_ancilla=QAny(self.guiding_state.junk_bitsize),
        )

    @cached_property
    def qpe_precision(self) -> SymbolicFloat:
        r"""The precision for phase estimation.

        Page 31, Eq 100 of the reference gives the precision value for estimating phases
        of $e^{iHt}$ with $t = \pi/(2s)$. But this bloq does phase estimation directly
        on the walk operator, with eigenphases $e^{-i \arccos(H/s)}$.

        To bound this, consider the two eigenvalues that are to be distinguished:
        $\lambda$ and $(1 - \alpha)\lambda$. We can bound the difference in estimated phases as

        $$
            |\arccos(\lambda / s) - \arccos((1-\alpha)\lambda / s)|
            \le \frac{\alpha \lambda}{s} \frac{1}{1 - ((1 - \alpha)\lambda / s)^2}
        $$

        As we know $\|H\| \le s/\sqrt{2}$, it means $\lambda/s \le 1/\sqrt{2}$,
        therefore the second term is at most $\sqrt{2}$.

        In the sparse encoding case, we can increase the sparsity to $\sqrt{2} s$
        when block-encoding the input, to ensure that we have an epsilon bound of
        $\alpha \lambda / s$.
        """
        return self.alpha * self.lambd / self.hamiltonian.alpha

    @cached_property
    def qpe_fail_prob(self) -> SymbolicFloat:
        """Page 31, above Eq 104."""
        return self.gamma**3

    @cached_property
    def n_rounds_amplification(self) -> SymbolicInt:
        return ceil(1 / self.gamma)

    @cached_property
    def qpe_bloq(self) -> GuidedHamiltonianPhaseEstimation:
        return GuidedHamiltonianPhaseEstimation(
            hamiltonian=self.hamiltonian,
            guiding_state=self.guiding_state,
            precision=self.qpe_precision,
            fail_prob=self.qpe_fail_prob,
        )

    @cached_property
    def _refl_guide_ancilla(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare.reflection_around_zero(
            bitsizes=[self.guiding_state.junk_bitsize], global_phase=-1
        )

    @cached_property
    def _refl_all(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(PrepareIdentity(tuple(self.signature)), global_phase=-1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        counts = Counter[Bloq]()

        # prepare the initial state
        counts[self.qpe_bloq] += 1

        # reflect about the ancilla being all 0
        counts[self._refl_guide_ancilla] += self.n_rounds_amplification

        # reflect about the prepared state
        counts[self.qpe_bloq.adjoint()] += self.n_rounds_amplification
        counts[self._refl_all] += self.n_rounds_amplification
        counts[self.qpe_bloq] += self.n_rounds_amplification

        return counts


@bloq_example
def _guided_hamiltonian_symb() -> GuidedHamiltonian:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat import GuidingState, KikuchiHamiltonian, KXorInstance
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.symbolics import ceil, log2

    n, k, m, c = sympy.symbols("n k m c", positive=True, integer=True)
    zeta = sympy.symbols(r"\zeta", positive=True)

    inst_guide = KXorInstance.symbolic(n, (1 - zeta) * m, k, max_rhs=2)
    inst_solve = KXorInstance.symbolic(n, zeta * m, k, max_rhs=2)
    l = c * k
    s = l * ceil(log2(n))

    Psi = GuidingState(inst_guide, l)
    H = KikuchiHamiltonian(inst_solve, c * k, s)

    lambd, alpha, gamma = sympy.symbols(r"\lambda \alpha \gamma", positive=True, real=True)
    guided_hamiltonian_symb = GuidedHamiltonian(H, BlackBoxPrepare(Psi), lambd, alpha, gamma)
    return guided_hamiltonian_symb


_GUIDED_HAMILTONIAN_DOC = BloqDocSpec(
    bloq_cls=GuidedHamiltonian, examples=[_guided_hamiltonian_symb]
)
