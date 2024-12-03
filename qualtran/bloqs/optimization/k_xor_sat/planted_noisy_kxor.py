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
from functools import cached_property
from typing import Optional

import numpy as np
import scipy
import sympy
from attrs import field, frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, SoquetT
from qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian import GuidedHamiltonian
from qualtran.bloqs.optimization.k_xor_sat.kikuchi_block_encoding import KikuchiHamiltonian
from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
    GuidingState,
    SimpleGuidingState,
)
from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import (
    ceil,
    HasLength,
    is_symbolic,
    ln,
    log2,
    prod,
    slen,
    ssqrt,
    SymbolicFloat,
    SymbolicInt,
)


def comb(n: SymbolicInt, k: SymbolicInt) -> SymbolicInt:
    """compute n choose k"""
    if is_symbolic(n, k):
        return sympy.binomial(n, k)
    return scipy.special.comb(n, k)


@frozen(kw_only=True)
class KikuchiAverageDegreeTheorem:
    """Compute the average degree of the Kikuchi matrix.

    The Alice theorem (Thm 2.21) guarantees this with high probability
    for random/planted instances.
    """

    n: SymbolicInt
    k: SymbolicInt
    ell: SymbolicInt

    @cached_property
    def delta(self) -> SymbolicFloat:
        """Eq 19"""
        n, k, l = self.n, self.k, self.ell
        term_1 = comb(k, k // 2)
        term_2 = comb(n - k, l - k // 2) / comb(n, l)
        return term_1 * term_2


@frozen(kw_only=True)
class AliceTheorem:
    r"""Alice theorem, E.g. Theorem 2.21.

    Consider a $k$XOR instance over $n$ variables and $m$ clauses, with Kikuchi parameter $\ell$.

    Assume:
    - $\ell \ge k/2$
    - $n \gg k \ell$

    For any parameters $\kappa \le 1$ and $0 < \epsilon \le \kappa/(2+\kappa)$,
    assume: $m$ satisfies
    $m/n \ge C_\kappa (n/\ell)^{(k-2)/2}$
    where
    $C_\kappa = 2(1+\epsilon)(1+\kappa) \kappa^{-2} {k \choose k/2}^{-1} \ln n$

    Then for a randomly drawn instance $\mathcal{I}$ (i.e. advantage 0),
    except with probability $3 n^{-\epsilon \ell}$, we are guaranteed:

    $\lambda_\max{\mathcal{K}_\ell(\mathcal{I})} \le \kappa d$
    where
    $d = \delta_{\ell,n,k} m$.

    Args:
        n: number of variables $n$
        k: number of variables per equation $k$
        ell: Kikuchi parameter
        kappa: parameter $\kappa$
        eps: parameter $\epsilon$

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
    """

    n: int
    k: int
    ell: int
    kappa: float
    eps: float = field()

    @eps.default
    def _default_max_eps(self):
        return self.kappa / (2 + self.kappa)

    def __attrs_post_init__(self):
        assert self.k % 2 == 0, "k must be even"
        assert self.ell % self.k == 0, "l must be a multiple of k"
        # assert self.n >= self.k * self.ell, "n must be atleast lk"
        assert 0 <= self.kappa <= 1
        assert 0 < self.eps <= self.kappa / (2 + self.kappa)

    @cached_property
    def C_kappa(self):
        """Eq 20 (right)"""
        term_1 = 2 * (1 + self.eps) * (1 + self.kappa) / self.kappa**2
        term_2 = comb(self.k, self.k // 2)
        term_3 = np.log(self.n)

        value = (term_1 / term_2) * term_3
        return value

    @cached_property
    def fail_prob(self):
        return 3 / self.n ** (self.eps * self.ell)

    @cached_property
    def min_m(self):
        """Eq 20 (left)"""
        m = self.C_kappa * (self.n / self.ell) ** (self.k // 2) * self.ell
        return ceil(m)


@frozen(kw_only=True)
class GuidingStateOverlapTheorem:
    r"""Lower-bound the overlap of the prepared guiding state with the eigenspace.

    This is an implementation of Theorem 2.40.

    Args:
        n: number of variables
        k: number of variables per constraint
        m_hat: total number of constraints
        ell: kikuchi parameter $\ell$
        zeta: the probability of picking a constraint for $\mathcal{I}_\text{guide}$.
        nu: parameter in $(0, .99]$.
        eps: parameter.
        rho: planted advantage.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Section 2.7, Theorem 2.40.
    """

    n: SymbolicInt
    k: SymbolicInt
    m_hat: SymbolicInt
    ell: SymbolicInt
    zeta: SymbolicFloat
    nu: SymbolicFloat
    eps: SymbolicFloat
    rho: SymbolicFloat

    @cached_property
    def part_k_l(self) -> SymbolicInt:
        ell, k = self.ell, self.k
        if is_symbolic(ell) or is_symbolic(k):
            return sympy.Symbol(r"Part_{k}(\ell)", positive=True, integer=True)
        return prod([comb(ell - i * k, k) for i in range(ell // k)])

    @cached_property
    def xi(self):
        r"""Eq 60 $\xi$"""
        term_1 = self.part_k_l
        term_2 = (self.rho * self.eps * self.nu) / (200 * self.ell * ln(self.n))
        term_3 = (self.rho**2 * self.zeta) ** (self.ell // self.k)
        return term_1 * term_2 * term_3

    @cached_property
    def overlap_probability(self) -> SymbolicFloat:
        term_2_base = self.m_hat / comb(self.n, self.k)
        term_2 = term_2_base ** (self.ell / self.k)
        return self.xi * term_2


@frozen
class PlantedNoisyKXOR(Bloq):
    r"""Algorithm for Planted Noisy kXOR.

    Problem (Problem 2.6 of Ref [1]):

    Given a noisy-kXOR instance $\hat{\mathcal{I}}$ which is drawn either:

    1. with planted advantage $\rho$, from $\tilde\mathcal{P}^{z}_{n, k}(m, \rho)$.
    2. at random, from $\tilde\mathcal{R}_{n, k}(m)$.

    output a single bit such that it is whp `1` in case 1, and `0` in case 2.

    Algorithm (Section 4.4, Theorem 4.18):
    We first split the instance into $\hat{\mathcal{I}} = \mathcal{I} \cup \mathcal{I}_\text{guide}$,
    by placing each constraint independently in $\mathcal{I}$ with prob. $1 - \zeta$,
    otherwise in $\mathcal{I}_\text{guide}$.
    $\zeta$ is picked to be $1 / \ln n$.

    Args:
        inst_guide: The subset of contraints $\mathcal{I}_\text{guide}$ for the guided state.
        inst_solve: The subset of constraints $\mathcal{I}$ for eigenvalue estimation.
        ell: Kikuchi parameter $\ell$.
        rho: the planted advantage $\rho$ in the planted case.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
    """

    inst_guide: KXorInstance
    inst_solve: KXorInstance
    ell: SymbolicInt
    rho: SymbolicFloat
    _guiding_state_overlap: Optional[SymbolicFloat] = field(kw_only=True, default=None)

    def __attrs_post_init__(self):
        k = self.inst_guide.k
        if not is_symbolic(k):
            assert k % 2 == 0, f"{k=} must be even"

        ell = self.ell
        if not is_symbolic(k, ell):
            assert ell % k == 0 and ell >= k, f"{ell=} must be a multiple of {k=}"

    @cached_property
    def signature(self) -> 'Signature':
        return self.guided_hamiltonian_bloq.signature

    @classmethod
    def from_inst(
        cls,
        inst: KXorInstance,
        ell: SymbolicInt,
        rho: SymbolicFloat,
        *,
        rng: np.random.Generator,
        zeta: Optional[SymbolicFloat] = None,
        guiding_state_overlap: Optional[SymbolicFloat] = None,
    ):
        if zeta is None:
            zeta = 1 / log2(inst.n)

        (use_for_guide,) = np.nonzero(np.atleast_1d(rng.random(inst.m) < zeta))
        inst_guide = inst.subset(tuple(use_for_guide))

        if is_symbolic(inst, use_for_guide):
            inst_solve = inst.subset(HasLength(inst.m - slen(use_for_guide)))
        else:
            mask = np.ones(inst.m)
            mask[np.array(use_for_guide)] = 0
            (rest,) = np.nonzero(mask)
            inst_solve = inst.subset(tuple(rest))

        return cls(
            inst_guide=inst_guide,
            inst_solve=inst_solve,
            ell=ell,
            rho=rho,
            guiding_state_overlap=guiding_state_overlap,
        )

    @cached_property
    def guiding_state_and_coefficient(self) -> tuple[PrepareOracle, SymbolicFloat]:
        r"""Return a bloq that prepares the guiding state, and its coefficient.

        If the bloq prepares $\beta |\Psi\rangle|0\rangle + |\perp\rangle|1\rangle$,
        then this will return $|\beta|$.

        The returned $\beta$ is an theoretical lower bound on the true value,
        and is correct for $1 - o(1)$ fraction of random instances.
        """
        if self.ell == self.inst_guide.k:
            return SimpleGuidingState(inst=self.inst_guide), 1
        bloq = GuidingState(inst=self.inst_guide, ell=self.ell)
        return bloq, bloq.amplitude_good_part

    @cached_property
    def guiding_state_overlap_guarantee(self) -> GuidingStateOverlapTheorem:
        """Invoke Theorem 2.40 to obtain a lower bound on the guiding state overlap.

        The below parameters are picked from Theorem 4.18, proof para 2.
        """
        n, k = self.inst_guide.n, self.inst_guide.k
        m_guide = self.inst_guide.m
        m_solve = self.inst_solve.m
        m_hat = m_guide + m_solve
        zeta = m_solve / m_hat
        return GuidingStateOverlapTheorem(
            n=n, k=k, ell=self.ell, m_hat=m_hat, zeta=zeta, nu=1 / ln(n), eps=0.005, rho=self.rho
        )

    @cached_property
    def guiding_state_overlap(self) -> SymbolicFloat:
        if self._guiding_state_overlap is not None:
            return self.guiding_state_overlap
        _, guiding_state_good_coeff = self.guiding_state_and_coefficient
        return guiding_state_good_coeff

    @cached_property
    def overlap(self) -> SymbolicFloat:
        # guiding state
        guiding_state_good_coeff = self.guiding_state_overlap

        # overlap of |\Gamma(A)> with the threshold eigenspace
        overlap_good_eigen = self.guiding_state_overlap_guarantee.overlap_probability**0.5

        # total overlap is the sqrt probability of the ancilla being 0,
        # and the state being in the >= lambda eigenspace.
        overlap = guiding_state_good_coeff * overlap_good_eigen

        return overlap

    @cached_property
    def degree_guarantee(self) -> KikuchiAverageDegreeTheorem:
        return KikuchiAverageDegreeTheorem(n=self.inst_solve.n, k=self.inst_solve.k, ell=self.ell)

    @cached_property
    def sparsity(self) -> SymbolicInt:
        """sparsity of the kikuchi matrix, $d$"""
        # d = \delta m
        d = self.degree_guarantee.delta * self.inst_solve.m
        if is_symbolic(d):
            return d  # type: ignore
        return ceil(d)

    @cached_property
    def hamiltonian(self) -> KikuchiHamiltonian:
        return KikuchiHamiltonian(
            inst=self.inst_solve, ell=self.ell, entry_bitsize=10, s=self.sparsity
        )

    @cached_property
    def guided_hamiltonian_bloq(self) -> GuidedHamiltonian:
        # Thm 4.18 proof para 2.
        # lambda = 0.995 rho d
        eigenvalue_threshold = 0.995 * self.rho * self.sparsity

        kappa = 0.99 * self.rho
        eps = 0.005

        # Thm 4.18 proof para 3
        # kappa' <= (1 - alpha) lambda
        # ==> alpha <= 1 - kappa'/lambda
        # we pick kappa' s.t. it satisfies the alice theorem for inst_solve.m
        # simple approximation: kappa' = kappa / sqrt(1-zeta)
        zeta = self.inst_guide.m / (self.inst_guide.m + self.inst_solve.m)
        kappa_prime = kappa / ssqrt(1 - zeta)
        alpha = 1 - kappa_prime / eigenvalue_threshold
        if not is_symbolic(alpha):
            assert alpha > 0, f"got negative {alpha=}"

        guiding_state, _ = self.guiding_state_and_coefficient

        return GuidedHamiltonian(
            self.hamiltonian,
            BlackBoxPrepare(guiding_state),
            lambd=eigenvalue_threshold,
            alpha=alpha,
            gamma=self.overlap,
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return bb.add_d(self.guided_hamiltonian_bloq, **soqs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {self.guided_hamiltonian_bloq: 1}


@bloq_example
def _solve_planted() -> PlantedNoisyKXOR:
    from qualtran.bloqs.optimization.k_xor_sat import KXorInstance

    rng = np.random.default_rng(42)
    n, m, k = 50, 1000, 4
    ell = k
    rho = 0.8

    inst = KXorInstance.random_instance(n, m, k, planted_advantage=rho, rng=rng)
    solve_planted = PlantedNoisyKXOR.from_inst(inst, ell=ell, rho=rho, zeta=0.1, rng=rng)
    return solve_planted


@bloq_example
def _solve_planted_symbolic() -> PlantedNoisyKXOR:
    from qualtran.bloqs.optimization.k_xor_sat import KXorInstance
    from qualtran.symbolics import HasLength

    n, m = sympy.symbols("n m", positive=True, integer=True)
    k = sympy.symbols("k", positive=True, integer=True, even=True)
    c = sympy.symbols("c", positive=True, integer=True)
    ell = c * k
    rho = sympy.Symbol(r"\rho", positive=True, real=True)

    inst = KXorInstance.symbolic(n, m, k)
    zeta = 1 / ln(n)
    solve_planted_symbolic = PlantedNoisyKXOR(
        inst_guide=inst.subset(HasLength((1 - zeta) * m)),
        inst_solve=inst.subset(HasLength(zeta * m)),
        ell=ell,
        rho=rho,
    )
    return solve_planted_symbolic


_PLANTED_NOISY_KXOR_DOC = BloqDocSpec(
    bloq_cls=PlantedNoisyKXOR, examples=[_solve_planted_symbolic, _solve_planted]
)
