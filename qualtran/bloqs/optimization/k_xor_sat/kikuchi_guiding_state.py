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
r"""Prepare the guiding state for a kXOR instance $\mathcal{I}$ with
Kikuchi parameter $\ell$.

References:
    [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
    Section 4.4.1, Theorem 4.15.
"""
from functools import cached_property

from attrs import evolve, field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.lists import HasDuplicates, SortInPlace
from qualtran.bloqs.basic_gates import Hadamard, OnEach, XGate
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
    SparseStatePreparationViaRotations,
)
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import ceil, is_symbolic, log2, pi, SymbolicFloat, SymbolicInt

from .kxor_instance import KXorInstance


@frozen
class SimpleGuidingState(PrepareOracle):
    r"""Prepare the guiding state for $\ell = k$.

    Given an kXOR instance $\mathcal{I}$, prepare the guiding state for
    parameter $\ell = k$ (i.e. $c = 1$), defined in Eq 134:
        $$
        |\phi\rangle
        \propto
        |\Gamma^k(\mathcal{A})\rangle
        =
        \frac{1}{\sqrt{\tilde{m}}}
        \sum_{S \in {[n] \choose k}} B_\mathcal{I}(S) |S\rangle
        $$

    Here, $\tilde{m}$ is the number of constraints in the input instance $\mathcal{I}$,
    and $\mathcal{A} = \sqrt{\frac{{n\choose k}}{\tilde{m}}} \mathcal{I}$.

    This bloq has a gate cost of $O(\tilde{m} \log n)$ (see Eq 142 and paragraph below).

    Args:
        inst: the kXOR instance $\mathcal{I}$.
        eps: Precision of the prepared state (defaults to 1e-6).

    Registers:
        S: a scope of $k$ variables, each in $[n]$.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Equation 134.
    """

    inst: KXorInstance
    eps: SymbolicFloat = field(default=1e-6, kw_only=True)

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(S=QAny(self.target_bitsize))

    @property
    def target_bitsize(self):
        """number of bits to represent a k-subset S"""
        return self.inst.k * self.inst.index_bitsize

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register('S', QAny(self.target_bitsize)),)

    @property
    def phasegrad_bitsize(self) -> SymbolicInt:
        return ceil(log2(2 * pi(self.eps) / self.eps))

    @property
    def _state_prep_bloq(self) -> SparseStatePreparationViaRotations:
        N = 2**self.target_bitsize

        if self.inst.is_symbolic():
            bloq = SparseStatePreparationViaRotations.from_n_coeffs(
                N, self.inst.num_unique_constraints, phase_bitsize=self.phasegrad_bitsize
            )
        else:
            assert not is_symbolic(self.inst.batched_scopes)

            bloq = SparseStatePreparationViaRotations.from_coefficient_map(
                N,
                {self.inst.scope_as_int(S): B_I for S, B_I in self.inst.batched_scopes},
                self.phasegrad_bitsize,
            )

        bloq = evolve(bloq, target_bitsize=self.target_bitsize)
        return bloq

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self._state_prep_bloq: 1}


@bloq_example
def _simple_guiding_state() -> SimpleGuidingState:
    from qualtran.bloqs.optimization.k_xor_sat import Constraint, KXorInstance

    inst = KXorInstance(
        n=4,
        k=2,
        constraints=(
            Constraint(S=(0, 1), b=1),
            Constraint(S=(2, 3), b=-1),
            Constraint(S=(1, 2), b=1),
        ),
    )
    simple_guiding_state = SimpleGuidingState(inst)
    return simple_guiding_state


@bloq_example
def _simple_guiding_state_symb() -> SimpleGuidingState:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat import KXorInstance

    n, m, k = sympy.symbols("n m k", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    simple_guiding_state_symb = SimpleGuidingState(inst)
    return simple_guiding_state_symb


_SIMPLE_GUIDING_STATE_DOC = BloqDocSpec(
    bloq_cls=SimpleGuidingState, examples=[_simple_guiding_state_symb, _simple_guiding_state]
)


@frozen
class ProbabilisticUncompute(Bloq):
    """Probabilistically uncompute a register using hadamards, and mark success in a flag qubit

    Apply hadamards to the register, and mark the flag conditioned on all input qubits being 0.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Eq. 129 and Eq. 130.
    """

    bitsize: SymbolicInt

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(q=QAny(self.bitsize), flag=QBit())

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q: 'Soquet', flag: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        q = bb.add(OnEach(self.bitsize, Hadamard()), q=q)
        q, flag = bb.add(
            XGate().controlled(CtrlSpec(qdtypes=QAny(self.bitsize), cvs=0)), ctrl=q, q=flag
        )
        return {'q': q, 'flag': flag}


@frozen
class GuidingState(PrepareOracle):
    r"""Prepare a guiding state for a kXOR instance with parameter $\ell$.

    Given an kXOR instance $\mathcal{I}$, and parameter $\ell$ (a multiple of $k$),
    we want to prepare the unit-length guiding state $|\mathbb{\Psi}\rangle$ (Eq 135):

        $$
        |\mathbb{\Psi}\rangle
        \propto
        |\Gamma^\ell(\mathcal{A})\rangle
        \propto
        \sum_{T \in {[n] \choose \ell}}
        \sum_{\{S_1, \ldots, S_c\} \in \text{Part}_k(T)}
        \left(
        \prod_{j = 1}^c B_{\mathcal{I}}(S)
        \right)
        |T\rangle
        $$

    This bloq prepares the state (Eq 136):
        $$ \beta |\mathbb{\Psi}\rangle |0^{\ell \log \ell + 3}\rangle
           + |\perp\rangle |1\rangle
        $$
    where $\beta \ge \Omega(1 / \ell^{\ell/2})$,
    and $\tilde{m}$ is the number of constraints in $\mathcal{I}$.

    This has a gate cost of $O(\ell \tilde{m} \log n)$.

    Args:
        inst: the kXOR instance $\mathcal{I}$.
        ell: the Kikuchi parameter $\ell$.
        amplitude_good_part: (optional) the amplitude $\beta$ of the guiding state $|\Psi\rangle$
            Defaults to $\beta = 0.99 / \ell^{\ell/2}$.
        eps: Precision of the prepared state (defaults to 1e-6).

    Registers:
        T: $\ell$ indices each in $[n]$.
        ancilla (RIGHT): (entangled) $\ell\log\ell+3$ ancilla qubits used for state preparation.
            The all zeros state of the ancilla is the good subspace.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Section 4.4.1 "Preparing the guiding state", Theorem 4.15. Eq 136.
    """

    inst: KXorInstance
    ell: SymbolicInt
    amplitude_good_part: SymbolicFloat = field(kw_only=True)
    eps: SymbolicFloat = field(default=1e-6, kw_only=True)

    @amplitude_good_part.default
    def _default_amplitude(self):
        return self.coeff_good

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('T', QAny(self.target_bitsize)),
                Register('ancilla', QAny(self.ancilla_bitsize)),
            ]
        )

    @property
    def target_bitsize(self) -> SymbolicInt:
        return self._index_dtype.num_qubits * self.ell

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        r"""total number of entangled ancilla.

        $\ell \log \ell$ for sorting, and 3 flag qubits.
        """
        return self.sort_ancilla_bitsize + 3

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register('T', QAny(self.target_bitsize)),)

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        return (Register('ancilla', QAny(self.ancilla_bitsize)),)

    @property
    def sort_ancilla_bitsize(self):
        r"""Number of entangled ancilla generated by the sorting algorithm.

        This is a sequence of $\ell$ numbers, each in $[\ell]$, therefore is $\ell \lceil \log \ell \rceil$.
        """
        logl = ceil(log2(self.ell))
        return self.ell * logl

    @property
    def c(self) -> SymbolicInt:
        r"""Value of $c = \ell / k$."""
        c = self.ell // self.inst.k
        try:
            return int(c)
        except TypeError:
            pass
        return c

    @property
    def simple_guiding_state(self) -> SimpleGuidingState:
        r"""The simple guiding state $|\phi\rangle$

        This is the simple guiding state defined in Eq. 142,
        which is proportional to $|\Gamma^k\rangle$ (Eq. 134).
        We will use $c$ copies of this state to prepare the required guiding state.

        References:
            [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
            Section 4.4.1 "Preparing the guiding state", Eq. 134.
        """
        return SimpleGuidingState(self.inst, eps=self.eps / self.c)

    @property
    def _index_dtype(self) -> QUInt:
        return QUInt(self.inst.index_bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', T: 'Soquet', ancilla: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        if is_symbolic(self.c):
            raise DecomposeTypeError(f"cannot decompose {self} with symbolic c=l/k={self.c}")

        partition_ancilla = Partition(
            self.ancilla_bitsize,
            (
                Register('ancilla', QAny(self.sort_ancilla_bitsize)),
                Register('flags', QBit(), shape=(3,)),
            ),
        )

        ancilla, [flag_duplicates, flag_uncompute, flag] = bb.add(partition_ancilla, x=ancilla)

        # Equation 144: |Phi> = |phi>^{\otimes c}
        partition_T_to_S = Partition(
            self.target_bitsize,
            (Register('S', dtype=QAny(self.simple_guiding_state.target_bitsize), shape=(self.c,)),),
        )
        S = bb.add(partition_T_to_S, x=T)
        for i in range(self.c):
            S[i] = bb.add(self.simple_guiding_state, S=S[i])
        T = bb.add(partition_T_to_S.adjoint(), S=S)

        # sort T using `l log l` entangled clean ancilla
        T, ancilla = bb.add_and_partition(
            SortInPlace(self.ell, self._index_dtype),
            [
                (Register('T', QAny(self.target_bitsize)), ['xs']),
                (Register('ancilla', QAny(self.sort_ancilla_bitsize)), ['pi']),
            ],
            T=T,
            ancilla=ancilla,
        )

        # mark if T has duplicates (i.e. not disjoint) (Eq 145)
        T, flag_duplicates = bb.add_and_partition(
            HasDuplicates(self.ell, self._index_dtype),
            [
                (Register('T', QAny(self.target_bitsize)), ['xs']),
                (Register('flag', QBit()), ['flag']),
            ],
            T=T,
            flag=flag_duplicates,
        )

        # probabilistically uncompute the sorting ancilla, and mark in a flag bit
        # note: flag is 0 for success (like syscall/c exit codes)
        ancilla, flag_uncompute = bb.add(
            ProbabilisticUncompute(self.sort_ancilla_bitsize), q=ancilla, flag=flag_uncompute
        )

        # compute the overall flag using OR, to obtain Eq 130.
        [flag_duplicates, flag_uncompute], flag = bb.add(
            MultiControlX(cvs=(0, 0)), controls=[flag_duplicates, flag_uncompute], target=flag
        )
        flag = bb.add(XGate(), q=flag)

        # join all the ancilla into a single bag of bits
        ancilla = bb.add(
            partition_ancilla.adjoint(),
            ancilla=ancilla,
            flags=[flag_duplicates, flag_uncompute, flag],
        )

        return {'T': T, 'ancilla': ancilla}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {
            self.simple_guiding_state: self.c,
            SortInPlace(self.ell, self._index_dtype): 1,
            HasDuplicates(self.ell, self._index_dtype): 1,
            ProbabilisticUncompute(self.sort_ancilla_bitsize): 1,
            MultiControlX(cvs=(0, 0)): 1,
            XGate(): 1,
        }

    @cached_property
    def coeff_good(self):
        """lower bound on beta, the coefficient of the good state.

        Sentence below Eq. 147.
        """
        return 0.99 / 2 ** (self.sort_ancilla_bitsize / 2)


@bloq_example
def _guiding_state() -> GuidingState:
    from qualtran.bloqs.optimization.k_xor_sat import Constraint, KXorInstance

    inst = KXorInstance(
        n=4,
        k=2,
        constraints=(
            Constraint(S=(0, 1), b=1),
            Constraint(S=(2, 3), b=-1),
            Constraint(S=(1, 2), b=1),
        ),
    )
    guiding_state = GuidingState(inst, ell=4)
    return guiding_state


@bloq_example
def _guiding_state_symb() -> GuidingState:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat import KXorInstance

    n, m, k = sympy.symbols("n m k", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    c = 2
    guiding_state_symb = GuidingState(inst, ell=c * inst.k)
    return guiding_state_symb


@bloq_example
def _guiding_state_symb_c() -> GuidingState:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat import KXorInstance

    n, m, c = sympy.symbols("n m c", positive=True, integer=True)
    k = sympy.symbols("k", positive=True, integer=True, even=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    guiding_state_symb_c = GuidingState(inst, ell=c * k)
    return guiding_state_symb_c


_GUIDING_STATE_DOC = BloqDocSpec(
    bloq_cls=GuidingState, examples=[_guiding_state_symb_c, _guiding_state_symb, _guiding_state]
)
