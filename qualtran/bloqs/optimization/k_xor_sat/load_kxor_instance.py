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
r"""We define three oracles that load a kXOR instance, which are used in the algorithm.

We are given a kXOR instance $\mathcal{I}$ of $n$ variables,
with $\bar{m}$ unique scopes $\{U_j | j \in [\bar{m}]\}$.
We provide oracles to:
1. `LoadConstraintScopes`: Given $j \in [\bar{m}]$, compute $U_j$.
2. `LoadUniqueScopeIndex`: Given $U_j$, compute $j \in [\bar{m}]$
3. `PRGAUniqueConstraintRHS` Given $j$, apply $Rx(arccos(\sqrt{B_\mathcal{I}(S)/M}))$ on a target qubit.
(for an appropriate normalization $M$).


The first two oracles are independent of the RHS.
All these oracles can output arbitrary values for invalid inputs.

References:
    [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
    Notation 2.24 for $B_\mathcal{I}$.
    Theorem 4.17, proof para 2 for $U_j$.
"""
from functools import cached_property
from typing import Counter, Sequence, Union

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    QFxp,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import EqualsAConstant, LessThanConstant
from qualtran.bloqs.basic_gates import Hadamard, SGate
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.data_loading import QROM
from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import ceil, HasLength, is_symbolic, is_zero, log2, SymbolicInt

from .kxor_instance import KXorInstance


@frozen
class LoadConstraintScopes(Bloq):
    r"""Given an index $j$, load the scope of the $j$-th unique constraint.

    Given a $k$-XOR-SAT instance `inst` with $n$ variables and $m$ constraints.
    Assuming `inst` has $\bar{m}$ unique constraints, we define $U_j \in {[n] \choose k}$
    for $j \in [\bar{m}]$ as the $j$-th unique constraint scope.

    The scopes are loaded using a QROM.

    If the input contains an invalid index, then any arbitrary value can be output.

    Registers:
        j: a number in [\bar{m}]
        U (RIGHT): $j$-th unique scope
        ancilla (RIGHT): entangled intermediate qubits, to be uncomputed by the adjoint.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Theorem 4.17, proof para 2.
    """

    inst: KXorInstance

    @cached_property
    def signature(self) -> 'Signature':
        registers: list[Register] = [
            Register('j', self.m_dtype),
            Register('U', QAny(self.scope_bitsize), side=Side.RIGHT),
        ]

        if not is_zero(self.ancilla_bitsize):
            registers.append(Register('ancilla', QAny(self.ancilla_bitsize), side=Side.RIGHT))

        return Signature(registers)

    @cached_property
    def scope_bitsize(self) -> SymbolicInt:
        """total number of bits to store `k` indices in `[n]`."""
        return self.inst.k * self.inst.index_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        """ancillas used by the underlying QRO(A)M"""
        return 0

    @cached_property
    def m_dtype(self):
        r"""number of bits to store $j \in [\bar{m}]$."""
        m = self.inst.num_unique_constraints
        bitsize = ceil(log2(m))
        return BQUInt(bitsize, m)

    @cached_property
    def _qrom_bloq(self) -> QROM:
        # TODO use QROAMClean?

        if self.inst.is_symbolic():
            return QROM.build_from_bitsize(self.inst.num_unique_constraints, self.scope_bitsize)

        assert isinstance(self.inst.batched_scopes, tuple)
        scopes = np.array([S for S, _ in self.inst.batched_scopes], dtype=int)
        assert scopes.shape == (self.inst.num_unique_constraints, self.inst.k)
        return QROM.build_from_data(
            *scopes.T, target_bitsizes=(self.inst.index_bitsize,) * self.inst.k
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', j: 'Soquet') -> dict[str, 'SoquetT']:
        if self.inst.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        targets = {
            f'target{i}_': bb.allocate(self.inst.index_bitsize) for i in range(int(self.inst.k))
        }
        targets = bb.add_d(self._qrom_bloq, selection=j, **targets)
        j = targets.pop('selection')

        U = bb.add(
            Partition(self.scope_bitsize, self._qrom_bloq.target_registers).adjoint(), **targets
        )
        return {'j': j, 'U': U}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {self._qrom_bloq: 1}


@bloq_example
def _load_scopes() -> LoadConstraintScopes:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import Constraint, KXorInstance

    inst = KXorInstance(
        n=6,
        k=4,
        constraints=(
            Constraint(S=(0, 1, 2, 3), b=1),
            Constraint(S=(0, 1, 4, 5), b=-1),
            Constraint(S=(1, 2, 4, 5), b=1),
            Constraint(S=(0, 3, 4, 5), b=1),
            Constraint(S=(2, 3, 4, 5), b=1),
            Constraint(S=(0, 1, 2, 3), b=1),
            Constraint(S=(0, 3, 4, 5), b=1),
            Constraint(S=(2, 3, 4, 5), b=1),
        ),
    )
    load_scopes = LoadConstraintScopes(inst)
    return load_scopes


@bloq_example
def _load_scopes_symb() -> LoadConstraintScopes:
    import sympy

    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance

    n, m, k = sympy.symbols("n m k", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    load_scopes_symb = LoadConstraintScopes(inst)
    return load_scopes_symb


_LOAD_INSTANCE_DOC = BloqDocSpec(
    bloq_cls=LoadConstraintScopes, examples=[_load_scopes_symb, _load_scopes]
)


@frozen
class LoadUniqueScopeIndex(Bloq):
    r"""Given a scope $S$, load $j$ such that $S = U_j$, the $j$-th unique scope.

    If the input contains an invalid scope, then any arbitrary value can be output.

    Registers:
        S: A scope $S \in {[n] \choose k}$.
        j (RIGHT): a number in $[\bar{m}]$ s.t. $S = U_j$.
        ancilla (RIGHT): entangled intermediate qubits, to be uncomputed by the adjoint.
    """

    inst: KXorInstance

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(j=self.m_dtype, U=QAny(self.scope_bitsize))

    @cached_property
    def scope_bitsize(self) -> SymbolicInt:
        """total number of bits to store `k` indices in `[n]`."""
        return self.inst.k * self.inst.index_bitsize

    @cached_property
    def m_dtype(self):
        r"""number of bits to store $j \in [\bar{m}]$."""
        m = self.inst.num_unique_constraints
        bitsize = ceil(log2(m))
        return BQUInt(bitsize, m)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        counts = Counter[Bloq]()

        c = ssa.new_symbol("c")
        counts[EqualsAConstant(self.scope_bitsize, c)] += self.inst.num_unique_constraints

        return counts


@frozen
class PRGAUniqueConstraintRHS(Bloq):
    r"""Map $|j\rangle |0\rangle$ to $|j\rangle (\sqrt{E_j} |0\rangle + \sqrt{1 - |E_j|}|1\rangle)$

    Given an instance $\mathcal{I}$, with unique scopes $U_j$ and corresponding RHS values
    $E_j = B_\mathcal{I}(U_j)/M$ (where $M$ is the max. abs. entry, usually 2)
    apply the above rotation on the target qubit.

    This is done by first rotating for $|E_j|$ (i.e. ignoring the sign),
    by loading the values $\arccos{\sqrt{|E_j|}} / (2 * \pi)$,
    and applying an `Rx` using an `RzViaPhaseGradient` surrounded by `H`.

    We then apply the sign correction of $i$ for the negative entries by an $S$ gate.
    We ensure that the input data is sorted, therefore we can simply compare $j$
    with the largest negative index, and apply a `CS` gate.

    Args:
        inst: kXOR instance $\mathcal{I}$.
        angle_bitsize: number of bits to load the amplitude rotation angles to.

    Registers:
        j: Selection index, loads the value of $E_j = B_\mathcal{I}(U_j)/M$
        q: rotation target.
    """

    inst: KXorInstance
    angle_bitsize: SymbolicInt
    is_controlled: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(ctrl=QAny(self.n_ctrl), j=self.m_dtype, q=QBit())

    @property
    def n_ctrl(self) -> int:
        return 1 if self.is_controlled else 0

    @cached_property
    def m_dtype(self):
        r"""number of bits to store $j \in [\bar{m}]$."""
        m = self.inst.num_unique_constraints
        bitsize = ceil(log2(m))
        return BQUInt(bitsize, m)

    @cached_property
    def _angle_dtype(self):
        return QFxp(self.angle_bitsize, self.angle_bitsize)

    @cached_property
    def _qrom_angle_data(
        self,
    ) -> tuple[Union[HasLength, Sequence[int]], Union[HasLength, Sequence[int]]]:
        M = self.inst.max_rhs
        scopes = self.inst.batched_scopes
        if is_symbolic(M) or is_symbolic(scopes):
            m = self.inst.num_unique_constraints
            return HasLength(m), HasLength(m)

        b = [b for _, b in scopes]
        assert np.all(b == np.sort(b)), "data must be sorted!"

        amplitude_angles = np.arccos(np.sqrt(np.abs(b) / M))
        amplitude_angles_int = np.round(amplitude_angles * 2**self.angle_bitsize)

        signs = tuple(np.sign(b))
        return amplitude_angles_int, signs

    @cached_property
    def _amplitude_qrom(self) -> QROM:
        data, _ = self._qrom_angle_data
        if is_symbolic(data):
            return QROM.build_from_bitsize(
                data_len_or_shape=self.inst.num_unique_constraints,
                target_bitsizes=self.angle_bitsize,
                num_controls=self.n_ctrl,
            )

        return QROM.build_from_data(
            data, target_bitsizes=(self.angle_bitsize,), num_controls=self.n_ctrl
        )

    @cached_property
    def _num_negative(self) -> SymbolicInt:
        """returns $k$ s.t. the first $k$ elements are negative."""
        _, signs = self._qrom_angle_data
        if is_symbolic(signs):
            return self.inst.num_unique_constraints // 2

        assert np.all(signs == np.sort(signs)), "data must be sorted!"
        return int(np.searchsorted(signs, 0))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        counts = Counter[Bloq]()

        # load the amplitudes
        counts[self._amplitude_qrom] += 1

        # apply a Rx rotation using Rx = H Rz H
        counts[Hadamard()] += 2
        counts[RzViaPhaseGradient(self._angle_dtype, self._angle_dtype)] += 1

        # apply the sign correction
        # TODO use the half-bloq once implemented to wire this correctly
        sign_compare = LessThanConstant(self.m_dtype.num_qubits, self._num_negative)
        counts[sign_compare] += 1
        counts[SGate().controlled()] += 1

        # unload amplitudes
        counts[self._amplitude_qrom.adjoint()] += 1

        return counts

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=self if self.is_controlled else attrs.evolve(self, is_controlled=True),
            ctrl_reg_name='ctrl',
        )
