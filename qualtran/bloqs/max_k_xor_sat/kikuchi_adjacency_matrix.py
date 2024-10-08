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
from typing import Optional

import attrs
import sympy
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqDocSpec,
    CtrlSpec,
    QAny,
    QBit,
    QFxp,
    Signature,
)
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt

from .arithmetic import SymmetricDifference
from .kxor_instance import KXorInstance
from .load_kxor_instance import LoadUniqueScopeIndex, PRGAUniqueConstraintRHS


@frozen
class KikuchiMatrixEntry(Bloq):
    r"""Adjacency matrix oracle for the Kikuchi matrix.

    Given a kXOR instance $\mathcal{I}$ with $n$ variables, $m$ constraints,
    the Kikuchi matrix with parameter $\ell$ is indexed by ${[n] \choose l}$.
    For $S, T \in {[n] \choose l}$, the entry is given by
    $H_{S, T} = B_{\mathcal{I}}(S \Delta T)/M$, where $M$ is the max entry.

    This bloq implements the transform:
        $$
        |0 \rangle |S\rangle |T\rangle
        \mapsto
        (\sqrt{H_{S, T}}|0\rangle + \sqrt{1 - |H_{S, T}|}|1\rangle)|S\rangle |T\rangle
        $$

    This is equivalent to $O_H$ (Def. 4.3) from the paper, but is optimized to classically
    compute the `arccos` of the entries, and directly apply the rotation,
    instead of computing them using a quantum circuit.

    This bloq performs the following steps
    1. Compute the symmetric difference $D = S \Delta T$.
    2. Compute the index $j$ s.t. $U_j = D$ (where $U_j$ are a list of unique scopes)
    4. Apply a controlled Y-rotation with angle for the $j$-th entry.
    5. Uncompute steps 3, 2, 1.

    Args:
        inst: k-XOR instance
        ell: the Kikuchi parameter $\ell$, must be a multiple of $k$.
        entry_bitsize: number of bits to approximate each rotation angle to.
        cv: single bit control value (0 or 1), or None for uncontrolled (default).

    Registers:
        S: row index
        T: column index
        q: the qubit to rotate by $Ry(2 \arccos(\sqrt{H_{S,T} / M}))$ as defined above.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Definition 4.3. Theorem 4.17 para 3.
    """

    inst: KXorInstance
    ell: SymbolicInt
    entry_bitsize: SymbolicInt
    cv: Optional[int] = None

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            ctrl=QAny(1 if self.cv is not None else 0),
            S=QAny(self.index_bitsize),
            T=QAny(self.index_bitsize),
            q=QBit(),
        )

    @cached_property
    def index_bitsize(self) -> SymbolicInt:
        """total number of bits to store `l` indices in `[n]`."""
        return self.ell * self.inst.index_bitsize

    @cached_property
    def rotation_angle_dtype(self):
        return QFxp(self.entry_bitsize, self.entry_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        counts = Counter[Bloq]()

        # S \Delta T
        symm_diff = SymmetricDifference(self.ell, self.ell, self.inst.k, self.inst.index_bitsize)
        counts[symm_diff] += 1
        counts[symm_diff.adjoint()] += 1

        # Map S to j, such that U_j = S
        load_idx = LoadUniqueScopeIndex(self.inst)
        counts[load_idx] += 1
        counts[load_idx.adjoint()] += 1

        # apply the rotation
        rotation: Bloq = PRGAUniqueConstraintRHS(self.inst, self.entry_bitsize)
        if self.cv is not None:
            rotation = rotation.controlled(CtrlSpec(cvs=self.cv))
        counts[rotation] += 1

        return counts

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.bloq_with_specialized_single_qubit_control import (
            get_ctrl_system_for_bloq_with_specialized_single_qubit_control,
        )

        return get_ctrl_system_for_bloq_with_specialized_single_qubit_control(self, ctrl_spec)

    def with_cv(self, *, cv: Optional[int]) -> 'Bloq':
        return attrs.evolve(self, cv=cv)

    @property
    def ctrl_reg_name(self) -> str:
        return 'ctrl'


@bloq_example
def _kikuchi_matrix_entry() -> KikuchiMatrixEntry:
    from qualtran.bloqs.max_k_xor_sat.kxor_instance import Constraint, KXorInstance

    n, k = 10, 4
    cs = (
        Constraint((0, 1, 2, 3), -1),
        Constraint((0, 2, 4, 5), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((1, 2, 3, 4), -1),
        Constraint((1, 3, 4, 5), -1),
        Constraint((1, 3, 4, 5), -1),
        Constraint((2, 3, 4, 5), 1),
    )
    inst = KXorInstance(n, k, cs)
    ell = 8

    kikuchi_matrix_entry = KikuchiMatrixEntry(inst, ell, entry_bitsize=3)
    return kikuchi_matrix_entry


@bloq_example
def _kikuchi_matrix_entry_symb() -> KikuchiMatrixEntry:
    from qualtran.bloqs.max_k_xor_sat.kxor_instance import KXorInstance

    n, m, k, c = sympy.symbols("n m k c", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    ell = c * k

    kikuchi_matrix_entry_symb = KikuchiMatrixEntry(inst, ell, entry_bitsize=3)
    return kikuchi_matrix_entry_symb


_KIKUCHI_MATRIX_ENTRY_DOC = BloqDocSpec(
    bloq_cls=KikuchiMatrixEntry, examples=[_kikuchi_matrix_entry_symb, _kikuchi_matrix_entry]
)
