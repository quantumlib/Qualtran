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
"""SELECT for the sparse chemistry Hamiltonian in second quantization."""

from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import cirq
from attrs import frozen

from qualtran import bloq_example, BloqBuilder, BloqDocSpec, Register, SelectionRegister, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.select_and_prepare import SelectOracle
from qualtran.bloqs.selected_majorana_fermion import SelectedMajoranaFermion
from qualtran.cirq_interop import CirqGateAsBloq

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SelectSparse(SelectOracle):
    r"""SELECT oracle for the sparse Hamiltonian.

    Implements the two applications of Fig. 13.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.

    Registers:
        flag_1b: a single qubit to flag whether the one-body Hamiltonian is to
            be applied or not during SELECT.
        swap_pq: a |+> state to restore the symmetries of the p and q indices.
        swap_rs: a |+> state to restore the symmetries of the r and s indices.
        swap_pqrs: a |+> state to restore the symmetries of between (pq) and (rs).
        theta: sign qubit.
        pqrs: the register to store the spatial orbital index.
        alpha: spin for (pq) indicies.
        beta: spin for (rs) indicies.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Fig 13.
    """
    num_spin_orb: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "p",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "q",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "r",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "s",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister("alpha", bitsize=1),
            SelectionRegister("beta", bitsize=1),
            SelectionRegister("flag_1b", bitsize=1),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("sys", bitsize=self.num_spin_orb),)

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        p: 'SoquetT',
        q: 'SoquetT',
        r: 'SoquetT',
        s: 'SoquetT',
        alpha: 'SoquetT',
        beta: 'SoquetT',
        flag_1b: 'SoquetT',
        sys: 'SoquetT',
    ) -> Dict[str, 'SoquetT']:
        flag_1b = bb.add(CirqGateAsBloq(cirq.S), q=flag_1b)
        # note no extraction of sign from theta as this is done during state prep.
        sel_pa = (self.signature.get_left('p'), self.signature.get_left('alpha'))
        bloq = SelectedMajoranaFermion(sel_pa, target_gate=cirq.Y)
        flag_1b, p, alpha, sys = bb.add(bloq, control=flag_1b, p=p, alpha=alpha, target=sys)
        sel_qa = (self.signature.get_left('q'), self.signature.get_left('alpha'))
        bloq = SelectedMajoranaFermion(sel_qa, target_gate=cirq.X)
        flag_1b, q, alpha, sys = bb.add(bloq, control=flag_1b, q=q, alpha=alpha, target=sys)
        sel_rb = (self.signature.get_left('r'), self.signature.get_left('beta'))
        r, beta, sys = bb.add(
            SelectedMajoranaFermion(sel_rb, target_gate=cirq.Y, control_regs=()),
            r=r,
            beta=beta,
            target=sys,
        )
        sel_sb = (self.signature.get_left('s'), self.signature.get_left('beta'))
        s, beta, sys = bb.add(
            SelectedMajoranaFermion(sel_sb, target_gate=cirq.X, control_regs=()),
            s=s,
            beta=beta,
            target=sys,
        )
        return {
            'p': p,
            'q': q,
            'r': r,
            's': s,
            'alpha': alpha,
            'beta': beta,
            'flag_1b': flag_1b,
            'sys': sys,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Pg 30, enumeration 1: 2 applications of SELECT in Fig. 13, one of
        # which is not controlled (for the two body part of the Ham). The figure
        # is a bit misleading as applying that circuit twice would square the
        # value in the sign. In reality the Z to pick up the sign could be done
        # after prepare (but only once).
        # In practice we would apply the selected majoranas to (p, q, alpha) and then (r, s, beta).
        return {(Toffoli(), (4 * self.num_spin_orb - 6))}


@bloq_example
def _sel_sparse() -> SelectSparse:
    num_spin_orb = 4
    sel_sparse = SelectSparse(num_spin_orb)
    return sel_sparse


_SPARSE_SELECT = BloqDocSpec(
    bloq_cls=SelectSparse,
    import_line='from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse',
    examples=(_sel_sparse,),
)
