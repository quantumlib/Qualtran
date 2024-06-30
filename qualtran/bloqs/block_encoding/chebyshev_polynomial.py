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
r"""High level bloqs for defining bloq encodings and operations on block encodings."""

from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING, Union

import attrs

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Register, Signature, SoquetT
from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncodingZeroState
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class ChebyshevPolynomial(Bloq):
    r"""Block encoding of $T_j[H]$ where $T_j$ is the $j$-th Chebyshev polynomial.

    Here H is a Hamiltonian with spectral norm $|H| \le 1$, we assume we have
    an $n_L$ qubit ancilla register, and assume that $j > 0$ to avoid block
    encoding the identity operator.

    Recall:

    \begin{align*}
        T_0[H] &= \mathbb{1} \\
        T_1[H] &= H \\
        T_2[H] &= 2 H^2 - \mathbb{1} \\
        T_3[H] &= 4 H^3 - 3 H \\
        &\dots
    \end{align*}

    See https://github.com/quantumlib/Qualtran/issues/984 for an alternative.

    Args:
        block_encoding: Block encoding of a Hamiltonian $H$, $\mathcal{B}[H]$.
            Assumes the $|G\rangle$ state of the block encoding is the identity operator.
        order: order of Chebychev polynomial.

    References:
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            von Burg et al. 2007. Page 45; Theorem 1.
    """

    block_encoding: LCUBlockEncodingZeroState
    order: int

    def __attrs_post_init__(self):
        if self.order < 1:
            raise ValueError(f"order must be greater >= 1. Found {self.order}.")

    def pretty_name(self) -> str:
        return f"T_{self.order}[{self.block_encoding.pretty_name()}]"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.block_encoding.selection_registers,
                *self.block_encoding.junk_registers,
                *self.block_encoding.target_registers,
            ]
        )

    def build_reflection_bloq(self) -> 'ReflectionUsingPrepare':
        refl_bitsizes = tuple(r.bitsize for r in self.block_encoding.selection_registers)
        return ReflectionUsingPrepare.reflection_around_zero(
            bitsizes=refl_bitsizes, global_phase=-1
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: SoquetT) -> Dict[str, 'SoquetT']:
        # includes selection registers and any selection registers used by PREPARE
        soqs |= bb.add_d(self.block_encoding, **soqs)

        def _extract_soqs(
            to_regs: Union[Signature, Tuple[Register, ...]],
            from_regs: Union[Signature, Tuple[Register, ...]],
            reg_map: Dict[str, 'SoquetT'],
        ) -> Dict[str, 'SoquetT']:
            return {t.name: reg_map[f.name] for t, f in zip(to_regs, from_regs)}

        refl_bloq = self.build_reflection_bloq()
        for iorder in range(1, self.order):
            refl_regs = _extract_soqs(
                refl_bloq.signature, self.block_encoding.selection_registers, soqs
            )
            refl_regs |= bb.add_d(self.build_reflection_bloq(), **refl_regs)
            soqs |= _extract_soqs(
                self.block_encoding.selection_registers, refl_bloq.signature, refl_regs
            )
            soqs |= bb.add_d(self.block_encoding, **soqs)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.order
        return {(self.build_reflection_bloq(), n - 1), (self.block_encoding, n)}


@bloq_example
def _chebyshev_poly() -> ChebyshevPolynomial:
    from qualtran.bloqs.block_encoding import LCUBlockEncodingZeroState
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard

    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    N = dim * dim * 2
    qlambda = 2 * N * t + (N * U) // 2
    block_bloq = LCUBlockEncodingZeroState(
        select=select, prepare=prepare, alpha=qlambda, epsilon=0.0
    )
    chebyshev_poly = ChebyshevPolynomial(block_bloq, order=3)
    return chebyshev_poly


@bloq_example
def _black_box_chebyshev_poly() -> ChebyshevPolynomial:
    from qualtran.bloqs.block_encoding import (
        BlackBoxPrepare,
        BlackBoxSelect,
        LCUBlockEncodingZeroState,
    )
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard

    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    N = dim * dim * 2
    qlambda = 2 * N * t + (N * U) // 2
    black_box_block_bloq = LCUBlockEncodingZeroState(
        select=BlackBoxSelect(select), prepare=BlackBoxPrepare(prepare), alpha=qlambda, epsilon=0.0
    )
    black_box_chebyshev_poly = ChebyshevPolynomial(black_box_block_bloq, order=3)
    return black_box_chebyshev_poly


_CHEBYSHEV_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ChebyshevPolynomial,
    import_line='from qualtran.bloqs.block_encoding import ChebyshevPolynomial',
    examples=(_chebyshev_poly, _black_box_chebyshev_poly),
)
