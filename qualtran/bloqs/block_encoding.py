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
r"""High level bloqs for defining bloq encodings and operations on block encodings.

Given an operator $V$ which can be expressed as a linear combination of unitaries

$$
    V = \sum_l^L w_l U_l,
$$
where $w_l \ge 0$, $w_l \in \mathbb{R}$, and $U_l$ is a unitary, then the block
encoding $\mathcal{B}\left[\frac{V}{\lambda}\right]$ satisifies
$$
    _a\langle 0| \mathcal{B}\left[\frac{V}{\lambda}\right] |0\rangle_a
    |\psi\rangle_s = \frac{V}{\lambda}|\psi\rangle_s
$$
where the subscripts $a$ and $s$ signify ancilla and system registers
respectively, and $\lambda = \sum_l w_l$. The ancilla register is at least of size $\log L$. In our
implementations we typically split the ancilla registers into selection registers (i.e.
the $l$ registers above) and junk registers which are extra qubits needed by
state preparation but not controlled upon during SELECT.
"""

from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING

import attrs

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.reflection import Reflection
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle
from qualtran.bloqs.util_bloqs import Partition

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class BlackBoxSelect(Bloq):
    r"""A 'black box' Select bloq.

    The `SELECT` operation applies the $l$'th unitary $U_{l}$ on the system register
    when the selection register stores integer $l$.
    When implementing specific `SelectOracle` bloqs, it is helpful to have multiple selection
    registers each with semantic meaning. For example: you could have spatial or spin coordinates
    on different, named registers. The `SelectOracle` interface encourages this. `BlackBoxSelect`
    uses the properties on the `SelectOracle` interface to provide a "black box" view of a select
    operation that just has a selection and system register.
    During decomposition, this bloq will use the `Partition` utility bloq to partition
    and route the parts of the unified selection register to the `Select` bloq.

    Args:
        select: The bloq implementing the `SelectOracle` interface.

    Registers:
        selection: The combined selection register
        system: The combined system register
    """

    select: SelectOracle

    @cached_property
    def selection_bitsize(self):
        return sum(reg.total_bits() for reg in self.select.selection_registers)

    @cached_property
    def system_bitsize(self):
        return sum(reg.total_bits() for reg in self.select.target_registers)

    def short_name(self) -> str:
        return 'SEL'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(selection=self.selection_bitsize, system=self.system_bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        # includes selection registers and any selection registers used by PREPARE
        sel_regs = self.select.selection_registers
        sel_part = Partition(self.selection_bitsize, regs=sel_regs)
        sel_out_regs = bb.add_t(sel_part, x=selection)
        sys_regs = tuple(self.select.target_registers)
        sys_part = Partition(self.system_bitsize, regs=sys_regs)
        sys_out_regs = bb.add_t(sys_part, x=system)
        out_regs = bb.add(
            self.select,
            **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)},
            **{reg.name: sp for reg, sp in zip(sys_regs, sys_out_regs)},
        )
        sel_out_regs = out_regs[: len(sel_regs)]
        sys_out_regs = out_regs[len(sel_regs) :]
        selection = bb.add(
            sel_part.adjoint(), **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)}
        )
        system = bb.add(
            sys_part.adjoint(), **{reg.name: sp for reg, sp in zip(sys_regs, sys_out_regs)}
        )
        return {'selection': selection, 'system': system}


@attrs.frozen
class BlackBoxPrepare(Bloq):
    """Provide a black-box interface to `Prepare` bloqs.

    This wrapper uses `Partition` to combine descriptive selection
    registers into one register named "selection".

    Args:
        prepare: The bloq following the `Prepare` interface to wrap.

    Registers:
        selection: selection register.
        junk: Additional junk registers not prepared upon.
    """

    prepare: PrepareOracle

    @cached_property
    def selection_bitsize(self):
        return sum(reg.total_bits() for reg in self.prepare.selection_registers)

    @cached_property
    def junk_bitsize(self):
        return sum(reg.total_bits() for reg in self.prepare.selection_registers)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register('selection', self.selection_bitsize), Register('junk', self.junk_bitsize)]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: 'SoquetT', junk: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        sel_regs = self.prepare.selection_registers
        sel_part = Partition(self.selection_bitsize, regs=sel_regs)
        sel_out_regs = bb.add_t(sel_part, x=selection)
        jnk_regs = tuple(self.prepare.junk_registers)
        jnk_part = Partition(self.junk_bitsize, regs=jnk_regs)
        jnk_out_regs = bb.add_t(jnk_part, x=junk)
        out_regs = bb.add(
            self.prepare,
            **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)},
            **{reg.name: sp for reg, sp in zip(jnk_regs, jnk_out_regs)},
        )
        sel_out_regs = out_regs[: len(sel_regs)]
        jnk_out_regs = out_regs[len(sel_regs) :]
        selection = bb.add(
            sel_part.adjoint(), **{reg.name: sp for reg, sp in zip(sel_regs, sel_out_regs)}
        )
        junk = bb.add(
            jnk_part.adjoint(), **{reg.name: sp for reg, sp in zip(jnk_regs, jnk_out_regs)}
        )
        return {'selection': selection, 'junk': junk}

    def short_name(self) -> str:
        return 'Prep'


@attrs.frozen
class BlackBoxBlockEncoding(Bloq):
    r"""Black box implementation of a block encoding using SELECT and PREPARE.

    Builds the block encoding via
    $$
        \mathcal{B}[V] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE},
    $$
    where
    $$
        \mathrm{PREPARE} |0\rangle_a = \sum_l \sqrt{\frac{w_l}{\lambda}} |l\rangle_a,
    $$
    and
    $$
        \mathrm{SELECT} |l\rangle_a|\psi\rangle_s = |l\rangle_a U_l |\psi\rangle_s.
    $$

    Args:
        select: The bloq implementing the `SelectOracle` interface.
        prepare: The bloq implementing the `SelectOracle` interface.

    Registers:
        selection: The combined selection register.
        junk: Additional junk registers not prepared upon.
        system: The combined system register.
    """

    select: BlackBoxSelect
    prepare: BlackBoxPrepare

    @cached_property
    def selection_bitsize(self):
        return self.prepare.selection_bitsize

    @cached_property
    def junk_bitsize(self):
        return self.prepare.junk_bitsize

    @cached_property
    def system_bitsize(self):
        return self.select.system_bitsize

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('selection', self.prepare.selection_bitsize),
                Register('junk', self.prepare.junk_bitsize),
                Register('system', self.select.system_bitsize),
            ]
        )

    def short_name(self) -> str:
        return 'B[V]'

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: 'SoquetT', junk: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        # includes selection registers and any selection registers used by PREPARE
        selection, junk = bb.add(self.prepare, selection=selection, junk=junk)
        selection, system = bb.add(self.select, selection=selection, system=system)
        selection, junk = bb.add(self.prepare.adjoint(), selection=selection, junk=junk)
        return {'selection': selection, 'junk': junk, 'system': system}


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

    Args:
        block_encoding: Block encoding of a Hamiltonian $H$, $\mathcal{B}[H]$.
        order: order of Chebychev polynomial.

    References:
        [Quantum computing enhanced computational catalysis](
            https://arxiv.org/abs/2007.14460). Page 45; Theorem 1.
    """

    block_encoding: BlackBoxBlockEncoding
    order: int

    def __attrs_post_init__(self):
        if self.order < 1:
            raise ValueError(f"order must be greater >= 1. Found {self.order}.")

    def short_name(self) -> str:
        return f"T_{self.order}[{self.block_encoding.short_name()}]"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('selection', self.block_encoding.selection_bitsize),
                Register('junk', self.block_encoding.junk_bitsize),
                Register('system', self.block_encoding.system_bitsize),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: 'SoquetT', junk: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        # includes selection registers and any selection registers used by PREPARE
        selection, junk, system = bb.add(
            self.block_encoding, selection=selection, junk=junk, system=system
        )
        for iorder in range(1, self.order):
            selection = bb.add(
                Reflection(bitsizes=(self.block_encoding.selection_bitsize,), cvs=(0,)),
                reg0=selection,
            )
            selection, junk, system = bb.add(
                self.block_encoding, selection=selection, junk=junk, system=system
            )
        return {'selection': selection, 'junk': junk, 'system': system}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.order
        num_refl = self.block_encoding.selection_bitsize
        return {(Reflection(bitsizes=(num_refl,), cvs=(0,)), n - 1), (self.block_encoding, n)}


@bloq_example
def _black_box_prepare() -> BlackBoxPrepare:
    from qualtran.bloqs.hubbard_model import PrepareHubbard

    prepare = PrepareHubbard(2, 2, 1, 4)
    black_box_prepare = BlackBoxPrepare(prepare=prepare)
    return black_box_prepare


@bloq_example
def _black_box_select() -> BlackBoxSelect:
    from qualtran.bloqs.hubbard_model import SelectHubbard

    select = SelectHubbard(2, 2)
    black_box_select = BlackBoxSelect(select=select)
    return black_box_select


@bloq_example
def _black_box_block_bloq() -> BlackBoxBlockEncoding:
    from qualtran.bloqs.block_encoding import BlackBoxBlockEncoding, BlackBoxPrepare, BlackBoxSelect
    from qualtran.bloqs.hubbard_model import PrepareHubbard, SelectHubbard

    dim = 3
    select = BlackBoxSelect(SelectHubbard(x_dim=dim, y_dim=dim))
    prepare = BlackBoxPrepare(PrepareHubbard(x_dim=dim, y_dim=dim, t=1, mu=4))
    black_box_block_bloq = BlackBoxBlockEncoding(select=select, prepare=prepare)
    return black_box_block_bloq


_BLACK_BOX_BLOCK_BLOQ_DOC = BloqDocSpec(
    bloq_cls=BlackBoxBlockEncoding,
    import_line='from qualtran.bloqs.block_encoding import BlackBoxBlockEncoding',
    examples=(_black_box_block_bloq,),
)


@bloq_example
def _chebyshev_poly() -> ChebyshevPolynomial:
    from qualtran.bloqs.block_encoding import BlackBoxBlockEncoding, BlackBoxPrepare, BlackBoxSelect
    from qualtran.bloqs.hubbard_model import PrepareHubbard, SelectHubbard

    dim = 3
    select = BlackBoxSelect(SelectHubbard(x_dim=dim, y_dim=dim))
    prepare = BlackBoxPrepare(PrepareHubbard(x_dim=dim, y_dim=dim, t=1, mu=4))
    black_box_block_bloq = BlackBoxBlockEncoding(select=select, prepare=prepare)
    chebyshev_poly = ChebyshevPolynomial(black_box_block_bloq, order=3)
    return chebyshev_poly


_CHEBYSHEV_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ChebyshevPolynomial,
    import_line='from qualtran.bloqs.block_encoding import ChebyshevPolynomial',
    examples=(_chebyshev_poly,),
)
