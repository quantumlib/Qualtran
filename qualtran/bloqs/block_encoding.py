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

In general, given an $s$-qubit operator $H$ then the $(s+a)$-qubit unitary $U$ is
a $(\alpha, a, \epsilon)$-block encoding of $H$ if it satisfies:

$$
    \lVert H - \alpha (\langle G|_a\otimes I_s U |G\rangle_a \otimes I_s) \rVert
    \le \epsilon,
$$

where $a$ is an ancilla register and $s$ is the system register, $U$ is a unitary sometimes
called a signal oracle and encodes $H$ in its top right corner, $\alpha \ge
\lVert H\rVert$ (where $\lVert \cdot \rVert$ denotes the spectral norm), and
$\epsilon$ is the precision to which the block encoding is prepared. The state
$|G\rangle_a$ is sometimes called the signal state, and its form depends on the
details of the block encoding. 

For LCU based block encodings 
we have
$$
U = \sum_l |l\rangle\langle l| \otimes U_l
$$
and $|G\rangle = \sum_l \sqrt{\frac{\alpha_l}{\alpha}}|0\rangle_a$, which define the
usual SELECT and PREPARE oracles.

Other ways of building block encodings exist so we define the abstract base
class `BlockEncoding` bloq, which expects values for $\alpha$, $\epsilon$,
system and ancilla registers and a bloq which prepares the state $|G\rangle$. 
"""

import abc
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING

import attrs

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.reflection import Reflection
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle
from qualtran.bloqs.util_bloqs import Partition
from qualtran.symbolics import SymbolicFloat, SymbolicInt

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
    def selection_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.select.selection_registers)

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.select.target_registers)

    def pretty_name(self) -> str:
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
    def selection_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.prepare.selection_registers)

    @cached_property
    def junk_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.prepare.selection_registers)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('selection', QAny(self.selection_bitsize)),
                Register('junk', QAny(self.junk_bitsize)),
            ]
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

    def pretty_name(self) -> str:
        return 'Prep'


class BlockEncoding(Bloq):
    r"""Abstract interface for an arbitrary block encoding.

    A $(\alpha, a, \epsilon) block encoding of an s-qubit operator $H$ if it obeys:

    $$
        \equiv \lVert H - \alpha (\langle G|_a\otimes I_s U |G\rangle_a \otimes I_s) \rVert
        \le \epsilon,
    $$

    where $a$ is an ancilla register and $s$ is the system register, $U$ is a unitary sometimes
    called a signal oracle and encodes $H$ in a subspace flagged by the ancilla
    state $|G\rangle_a$, which is sometimes called the signal state.

    Developers users must implement a method to return a bloq preparing the state $|G\rangle$.

    Users must specify:
        1. the normalization constant $\alpha \ge \lVert A \rVert$, where
            $\lVert \cdot \rVert denotes the spectral norm.
        2. the precision to which the block encoding is to be prepared ($\epsilon$).

    References:
        [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163/)
            Sec 2 and 3 for introduction and definition of terms.

        [The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation](https://arxiv.org/abs/1804.01973)
            Definition 3 page 8.

    """

    alpha: SymbolicFloat
    epsilon: SymbolicFloat

    def pretty_name(self) -> str:
        return 'B[H]'

    @property
    @abc.abstractmethod
    def selection_bitsize(self) -> SymbolicInt:
        """The bitsize for the register `a` registers above."""

    @property
    @abc.abstractmethod
    def junk_bitsize(self) -> SymbolicInt:
        """The bitsize of any additional junk register."""

    @property
    @abc.abstractmethod
    def system_bitsize(self) -> SymbolicInt:
        """The system bitsize `s`."""

    @property
    @abc.abstractmethod
    def signal_state(self) -> PrepareOracle:
        r"""Construct the signal state $|G\rangle."""


@attrs.frozen
class LCUBlockEncoding(BlockEncoding):
    r"""LCU based block encoding using SELECT and PREPARE oracles.

    Builds the block encoding via
    $$
        U[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE},
    $$
    where
    $$
        \mathrm{PREPARE} |0\rangle_a = \sum_l \sqrt{\frac{w_l}{\lambda}} |l\rangle_a,
    $$
    and
    $$
        \mathrm{SELECT} |l\rangle_a|\psi\rangle_s = |l\rangle_a U_l |\psi\rangle_s.
    $$

    The ancilla register is at least of size $\log L$.

    In our implementations we typically split the ancilla registers into
    selection registers (i.e.  the $l$ registers above) and junk registers which
    are extra qubits needed by state preparation but not controlled upon during
    SELECT.

    Here $|G\rangle = \mathrm{PREPARE}|0\rangle$.

    Args:
        alpha: The normalization constant upper bounding the spectral norm of
            the Hamiltonian. Often called lambda.
        epsilon: The precision to which the block encoding is performed.
            Currently this isn't used: see https://github.com/quantumlib/Qualtran/issues/985
        select: The bloq implementing the `SelectOracle` interface.
        prepare: The bloq implementing the `PrepareOracle` interface.

    Registers:
        selection: The combined selection register.
        junk: Additional junk registers not prepared upon.
        system: The combined system register.

    References:
        [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163/)
            Sec 3.1, page 7 and 8 for high level overview and definitions. A
            block encoding is called a standard form encoding there.

        [The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation](https://arxiv.org/abs/1804.01973)
            Definition 3 page 8.
    """

    alpha: SymbolicFloat
    epsilon: SymbolicFloat
    select: BlackBoxSelect = attrs.field(
        converter=lambda x: BlackBoxSelect(x) if isinstance(x, SelectOracle) else x
    )
    prepare: BlackBoxPrepare = attrs.field(
        converter=lambda x: BlackBoxPrepare(x) if isinstance(x, PrepareOracle) else x
    )

    @cached_property
    def selection_bitsize(self) -> SymbolicInt:
        return self.prepare.selection_bitsize

    @cached_property
    def junk_bitsize(self) -> SymbolicInt:
        return self.prepare.junk_bitsize

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.select.system_bitsize

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('selection', QAny(self.prepare.selection_bitsize)),
                Register('junk', QAny(self.prepare.junk_bitsize)),
                Register('system', QAny(self.select.system_bitsize)),
            ]
        )

    @cached_property
    def signal_state(self) -> BlackBoxPrepare:
        return self.prepare

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

    See https://github.com/quantumlib/Qualtran/issues/984 for an alternative.

    Args:
        block_encoding: Block encoding of a Hamiltonian $H$, $\mathcal{B}[H]$.
            Assumes the $|G\rangle$ state of the block encoding is the identity operator.
        order: order of Chebychev polynomial.

    References:
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Page 45; Theorem 1.
    """

    block_encoding: BlockEncoding
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
                Register('selection', QAny(self.block_encoding.selection_bitsize)),
                Register('junk', QAny(self.block_encoding.junk_bitsize)),
                Register('system', QAny(self.block_encoding.system_bitsize)),
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
def _black_box_block_bloq() -> LCUBlockEncoding:
    from qualtran.bloqs.hubbard_model import PrepareHubbard, SelectHubbard

    # 3x3 hubbard model U/t = 4
    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    N = dim * dim * 2
    qlambda = 2 * N * t + (N * U) // 2
    black_box_block_bloq = LCUBlockEncoding(
        select=select, prepare=prepare, alpha=qlambda, epsilon=0.0
    )
    return black_box_block_bloq


_BLACK_BOX_BLOCK_BLOQ_DOC = BloqDocSpec(
    bloq_cls=LCUBlockEncoding,
    examples=(_black_box_block_bloq,),
    import_line='from qualtran.bloqs.block_encoding import LCUBlockEncoding',
)


@bloq_example
def _chebyshev_poly() -> ChebyshevPolynomial:
    from qualtran.bloqs.block_encoding import LCUBlockEncoding
    from qualtran.bloqs.hubbard_model import PrepareHubbard, SelectHubbard

    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    N = dim * dim * 2
    qlambda = 2 * N * t + (N * U) // 2
    black_box_block_bloq = LCUBlockEncoding(
        select=select, prepare=prepare, alpha=qlambda, epsilon=0.0
    )
    chebyshev_poly = ChebyshevPolynomial(black_box_block_bloq, order=3)
    return chebyshev_poly


_CHEBYSHEV_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ChebyshevPolynomial,
    import_line='from qualtran.bloqs.block_encoding import ChebyshevPolynomial',
    examples=(_chebyshev_poly,),
)
