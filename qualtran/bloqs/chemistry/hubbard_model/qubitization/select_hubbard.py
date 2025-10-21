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
r"""Subroutines for the SELECT operation for the Hubbard model Hamiltonian.

This module follows section V. of Encoding Electronic Spectra in Quantum Circuits with Linear T
Complexity. Babbush et al. 2018. [arxiv:1805.03662](https://arxiv.org/abs/1805.03662).


The goal is to construct a SELECT operator optimized for the following
Hamiltonian:

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = -\frac{t}{2} \sum_{\langle p,q \rangle, \sigma} (\hop{X} + \hop{Y})
  + \frac{u}{8} \sum_{p,\alpha\ne\beta} Z_{p,\alpha}Z_{p,\beta}
  - \frac{u}{4} \sum_{p,\sigma} Z_{p,\sigma} + \frac{uN}{4}\mathbb{1}
$$

With these operators, our selection register has indices
for $p$, $\alpha$, $q$, and $\beta$ as well as two indicator bits $U$ and $V$. There are four cases
considered in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U=1$, single-body Z
 - $V=1$, spin-spin ZZ term
 - $p<q$, XZX term
 - $p>q$, YZY term.
"""

from functools import cached_property
from typing import Dict, Iterator, Optional, Set, Tuple, Union

import attrs
import cirq
import sympy
from numpy.typing import NDArray

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    CompositeBloq,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, TwoBitCSwap
from qualtran.bloqs.bookkeeping import Join2, Split2
from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.multiplexers.selected_majorana_fermion import SelectedMajoranaFermion
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
from qualtran.symbolics import ceil, is_symbolic, log2, SymbolicInt


@attrs.frozen
class SelectHubbard(SelectOracle):
    r"""The SELECT operation optimized for the 2D Hubbard model.

    In contrast to SELECT for an arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` index for phases.

    If neither $U$ nor $V$ is set we apply the kinetic terms of the Hamiltonian:

    $$
    \def\Zvec{\overrightarrow{Z}}
    \def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
    -\hop{X} \quad p < q \\
    -\hop{Y} \quad p > q
    $$

    If $U$ is set we know $(p,\alpha)=(q,\beta)$ and apply the single-body term: $-Z_{p,\alpha}$.
    If $V$ is set we know $p=q, \alpha=0$, and $\beta=1$ and apply the spin term:
    $Z_{p,\alpha}Z_{p,\beta}$

    `SelectHubbard`'s construction uses $10 * N + log(N)$ T-gates.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means un-controlled.

    Registers:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 19.
    """

    x_dim: int
    y_dim: int
    control_val: Optional[int] = attrs.field(default=None, kw_only=True)

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
        if self.control_val == 0:
            raise NotImplementedError(
                "control_val=0 not supported, use `SelectHubbard(x, y).controlled(CtrlSpec(cvs=0))` instead"
            )

    @cached_property
    def log_m(self) -> SymbolicInt:
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
        return ceil(log2(self.x_dim))

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('U', BQUInt(1, 2)),
            Register('V', BQUInt(1, 2)),
            Register('p_x', BQUInt(self.log_m, self.x_dim)),
            Register('p_y', BQUInt(self.log_m, self.y_dim)),
            Register('alpha', BQUInt(1, 2)),
            Register('q_x', BQUInt(self.log_m, self.x_dim)),
            Register('q_y', BQUInt(self.log_m, self.y_dim)),
            Register('beta', BQUInt(1, 2)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.x_dim * self.y_dim * 2)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        control, target = quregs.get('control', ()), quregs['target']

        yield HubbardMajorannaOperator(
            x_dim=self.x_dim, y_dim=self.y_dim, gate='Y', control_val=self.control_val
        ).on_registers(x=p_x, y=p_y, spin=alpha, target=target, control=control)

        yield CSwap.make_on(ctrl=V, x=p_x, y=q_x)
        yield CSwap.make_on(ctrl=V, x=p_y, y=q_y)
        yield TwoBitCSwap().on_registers(ctrl=V, x=alpha, y=beta)

        yield HubbardMajorannaOperator(
            x_dim=self.x_dim, y_dim=self.y_dim, gate='X', control_val=self.control_val
        ).on_registers(x=q_x, y=q_y, spin=beta, target=target, control=control)

        yield TwoBitCSwap().on_registers(ctrl=V, x=alpha, y=beta)
        yield CSwap.make_on(ctrl=V, x=p_y, y=q_y)
        yield CSwap.make_on(ctrl=V, x=p_x, y=q_x)

        yield (
            cirq.S(*control) ** -1 if control else cirq.global_phase_operation(-1j)
        )  # Fix errant i from XY=iZ
        yield cirq.Z(*U).controlled_by(*control)  # Fix errant -1 from multiple pauli applications

        yield HubbardSpinUpZ(
            x_dim=self.x_dim, y_dim=self.y_dim, control_val=self.control_val
        ).on_registers(x=q_x, y=q_y, V=V, control=control, target=target)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=self.control_val,
            bloq_with_ctrl=attrs.evolve(self, control_val=1),
            ctrl_reg_name='control',
        )

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.mcmt.specialized_ctrl import (
            AdjointWithSpecializedCtrl,
            SpecializeOnCtrlBit,
        )

        return AdjointWithSpecializedCtrl(self, specialize_on_ctrl=SpecializeOnCtrlBit.ONE)

    def __str__(self):
        cstr = 'C' if self.control_val is not None else ''
        return f'{cstr}SelectHubbard({self.x_dim}, {self.y_dim})'


@bloq_example
def _sel_hubb() -> SelectHubbard:
    x_dim = 4
    y_dim = 4
    sel_hubb = SelectHubbard(x_dim, y_dim)
    return sel_hubb


_SELECT_HUBBARD_DOC: BloqDocSpec = BloqDocSpec(bloq_cls=SelectHubbard, examples=(_sel_hubb,))


@attrs.frozen
class HubbardMajorannaOperator(Bloq):
    r"""Apply majoranna fermion operation to the Hubbard system.

    Specifically apply $\overrightarrow{Z} P_{x,y,\sigma}$ for single-qubit
    Pauli $P$.

    This is a subroutine of `HubbardSelect`.

    This uses $N$ Toffoli gates, where `N=2*x_dim*y_dim`.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        gate: Either "X" or "Y" to do the corresponding Majoranna operation.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means un-controlled.

    Registers:
        x: Site indices, x component.
        y: Site indices, y component.
        spin: Sites' spin indicator.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 19, "Majoranna operators". See also Figure 9.
    """

    x_dim: SymbolicInt
    y_dim: SymbolicInt
    gate: str = 'Y'
    control_val: Optional[int] = attrs.field(default=None, kw_only=True)

    @cached_property
    def log_m(self) -> SymbolicInt:
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
        return ceil(log2(self.x_dim))

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('x', BQUInt(self.log_m, self.x_dim)),
            Register('y', BQUInt(self.log_m, self.y_dim)),
            Register('spin', BQUInt(1, 2)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.x_dim * self.y_dim * 2)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    @cached_property
    def _target_cirq_gate(self):
        if self.gate == 'X':
            return cirq.X
        elif self.gate == 'Y':
            return cirq.Y
        else:
            raise ValueError(f"Unknown gate {self.gate}")

    @cached_property
    def selected_majoranna_fermion_bloq(self) -> SelectedMajoranaFermion:
        return SelectedMajoranaFermion(
            selection_regs=(
                Register('x', BQUInt(self.log_m, self.x_dim)),
                Register('y', BQUInt(self.log_m, self.y_dim)),
                Register('spin', BQUInt(1, 2)),
            ),
            control_regs=self.control_registers,
            target_gate=self._target_cirq_gate,
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x, y, spin, target, control=None
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.x_dim, self.y_dim):
            raise DecomposeTypeError(f"Cannot decompose symbolic x_dim, y_dim in {self}")

        smf = self.selected_majoranna_fermion_bloq
        if self.control_val:
            control, x, y, spin, target = bb.add_from(
                smf, control=control, x=x, y=y, spin=spin, target=target
            )
            return {'control': control, 'x': x, 'y': y, 'spin': spin, 'target': target}
        else:
            x, y, spin, target = bb.add_from(smf, x=x, y=y, spin=spin, target=target)
            return {'x': x, 'y': y, 'spin': spin, 'target': target}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self.selected_majoranna_fermion_bloq.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return TextBox("")
        if reg.name == 'control':
            return Circle()
        if reg.name == 'target':
            return TextBox(r"$\overrightarrow{Z} %s_{x,y,\sigma}$" % self.gate)
        return TextBox(reg.name)

    def __str__(self):
        cstr = 'C' if self.control_val is not None else ''
        return f'{cstr}HubbardMajorannaOperator({self.x_dim}, {self.y_dim}, {self.gate})'


@bloq_example
def _hubb_majoranna() -> HubbardMajorannaOperator:
    n = sympy.Symbol('n')
    hubb_majoranna = HubbardMajorannaOperator(x_dim=n, y_dim=n, gate='X', control_val=1)
    return hubb_majoranna


@bloq_example
def _hubb_majoranna_small() -> HubbardMajorannaOperator:
    hubb_majoranna_small = HubbardMajorannaOperator(x_dim=3, y_dim=3, gate='Y')
    return hubb_majoranna_small


_HUBBARD_MAJORANNA_OPERATOR_DOC: BloqDocSpec = BloqDocSpec(
    bloq_cls=HubbardMajorannaOperator,
    examples=(_hubb_majoranna, _hubb_majoranna_small),
    call_graph_example=None,
)


@attrs.frozen
class HubbardSpinUpZ(Bloq):
    r"""Phase the spin up subspace of the Hubbard system.

    Specifically, apply $Z_{q,1}$.

    In combination with the `HubbardMajorannaOperator` subroutines, this applies the spin term
    $Z_{p,\alpha}Z_{p,\beta}$ if $V$ is set. This is a subroutine of `HubbardSelect`.

    This uses $N/2$ Toffoli gates, where `N=2*x_dim*y_dim`.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means un-controlled.

    Registers:
        V: Whether we're applying the pairwise part of the potential. If not set, this bloq does
            nothing.
        x: Site indices, x component.
        y: Site indices, y component.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 19, last operation; which references Fig 7.
    """

    x_dim: SymbolicInt
    y_dim: SymbolicInt
    control_val: Optional[int] = attrs.field(default=None, kw_only=True)

    @cached_property
    def log_m(self) -> SymbolicInt:
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
        return ceil(log2(self.x_dim))

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('V', BQUInt(1, 2)),
            Register('x', BQUInt(self.log_m, self.x_dim)),
            Register('y', BQUInt(self.log_m, self.y_dim)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.x_dim * self.y_dim * 2)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    @cached_property
    def _apply_z_to_lth(self) -> ApplyGateToLthQubit:
        c_size = 1 if self.control_val is None else 2
        return ApplyGateToLthQubit(
            selection_regs=(
                Register('y', BQUInt(self.signature.get_left('y').total_bits(), self.y_dim)),
                Register('x', BQUInt(self.signature.get_left('x').total_bits(), self.x_dim)),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Register('control', QAny(c_size)),
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', V, x, y, target, control=None
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.x_dim, self.y_dim):
            raise DecomposeTypeError(f"Cannot decompose symbolic x_dim, y_dim in {self}")

        # If we have a control bit, pack it into `ApplyToLthQubit` control register
        if self.control_val is not None:
            control = bb.join([V, control])
        else:
            control = V

        # `target` is a QAny(xdim * ydim * 2).
        # We index into it like (alpha, y, x),
        # So we can access the spin-up portion by splitting `target` in half.
        n_half = self.x_dim * self.y_dim
        spin_down, spin_up = bb.add(Split2(n_half, n_half), x=target)

        # Delegate to `ApplyGateToLthQubit`.
        control, y, x, spin_up = bb.add_from(
            self._apply_z_to_lth, x=x, y=y, control=control, target=spin_up
        )
        target = bb.add(Join2(n_half, n_half), y1=spin_down, y2=spin_up)

        # If we had a control bit, unpack it from `ApplyToLthQubit` control register
        if self.control_val is not None:
            V, control = bb.split(control)
            ret = {'control': control, 'V': V}
        else:
            V = control
            ret = {'V': V}

        return ret | {'x': x, 'y': y, 'target': target}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._apply_z_to_lth.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return TextBox('')
        if reg.name == 'control' or reg.name == 'V':
            return Circle()
        if reg.name == 'target':
            return TextBox(r'$Z_{x, y, 1}$')
        return TextBox(reg.name)

    def __str__(self):
        cstr = 'C' if self.control_val is not None else ''
        return f'{cstr}HubbardSpinUpZ({self.x_dim}, {self.y_dim})'


@bloq_example
def _hubb_spin_up_z() -> HubbardSpinUpZ:
    n = sympy.Symbol('n')
    hubb_spin_up_z = HubbardSpinUpZ(x_dim=n, y_dim=n, control_val=1)
    return hubb_spin_up_z


@bloq_example
def _hubb_spin_up_z_small() -> HubbardSpinUpZ:
    hubb_spin_up_z_small = HubbardSpinUpZ(x_dim=3, y_dim=3)
    return hubb_spin_up_z_small


_HUBBARD_SPIN_UP_Z_DOC: BloqDocSpec = BloqDocSpec(
    bloq_cls=HubbardSpinUpZ,
    examples=(_hubb_spin_up_z, _hubb_spin_up_z_small),
    call_graph_example=None,
)
