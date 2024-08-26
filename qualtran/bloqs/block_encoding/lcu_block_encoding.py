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

from functools import cached_property
from typing import Dict, Optional, Tuple, Union

import attrs

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    Register,
    Signature,
    SoquetT,
)
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding
from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import SymbolicFloat


def _total_bits(registers: Union[Tuple[Register, ...], Signature]) -> int:
    """Get the bitsize of a collection of registers"""
    return sum(r.total_bits() for r in registers)


@attrs.frozen
class SelectBlockEncoding(BlockEncoding, SpecializedSingleQubitControlledExtension):
    r"""LCU based block encoding using SELECT and PREPARE oracles.

    Builds the block encoding via
    $$
        B[H] = \mathrm{SELECT}
    $$

    $$
        \mathrm{SELECT} |l\rangle_a|\psi\rangle_s = |l\rangle_a U_l |\psi\rangle_s.
    $$

    The Hamiltonian can be extracted via

    $$
        \langle G | B[H] | G \rangle = H / \alpha
    $$

    where

    $$
        |G\rangle = \mathrm{PREPARE} |0\rangle_a = \sum_l \sqrt{\frac{w_l}{\alpha}} |l\rangle_a,
    $$

    The ancilla register is at least of size $\log L$.

    In our implementations we typically split the ancilla registers into
    selection registers (i.e.  the $l$ registers above) and junk registers which
    are extra qubits needed by state preparation but not controlled upon during
    SELECT.

    Args:
        select: The bloq implementing the `SelectOracle` interface.
        prepare: The bloq implementing the `PrepareOracle` interface.

    Registers:
        selection: The combined selection register.
        junk: Additional junk registers not prepared upon.
        system: The combined system register.

    References:
        [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163/)
            Low et al. 2019. Sec 3.1, page 7 and 8 for high level overview and definitions. A
            block encoding is called a standard form encoding there.

        [The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation](https://arxiv.org/abs/1804.01973)
            Chakraborty et al. 2018. Definition 3 page 8.
    """

    select: Union[BlackBoxSelect, SelectOracle]
    prepare: Union[BlackBoxPrepare, PrepareOracle]
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.select.control_registers

    @cached_property
    def ancilla_bitsize(self) -> int:
        return _total_bits(self.prepare.selection_registers)

    @cached_property
    def resource_bitsize(self) -> int:
        return _total_bits(self.prepare.junk_registers)

    @cached_property
    def system_bitsize(self) -> int:
        return _total_bits(self.select.target_registers)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.select.selection_registers

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return ()

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return self.select.target_registers

    @property
    def alpha(self) -> SymbolicFloat:
        return self.prepare.l1_norm_of_coeffs

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        # TODO: implement https://github.com/quantumlib/Qualtran/issues/1247
        return 0.0

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.junk_registers,
                *self.target_registers,
            ]
        )

    @cached_property
    def signal_state(self) -> Union[BlackBoxPrepare, PrepareOracle]:
        return self.prepare

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: SoquetT) -> Dict[str, 'SoquetT']:
        select_reg = {reg.name: soqs[reg.name] for reg in self.select.signature}
        soqs |= bb.add_d(self.select, **select_reg)
        return soqs

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'control':
            return Circle(filled=bool(self.control_val))
        else:
            return TextBox('B[H]')

    def get_single_qubit_controlled_bloq(self, control_val: int) -> 'SelectBlockEncoding':
        if self.control_val is not None:
            raise ValueError(
                "control_val is not None but trying to build controlled SelectBlockEncoding."
            )
        c_select = self.select.controlled(ctrl_spec=CtrlSpec(cvs=control_val))
        if not isinstance(c_select, SelectOracle):
            raise TypeError(
                f"controlled version of {self.select} = {c_select} must also be a SelectOracle"
            )
        return attrs.evolve(self, select=c_select, control_val=control_val)


@attrs.frozen
class LCUBlockEncoding(BlockEncoding, SpecializedSingleQubitControlledExtension):
    r"""LCU based block encoding using SELECT and PREPARE oracles.

    Builds the standard block encoding from an LCU as
    $$
        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE},
    $$
    where
    $$
        \mathrm{PREPARE} |0\rangle_a = \sum_l \sqrt{\frac{w_l}{\alpha}} |l\rangle_a,
    $$
    and
    $$
        \mathrm{SELECT} |l\rangle_a|\psi\rangle_s = |l\rangle_a U_l |\psi\rangle_s.
    $$

    The Hamiltonian can be extracted via
    $$
        \langle 0 | B[H] | 0 \rangle = H / \alpha,
    $$
    This differs from the `SelectBlockEncoding` which uses Prepare for the signal state, while here
    it is the identity operator.

    The ancilla register is at least of size $\log L$.

    In our implementations we typically split the ancilla registers into
    selection registers (i.e.  the $l$ registers above) and junk registers which
    are extra qubits needed by state preparation but not controlled upon during
    SELECT.

    Args:
        select: The bloq implementing the `SelectOracle` interface.
        prepare: The bloq implementing the `PrepareOracle` interface.

    Registers:
        selection: The combined selection register.
        junk: Additional junk registers not prepared upon.
        system: The combined system register.

    References:
        [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163/)
            Low et al. 2019. Sec 3.1, page 7 and 8 for high level overview and definitions. A
            block encoding is called a standard form encoding there.

        [The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation](https://arxiv.org/abs/1804.01973)
            Chakraborty et al. 2018. Definition 3 page 8.
    """

    select: Union[BlackBoxSelect, SelectOracle]
    prepare: Union[BlackBoxPrepare, PrepareOracle]
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.select.control_registers

    @cached_property
    def ancilla_bitsize(self) -> int:
        return _total_bits(self.prepare.selection_registers)

    @cached_property
    def resource_bitsize(self) -> int:
        return _total_bits(self.prepare.junk_registers)

    @cached_property
    def system_bitsize(self) -> int:
        return _total_bits(self.select.target_registers)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.prepare.selection_registers

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return self.prepare.junk_registers

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return self.select.target_registers

    @property
    def alpha(self) -> SymbolicFloat:
        return self.prepare.l1_norm_of_coeffs

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        # TODO: Implement epsilon for all block encodings
        # https://github.com/quantumlib/Qualtran/issues/1247
        return 0.0

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.junk_registers,
                *self.target_registers,
            ]
        )

    @cached_property
    def signal_state(self) -> Union[BlackBoxPrepare, PrepareOracle]:
        return PrepareIdentity(self.selection_registers)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: SoquetT) -> Dict[str, 'SoquetT']:
        def _extract_soqs(bloq: Bloq) -> Dict[str, 'SoquetT']:
            return {reg.name: soqs.pop(reg.name) for reg in bloq.signature.lefts()}

        soqs |= bb.add_d(self.prepare, **_extract_soqs(self.prepare))
        soqs |= bb.add_d(self.select, **_extract_soqs(self.select))
        soqs |= bb.add_d(self.prepare.adjoint(), **_extract_soqs(self.prepare.adjoint()))
        return soqs

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'control':
            return Circle(filled=bool(self.control_val))
        else:
            return TextBox('B[H]')

    def get_single_qubit_controlled_bloq(self, control_val: int) -> 'LCUBlockEncoding':
        if self.control_val is not None:
            raise ValueError(
                "control_val is not None but trying to build controlled SelectBlockEncoding."
            )
        c_select = self.select.controlled(ctrl_spec=CtrlSpec(cvs=control_val))
        if not isinstance(c_select, SelectOracle):
            raise TypeError(
                f"controlled version of {self.select} = {c_select} must also be a SelectOracle"
            )
        return attrs.evolve(self, select=c_select, control_val=control_val)


@bloq_example
def _select_block() -> SelectBlockEncoding:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard

    # 3x3 hubbard model U/t = 4
    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    select_block = SelectBlockEncoding(select=select, prepare=prepare)
    return select_block


@bloq_example
def _black_box_select_block() -> SelectBlockEncoding:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    # 3x3 hubbard model U/t = 4
    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    black_box_select_block = SelectBlockEncoding(
        select=BlackBoxSelect(select), prepare=BlackBoxPrepare(prepare)
    )
    return black_box_select_block


@bloq_example
def _lcu_block() -> LCUBlockEncoding:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard

    # 3x3 hubbard model U/t = 4
    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    lcu_block = LCUBlockEncoding(select=select, prepare=prepare)
    return lcu_block


@bloq_example
def _black_box_lcu_block() -> LCUBlockEncoding:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    # 3x3 hubbard model U/t = 4
    dim = 3
    select = SelectHubbard(x_dim=dim, y_dim=dim)
    U = 4
    t = 1
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=t, u=U)
    black_box_lcu_block = LCUBlockEncoding(
        select=BlackBoxSelect(select), prepare=BlackBoxPrepare(prepare)
    )
    return black_box_lcu_block


_SELECT_BLOCK_ENCODING_DOC = BloqDocSpec(
    bloq_cls=SelectBlockEncoding, examples=(_select_block, _black_box_select_block)
)

_LCU_BLOCK_ENCODING_DOC = BloqDocSpec(
    bloq_cls=LCUBlockEncoding, examples=(_lcu_block, _black_box_lcu_block)
)
