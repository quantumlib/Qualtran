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

from functools import cached_property
from typing import cast, Iterable, Optional, Sequence, Set, Tuple, Union

import cirq
import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, BloqDocSpec, BQUInt, QBit, Register, Side
from qualtran._infra.gate_with_registers import merge_qubits
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.resource_counting import BloqCountT
from qualtran.symbolics import ceil, log2


@frozen
class ApplyLthBloq(UnaryIterationGate, SpecializedSingleQubitControlledExtension, SelectOracle):  # type: ignore[misc]
    r"""A SELECT operation that executes one of a list of bloqs $U_l$ based on a quantum index:

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    This bloq uses the unary iteration scheme to apply `ops[selection]` controlled on an optional
    single-bit `control` register.

    Args:
        ops: NDArray of bloqs. Each bloq must have identical registers that are all THRU.
        selection_regs: List of selection registers, defaults to N-D selection index based on `ops`.
        control_val: If provided, a singly controlled gate is constructed.

    Registers:
        selection: The indices of the bloq in `ops` to execute.
        control: The control bit if specified above.
        [user_spec]: The output registers of the bloqs in `ops`.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](
        https://arxiv.org/abs/1805.03662). Babbush et. al. (2018). Section III.A. and Figure 7.
    """

    # type ignore needed here for Bloq as NDArray parameter
    ops: NDArray[Bloq] = field(  # type: ignore[type-var]
        converter=lambda x: np.array(x) if isinstance(x, Iterable) else x,
        eq=lambda d: tuple(d.flat),
    )
    selection_regs: Optional[Tuple[Register, ...]] = None
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        if np.prod(self.ops.shape) <= 1:
            raise ValueError("Must have at least two operations.")
        if not all(tuple(u.signature) == self.target_registers for u in self.ops.flat):
            raise ValueError("All ops must have the same signature.")
        if not all(r.side == Side.THRU for u in self.ops.flat for r in u.signature):
            raise ValueError("All ops must have only THRU registers.")

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        if self.selection_regs is None:
            return tuple(
                Register(
                    "selection" if len(self.ops.shape) == 1 else f"selection{i}",
                    BQUInt(ceil(log2(k)), k),
                )
                for i, k in enumerate(self.ops.shape)
            )
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.ops.flat[0].signature)

    def nth_operation_callgraph(self, **kwargs: int) -> Set[BloqCountT]:
        return {(self.ops[tuple(kwargs.values())].controlled(), 1)}

    def nth_operation(
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        **kwargs: Union[int, Sequence[cirq.Qid]],
    ) -> cirq.OP_TREE:
        selection_indices = list(cast(int, kwargs[reg.name]) for reg in self.selection_registers)
        targets = {
            reg.name: cast(Sequence[cirq.Qid], kwargs[reg.name]) for reg in self.target_registers
        }
        bloq = self.ops[tuple(selection_indices)]
        target_qubits = merge_qubits(bloq.signature, **targets)
        return bloq.controlled().on(control, *target_qubits)


@bloq_example
def _apply_lth_bloq() -> ApplyLthBloq:
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate

    ops = np.array((TGate(), Hadamard(), ZGate(), XGate()))
    apply_lth_bloq = ApplyLthBloq(ops, control_val=1)
    return apply_lth_bloq


_APPLY_LTH_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ApplyLthBloq,
    import_line='from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq',
    examples=(_apply_lth_bloq,),
)
