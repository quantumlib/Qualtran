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
from typing import Sequence, Tuple

import cirq
import numpy as np
from attrs import field, frozen, validators

from qualtran import Bloq, bloq_example, BloqDocSpec, BoundedQUInt, QBit, Register, Side
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate


@frozen
class ApplyLthBloq(UnaryIterationGate):
    r"""A SELECT operation that executes one of a list of bloqs $U_l$ based on a quantum index:

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    This bloq uses the unary iteration scheme to apply `ops[selection]` controlled on an optional single-bit `control` register.

    Args:
        ops: List of bloqs to select from, each with identical registers that are all THRU.
        control: If True, a singly controlled gate is constructed.

    Registers:
        selection: The index of the bloq in `ops` to execute.
        control: The control bit if specified above.
        [user_spec]: The output registers of the bloqs in `ops`.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.A. and Figure 7.
    """

    ops: Tuple[Bloq, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )
    control: bool = False

    def __attrs_post_init__(self):
        if not all(u.signature == self.ops[0].signature for u in self.ops):
            raise ValueError("All ops must have the same signature.")
        if not all(r.side == Side.THRU for u in self.ops for r in u.signature):
            raise ValueError("All ops must have only THRU registers.")

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (Register('control', QBit()),) if self.control else ()

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                'selection', BoundedQUInt(int(np.ceil(np.log2(len(self.ops)))), len(self.ops))
            ),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.ops[0].signature)

    def nth_operation(
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        selection: int,
        **targets: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        return self.ops[selection].on_registers(**targets).controlled_by(control)


@bloq_example
def _apply_lth_bloq() -> ApplyLthBloq:
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate

    ops = (TGate(), Hadamard(), ZGate(), XGate())
    apply_lth_bloq = ApplyLthBloq(ops=ops, control=True)
    return apply_lth_bloq


_APPLY_LTH_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ApplyLthBloq,
    import_line='from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq',
    examples=(_apply_lth_bloq,),
)
