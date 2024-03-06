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
r"""Bloq to reflect about zero."""

from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.drawing import Circle, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class Reflection(Bloq):
    r"""Perform a reflection about zero: $2|0\rangle\langle 0| - 1$

    This is implemented as a large multi-controlled Z operation. It's convenient
    for drawing diagrams to hide the decompostion into MultiControlPauli.

    Args:
        bitsizes: The bitsizes of each of the registers to reflect about.
        cvs: The control values for each register.
    """
    bitsizes: Tuple[int]
    cvs: Tuple[int]

    def __attrs_post_init__(self):
        if len(self.bitsizes) != len(self.cvs):
            raise ValueError(
                f"cvs must be same length as bitsizes: {len(self.cvs)} vs {len(self.bitsizes)}"
            )

    def short_name(self) -> str:
        return 'Refl'

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register(f'reg{i}', QAny(bitsize=b)) for i, b in enumerate(self.bitsizes)]
        )

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        idx = int(soq.pretty()[3:])
        filled = bool(self.cvs[idx])
        return Circle(filled)

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs) -> Dict[str, 'Soquet']:
        unpacked_cvs = sum(((c,) * b for c, b in zip(self.cvs, self.bitsizes)), ())
        # the last qubit is used as the target for the Z
        mcp = MultiControlPauli(cvs=unpacked_cvs[:-1], target_gate=cirq.Z)
        split_regs = np.concatenate([bb.split(r) for r in regs.values()])
        ctrls, target = bb.add(mcp, controls=bb.join(split_regs[:-1]), target=split_regs[-1])
        splt_ctrls = bb.split(ctrls)
        join_regs = np.concatenate([splt_ctrls, [target]])
        out_regs = {}
        start = 0
        for i, b in enumerate(self.bitsizes):
            out_regs[f'reg{i}'] = bb.join(join_regs[start : start + b])
            start += b
        return out_regs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        nbits = sum(self.bitsizes) - 1
        return {(Toffoli(), nbits - 1)}


@bloq_example(generalizer=ignore_split_join)
def _reflection() -> Reflection:
    reflection = Reflection(bitsizes=(2, 3, 1), cvs=(0, 1, 1))
    return reflection


_REFLECTION_DOC = BloqDocSpec(
    bloq_cls=Reflection,
    import_line='from qualtran.bloqs.reflection import Reflection',
    examples=(_reflection,),
)
