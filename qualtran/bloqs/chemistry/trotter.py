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

import itertools
from functools import cached_property
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import quimb.tensor as qtn
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, Register, Side, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.drawing import Circle, directional_text_box, WireSymbol
from qualtran.resource_counting import big_O, SympySymbolAllocator


@frozen
class FunctionInterpolation(Bloq):
    """

    Args:

    Registers:
     -
     -

    References:
        (Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial
            Optimization)[https://arxiv.org/pdf/2007.07391.pdf].
    """

    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('ctrl', 1, shape=(2,)),
                Register('target', 1, side=Side.RIGHT if not self.adjoint else Side.LEFT),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if isinstance(self.cv1, sympy.Expr) or isinstance(self.cv2, sympy.Expr):
            pre_post_cliffords = big_O(1)
        else:
            pre_post_cliffords = 2 - self.cv1 - self.cv2
        if self.adjoint:
            return {(4 + 2 * pre_post_cliffords, ArbitraryClifford(n=2))}

        return {(9 + 2 * pre_post_cliffords, ArbitraryClifford(n=2)), (4, TGate())}

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'And{dag}'

    def on_classical_vals(self, ctrl: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        if self.adjoint:
            raise NotImplementedError("Come back later.")

        target = 1 if tuple(ctrl) == (self.cv1, self.cv2) else 0
        return {'ctrl': ctrl, 'target': target}

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        # Fill in our tensor using "and" logic.
        data = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        for c1, c2 in itertools.product((0, 1), repeat=2):
            if c1 == self.cv1 and c2 == self.cv2:
                data[c1, c2, c1, c2, 1] = 1
            else:
                data[c1, c2, c1, c2, 0] = 1

        # Here: adjoint just switches the direction of the target index.
        if self.adjoint:
            trg = incoming['target']
        else:
            trg = outgoing['target']

        tn.add(
            qtn.Tensor(
                data=data,
                inds=(
                    incoming['ctrl'][0],
                    incoming['ctrl'][1],
                    outgoing['ctrl'][0],
                    outgoing['ctrl'][1],
                    trg,
                ),
                tags=['And', tag],
            )
        )

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'target':
            return directional_text_box('∧', side=soq.reg.side)

        (c_idx,) = soq.idx
        filled = bool(self.cv1 if c_idx == 0 else self.cv2)
        return Circle(filled)
