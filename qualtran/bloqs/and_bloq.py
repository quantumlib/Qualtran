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
class And(Bloq):
    """A two-bit 'and' operation.

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:
     - ctrl: A two-bit control register.
     - (right) target: The output bit.

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
        (Verifying Measurement Based Uncomputation)[https://algassert.com/post/1903].
            Gidney, C. 2019.
    """

    cv1: int = 1
    cv2: int = 1
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


@frozen
class MultiAnd(Bloq):
    """A many-bit (multi-control) 'and' operation.

    Args:
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.

    Registers:
     - ctrl: An n-bit control register.
     - (right) An `n-2` bit junk register to be cleaned up by the inverse operation.
     - (right) target: The output bit.
    """

    cvs: Tuple[int, ...] = field(validator=lambda i, f, v: len(v) >= 3, converter=tuple)
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        one_side = Side.RIGHT if not self.adjoint else Side.LEFT
        return Signature(
            [
                Register('ctrl', 1, shape=(len(self.cvs),)),
                Register('junk', 1, shape=(len(self.cvs) - 2,), side=one_side),
                Register('target', 1, side=one_side),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'And{dag}'

    def decompose_bloq(self) -> 'CompositeBloq':
        cbloq = super().decompose_bloq()
        if self.adjoint:
            raise NotImplementedError("Come back soon.")
        return cbloq

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, ctrl: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        # 'and' the first two control lines together into an ancilla.
        cv1, cv2, *_ = self.cvs
        c1c2, anc = bb.add(And(cv1=cv1, cv2=cv2), ctrl=ctrl[:2])

        if len(self.cvs) == 3:
            # Base case: add a final `And` to complete the ladder.
            (anc, c3), target = bb.add(And(cv1=1, cv2=self.cvs[2]), ctrl=[anc, ctrl[2]])
            return {
                'ctrl': np.concatenate((c1c2, [c3])),
                'junk': np.asarray([anc]),
                'target': target,
            }

        # Recursive step: Replace the first two controls with the ancilla.
        # Note: change `bb.add_from` to `bb.add` to decompose one recursive step at a time.
        (anc, *c_rest), junk, target = bb.add_from(
            MultiAnd(cvs=(1, *self.cvs[2:])), ctrl=np.concatenate(([anc], ctrl[2:]))
        )
        return {
            'ctrl': np.concatenate((c1c2, c_rest)),
            'junk': np.concatenate(([anc], junk)),
            'target': target,
        }

    def on_classical_vals(self, ctrl: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        if self.adjoint:
            raise NotImplementedError("Come back later.")

        accumulate_and = np.bitwise_and.accumulate(np.equal(ctrl, self.cvs).astype(np.uint8))
        junk, target = accumulate_and[1:-1], accumulate_and[-1]
        return {'ctrl': ctrl, 'junk': junk, 'target': target}
