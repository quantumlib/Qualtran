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
from collections import Counter
from typing import Union

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    QBit,
    QInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import LinearDepthHalfLessThan
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import HasLength, is_symbolic, SymbolicInt


@frozen
class HasDuplicates(Bloq):
    r"""Given a sorted list of `l` numbers, check if it contains any duplicates.

    Produces a single qubit which is `1` if there are duplicates, and `0` if all are disjoint.
    It compares every adjacent pair, and therefore uses `l - 1` comparisons.
    It then uses a single MCX on `l - 1` bits gate to compute the flag.

    Args:
        l: number of elements in the list
        dtype: type of each element to store `[n]`.

    Registers:
        xs: a list of `l` registers of `dtype`.
        flag: single qubit. Value is flipped if the input list has duplicates, otherwise stays same.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Lemma 4.12. Eq. 122.
    """

    l: SymbolicInt
    dtype: Union[QUInt, QInt]
    is_controlled: bool = False

    @property
    def signature(self) -> 'Signature':
        registers = [Register('xs', self.dtype, shape=(self.l,)), Register('flag', QBit())]
        if self.is_controlled:
            registers.append(Register('ctrl', QBit()))
        return Signature(registers)

    @property
    def _le_bloq(self) -> LinearDepthHalfLessThan:
        return LinearDepthHalfLessThan(self.dtype)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', xs: 'SoquetT', flag: 'Soquet', **extra_soqs: 'SoquetT'
    ) -> dict[str, 'SoquetT']:
        assert not is_symbolic(self.l)
        assert isinstance(xs, np.ndarray)

        cs = []
        oks = []
        if self.is_controlled:
            oks = [extra_soqs.pop('ctrl')]
        assert not extra_soqs

        for i in range(1, self.l):
            xs[i - 1], xs[i], c, ok = bb.add(self._le_bloq, a=xs[i - 1], b=xs[i])
            cs.append(c)
            oks.append(ok)

        oks, flag = bb.add(MultiControlX((1,) * len(oks)), controls=np.array(oks), target=flag)
        if not self.is_controlled:
            flag = bb.add(XGate(), q=flag)
        else:
            oks[0], flag = bb.add(CNOT(), ctrl=oks[0], target=flag)

        oks = list(oks)
        for i in reversed(range(1, self.l)):
            xs[i - 1], xs[i] = bb.add(
                self._le_bloq.adjoint(), a=xs[i - 1], b=xs[i], c=cs.pop(), target=oks.pop()
            )

        if self.is_controlled:
            extra_soqs = {'ctrl': oks.pop()}
        assert not oks

        return {'xs': xs, 'flag': flag} | extra_soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        counts = Counter[Bloq]()

        counts[self._le_bloq] += self.l - 1
        counts[self._le_bloq.adjoint()] += self.l - 1

        n_ctrls = self.l - (1 if not self.is_controlled else 0)
        counts[MultiControlX(HasLength(n_ctrls))] += 1

        counts[XGate() if not self.is_controlled else CNOT()] += 1

        return counts

    def on_classical_vals(self, **vals: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        xs = np.asarray(vals['xs'])
        assert np.all(xs == np.sort(xs))
        if np.any(xs[:-1] == xs[1:]):
            vals['flag'] ^= 1
        return vals

    def adjoint(self) -> 'HasDuplicates':
        return self

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True),
            ctrl_reg_name='ctrl',
        )


@bloq_example
def _has_duplicates() -> HasDuplicates:
    has_duplicates = HasDuplicates(4, QUInt(3))
    return has_duplicates


@bloq_example
def _has_duplicates_symb() -> HasDuplicates:
    import sympy

    n = sympy.Symbol("n")
    has_duplicates_symb = HasDuplicates(4, QUInt(n))
    return has_duplicates_symb


@bloq_example
def _has_duplicates_symb_len() -> HasDuplicates:
    import sympy

    l, n = sympy.symbols("l n")
    has_duplicates_symb_len = HasDuplicates(l, QUInt(n))
    return has_duplicates_symb_len


_HAS_DUPLICATES_DOC = BloqDocSpec(
    bloq_cls=HasDuplicates, examples=[_has_duplicates_symb, _has_duplicates]
)
