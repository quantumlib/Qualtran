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
from typing import Dict, Optional, Tuple

import attrs
import numpy as np

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt import And, MultiTargetCNOT
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import SymbolicInt


@attrs.frozen
class CSwapViaAnd(Bloq):
    """CSWAP(a, b, c) when c is guaranteed to be 0."""

    cvs: tuple[int, int] = (1, 1)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit()),
                Register('x', QBit()),
                Register('y', QBit(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        (ctrl, x), y = bb.add(And(*self.cvs), ctrl=[ctrl, x])
        y, x = bb.add(CNOT(), ctrl=y, target=x)
        return {'ctrl': ctrl, 'x': x, 'y': y}

    def wire_symbol(self, reg: Optional['Register'], idx: Tuple[int, ...] = ()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle(filled=True)
        else:
            return TextBox('×')


@attrs.frozen
class OneHotLinearDepth(Bloq):
    r"""Linear depth one hot encoding using N - 1 CSWAPs."""

    selection_dtype: BQUInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.selection_dtype),
                Register(
                    'out', QBit(), shape=(self.selection_dtype.iteration_length,), side=Side.RIGHT
                ),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'SoquetT') -> Dict[str, 'SoquetT']:
        x = bb.split(x)[::-1]
        out = [bb.allocate(dtype=QBit())]
        out[0] = bb.add(XGate(), q=out[0])
        for i in range(len(x)):
            new_out = []
            for j in range(2**i):
                if j + 2**i < self.selection_dtype.iteration_length:
                    x[i], out[j], out_k = bb.add(CSwapViaAnd(), ctrl=x[i], x=out[j])
                    new_out.append(out_k)
            out.extend(new_out)
        return {'x': bb.join(x[::-1], dtype=self.selection_dtype), 'out': np.array(out)}


@bloq_example(generalizer=[ignore_split_join])
def _one_hot_linear_depth() -> OneHotLinearDepth:
    from qualtran import BQUInt

    one_hot_linear_depth = OneHotLinearDepth(BQUInt(4, 14))
    return one_hot_linear_depth


@attrs.frozen
class Fanout(Bloq):
    """Fanout via Multi-Target CNOT"""

    n_copies: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('x', QBit()), Register('y', QBit(), shape=(self.n_copies,), side=Side.RIGHT)]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'SoquetT') -> Dict[str, 'SoquetT']:
        y = bb.allocate(self.n_copies)
        x, y = bb.add(MultiTargetCNOT(self.n_copies), control=x, targets=y)
        return {'x': x, 'y': bb.split(y)}


@attrs.frozen
class OneHotLogDepth(Bloq):
    r"""Log depth one hot encoding using N - 1 CSWAPs."""

    selection_dtype: BQUInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.selection_dtype),
                Register(
                    'out', QBit(), shape=(self.selection_dtype.iteration_length,), side=Side.RIGHT
                ),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'SoquetT') -> Dict[str, 'SoquetT']:
        x = bb.split(x)[::-1]
        out = [bb.allocate(dtype=QBit())]
        out[0] = bb.add(XGate(), q=out[0])
        for i in range(len(x)):
            new_out = []
            n_alloc = max(0, min(self.selection_dtype.iteration_length, 2 ** (i + 1)) - 2**i)
            if n_alloc:
                x[i], xx = bb.add(Fanout(n_alloc), x=x[i])
            for j in range(n_alloc):
                assert j + 2**i < self.selection_dtype.iteration_length
                xx[j], out[j], out_k = bb.add(CSwapViaAnd(), ctrl=xx[j], x=out[j])
                new_out.append(out_k)
            out.extend(new_out)
            if n_alloc:
                x[i] = bb.add(Fanout(n_alloc).adjoint(), x=x[i], y=xx)
        return {'x': bb.join(x[::-1], dtype=self.selection_dtype), 'out': np.array(out)}


@bloq_example(generalizer=[ignore_split_join])
def _one_hot_log_depth() -> OneHotLogDepth:
    from qualtran import BQUInt

    one_hot_log_depth = OneHotLogDepth(BQUInt(4, 14))
    return one_hot_log_depth


_ONE_HOT_LINEAR_DEPTH_DOC = BloqDocSpec(
    bloq_cls=OneHotLinearDepth, examples=(_one_hot_linear_depth,)
)
_ONE_HOT_LOG_DEPTH_DOC = BloqDocSpec(bloq_cls=OneHotLogDepth, examples=(_one_hot_log_depth,))
