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
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    DecomposeTypeError,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol

if TYPE_CHECKING:
    import quimb.tensor as qtn


@frozen
class Allocate(_BookkeepingBloq):
    """Allocate an `n` bit register.

    Args:
        dtype: the quantum data type of the allocated register.

    Registers:
        reg [right]: The allocated register.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('reg', self.dtype, side=Side.RIGHT)])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic.')

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.bookkeeping.free import Free

        return Free(self.dtype)

    def on_classical_vals(self) -> Dict[str, int]:
        return {'reg': 0}

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        data = np.zeros(1 << self.dtype.num_qubits)
        data[0] = 1
        tn.add(qtn.Tensor(data=data, inds=(outgoing['reg'],), tags=['Allocate', tag]))

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        assert reg.name == 'reg'
        return directional_text_box('alloc', Side.RIGHT)


@bloq_example
def _alloc() -> Allocate:
    n = sympy.Symbol('n')
    alloc = Allocate(QUInt(n))
    return alloc


_ALLOC_DOC = BloqDocSpec(bloq_cls=Allocate, examples=[_alloc])
