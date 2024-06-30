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

    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Free(_BookkeepingBloq):
    """Free (i.e. de-allocate) a register.

    The tensor decomposition assumes the register is uncomputed and is in the zero
    state before getting freed. To verify that is the case, one can compute the resulting state
    vector after freeing qubits and make sure it is normalized.

    Args:
        dtype: The quantum data type of the register to be freed.

    Registers:
        reg [left]: The register to free.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('reg', self.dtype, side=Side.LEFT)])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic.')

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.bookkeeping.allocate import Allocate

        return Allocate(self.dtype)

    def on_classical_vals(self, reg: int) -> Dict[str, 'ClassicalValT']:
        if reg != 0:
            raise ValueError(f"Tried to free a non-zero register: {reg}.")
        return {}

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
        tn.add(qtn.Tensor(data=data, inds=(incoming['reg'],), tags=['Free', tag]))

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        assert reg.name == 'reg'
        return directional_text_box('free', Side.LEFT)


@bloq_example
def _free() -> Free:
    n = sympy.Symbol('n')
    free = Free(QUInt(n))
    return free


_FREE_DOC = BloqDocSpec(bloq_cls=Free, examples=[_free])
