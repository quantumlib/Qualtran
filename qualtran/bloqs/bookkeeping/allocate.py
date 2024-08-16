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
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT


@frozen
class Allocate(_BookkeepingBloq):
    """Allocate an `n` bit register.

    Args:
        dtype: the quantum data type of the allocated register.
        dirty: If true, represents a borrowing operation where allocated qubits can be dirty.

    Registers:
        reg [right]: The allocated register.
    """

    dtype: QDType
    dirty: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('reg', self.dtype, side=Side.RIGHT)])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic.')

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.bookkeeping.free import Free

        return Free(self.dtype, self.dirty)

    def on_classical_vals(self) -> Dict[str, int]:
        return {'reg': 0}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        from qualtran.bloqs.basic_gates.z_basis import _ZERO

        return [
            qtn.Tensor(data=_ZERO, inds=[(outgoing['reg'], i)], tags=[str(self)])
            for i in range(self.dtype.num_qubits)
        ]

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        assert reg.name == 'reg'
        return directional_text_box('alloc', Side.RIGHT)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        shape = (*self.signature[0].shape, self.signature[0].bitsize)
        qubits = (
            qubit_manager.qborrow(self.signature.n_qubits())
            if self.dirty
            else qubit_manager.qalloc(self.signature.n_qubits())
        )
        return (None, {'reg': np.array(qubits).reshape(shape)})


@bloq_example
def _alloc() -> Allocate:
    n = sympy.Symbol('n')
    alloc = Allocate(QUInt(n))
    return alloc


_ALLOC_DOC = BloqDocSpec(bloq_cls=Allocate, examples=[_alloc])
