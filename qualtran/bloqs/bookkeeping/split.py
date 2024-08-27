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
from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QBit,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Split(_BookkeepingBloq):
    """Split a register of a given `dtype` into an array of `QBit`s.

    A logical operation may be defined on e.g. a quantum integer, but to define its decomposition
    we must operate on individual bits. `Split` can be used for this purpose. See `Join` for the
    inverse operation.

    Args:
        dtype: The quantum data type of the incoming data that will be split into an array of `QBit`s.

    Registers:
        reg: The register to be split. On its left, it is of the given data type. On the right, it is an array
            of `QBit()`s of shape `(dtype.num_qubits,)`.
    """

    dtype: QDType = field()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', self.dtype, shape=tuple(), side=Side.LEFT),
                Register('reg', QBit(), shape=(self.dtype.num_qubits,), side=Side.RIGHT),
            ]
        )

    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if value.is_symbolic():
            raise ValueError(f"{self} cannot have a symbolic data type.")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.bookkeeping.join import Join

        return Join(dtype=self.dtype)

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'reg': reg.reshape((self.dtype.num_qubits, 1))}

    def on_classical_vals(self, reg: int) -> Dict[str, 'ClassicalValT']:
        return {'reg': np.asarray(self.dtype.to_bits(reg))}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        eye = np.eye(2)
        incoming = incoming['reg']
        outgoing = cast(NDArray, outgoing['reg'])
        return [
            qtn.Tensor(data=eye, inds=[(outgoing[i], 0), (incoming, i)], tags=[str(self)])
            for i in range(self.dtype.num_qubits)
        ]

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.shape:
            text = f'[{", ".join(str(i) for i in idx)}]'
            return directional_text_box(text, side=reg.side)
        return directional_text_box(' ', side=reg.side)


@bloq_example
def _split() -> Split:
    split = Split(QUInt(4))
    return split


_SPLIT_DOC = BloqDocSpec(bloq_cls=Split, examples=[_split], call_graph_example=None)
