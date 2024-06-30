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
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    DecomposeTypeError,
    QBit,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.simulation.classical_sim import bits_to_ints

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from numpy.typing import NDArray

    from qualtran.cirq_interop import CirqQuregT


@frozen
class Join(_BookkeepingBloq):
    """Join an array of `QBit`s into one register of type `dtype`.

    Args:
        dtype: The quantum data type of the right (joined) register.

    Registers:
        reg: The register to be joined. On its left, it is an array of qubits. On the right, it is a register
            of the given data type.
    """

    dtype: QDType = field()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', QBit(), shape=(self.dtype.num_qubits,), side=Side.LEFT),
                Register('reg', self.dtype, shape=tuple(), side=Side.RIGHT),
            ]
        )

    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if value.is_symbolic():
            raise ValueError(f"{self} cannot have a symbolic data type.")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self) -> 'Bloq':
        from qualtran.bloqs.bookkeeping.split import Split

        return Split(dtype=self.dtype)

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'reg': reg.reshape(self.dtype.num_qubits)}

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        if not isinstance(incoming['reg'], np.ndarray):
            raise ValueError('Incoming register must be a numpy array')
        tn.add(
            qtn.Tensor(
                data=np.eye(2**self.dtype.num_qubits, 2**self.dtype.num_qubits).reshape(
                    (2,) * self.dtype.num_qubits + (2**self.dtype.num_qubits,)
                ),
                inds=incoming['reg'].tolist() + [outgoing['reg']],
                tags=['Join', tag],
            )
        )

    def on_classical_vals(self, reg: 'NDArray[np.uint]') -> Dict[str, int]:
        return {'reg': bits_to_ints(reg)[0]}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.shape:
            text = f'[{", ".join(str(i) for i in idx)}]'
            return directional_text_box(text, side=reg.side)
        return directional_text_box(' ', side=reg.side)


@bloq_example
def _join() -> Join:
    join = Join(dtype=QUInt(4))
    return join


_JOIN_DOC = BloqDocSpec(bloq_cls=Join, examples=[_join], call_graph_example=None)
