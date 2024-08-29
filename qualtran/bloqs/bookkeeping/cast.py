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
from typing import Dict, List, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QDType,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Cast(_BookkeepingBloq):
    """Cast a register from one n-bit QDType to another QDType.

    This re-interprets the register's data type from `inp_dtype` to `out_dtype`.

    Args:
        inp_dtype: Input QDType to cast from.
        out_dtype: Output QDType to cast to.
        shape: shape of the register to cast.

    Registers:
        in: input register to cast from.
        out: input register to cast to.
    """

    inp_dtype: QDType
    out_dtype: QDType
    shape: Tuple[int, ...] = attrs.field(
        default=tuple(), converter=lambda v: (v,) if isinstance(v, int) else tuple(v)
    )

    def __attrs_post_init__(self):
        if isinstance(self.inp_dtype.num_qubits, int):
            if self.inp_dtype.num_qubits != self.out_dtype.num_qubits:
                raise ValueError("Casting only permitted between same sized registers.")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', dtype=self.inp_dtype, shape=self.shape, side=Side.LEFT),
                Register('reg', dtype=self.out_dtype, shape=self.shape, side=Side.RIGHT),
            ]
        )

    def adjoint(self) -> 'Bloq':
        return Cast(inp_dtype=self.out_dtype, out_dtype=self.inp_dtype)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=np.eye(2), inds=[(outgoing['reg'], j), (incoming['reg'], j)], tags=[str(self)]
            )
            for j in range(self.inp_dtype.num_qubits)
        ]

    def on_classical_vals(self, reg: int) -> Dict[str, 'ClassicalValT']:
        res = self.out_dtype.from_bits(self.inp_dtype.to_bits(reg))
        return {'reg': res}

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'reg': reg}


@bloq_example
def _cast() -> Cast:
    from qualtran import QFxp, QInt

    cast = Cast(QInt(32), QFxp(32, 32))
    return cast


_CAST_DOC = BloqDocSpec(bloq_cls=Cast, examples=[_cast], call_graph_example=None)
