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
from typing import Dict, List, Optional, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    GateWithRegisters,
    Signature,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import CostKey, GateCounts, QECGatesCost

if TYPE_CHECKING:
    import quimb.tensor as qtn


@frozen(repr=False)
class TestAtom(Bloq):
    """An atomic bloq useful for generic testing and demonstration.

    Args:
        tag: An optional string for differentiating `TestAtom`s.

    Registers:
        q: One bit
    """

    tag: Optional[str] = None

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=np.array([[0, 1], [1, 0]]),
                inds=[(outgoing['q'], 0), (incoming['q'], 0)],
                tags=[str(self)],
            )
        ]

    def my_static_costs(self, cost_key: 'CostKey'):
        if isinstance(cost_key, QECGatesCost):
            return GateCounts(t=100)
        return NotImplemented

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(100)

    def __repr__(self):
        if self.tag:
            return f'TestAtom({self.tag!r})'
        else:
            return 'TestAtom()'

    def __str__(self):
        if self.tag:
            return f'TestAtom({self.tag!r})'
        return 'TestAtom'


@frozen
class TestTwoBitOp(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        _I = [[1, 0], [0, 1]]
        _NULL = [[0, 0], [0, 0]]
        _X = [[0, 1], [1, 0]]
        return [
            qtn.Tensor(
                data=np.array([[_I, _NULL], [_NULL, _X]]),
                inds=[
                    (outgoing['ctrl'], 0),
                    (incoming['ctrl'], 0),
                    (outgoing['target'], 0),
                    (incoming['target'], 0),
                ],
                tags=[str(self)],
            )
        ]


@frozen(repr=False)
class TestGWRAtom(GateWithRegisters):
    """An atomic gate that derives from `GateWithRegisters` useful for testing.

    The single qubit gate has a unitary effect corresponding to a 2x2 identity matrix.

    Args:
        tag: An optional string for differentiating `TestGWRAtom`s.

    Registers:
        q: One bit
    """

    tag: Optional[str] = None
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        from qualtran.cirq_interop._cirq_to_bloq import _my_tensors_from_gate

        return _my_tensors_from_gate(self, self.signature, incoming=incoming, outgoing=outgoing)

    def _unitary_(self):
        return np.eye(2)

    def adjoint(self) -> 'TestGWRAtom':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(100)

    def __repr__(self):
        tag = f'{self.tag!r}' if self.tag else ''
        dagger = '†' if self.is_adjoint else ''
        return f'TestGWRAtom({tag}){dagger}'

    def __str__(self) -> str:
        if self.tag:
            return f'GWRAtom({self.tag})'
        return 'GWRAtom'
