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
from typing import Any, Dict, Optional, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import Bloq, CompositeBloq, DecomposeTypeError, GateWithRegisters, Signature, SoquetT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

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

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        tn.add(
            qtn.Tensor(
                data=np.array([[0, 1], [1, 0]]),
                inds=(outgoing['q'], incoming['q']),
                tags=[self.pretty_name(), tag],
            )
        )

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(100)

    def __repr__(self):
        if self.tag:
            return f'TestAtom({self.tag!r})'
        else:
            return 'TestAtom()'

    def __str__(self):
        return repr(self)

    def pretty_name(self) -> str:
        if self.tag:
            return self.tag
        else:
            return 'TestAtom'


@frozen
class TestTwoBitOp(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        _I = [[1, 0], [0, 1]]
        _NULL = [[0, 0], [0, 0]]
        _X = [[0, 1], [1, 0]]
        tn.add(
            qtn.Tensor(
                data=np.array([[_I, _NULL], [_NULL, _X]]),
                inds=(outgoing['ctrl'], incoming['ctrl'], outgoing['target'], incoming['target']),
                tags=[self.pretty_name(), tag],
            )
        )

    def pretty_name(self) -> str:
        return 'TestTwoBitOp'


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

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        tn.add(
            qtn.Tensor(
                data=self._unitary_(),
                inds=(outgoing['q'], incoming['q']),
                tags=[self.pretty_name(), tag],
            )
        )

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

    def pretty_name(self) -> str:
        if self.tag:
            return self.tag
        else:
            return 'GWRAtom'
