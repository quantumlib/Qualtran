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
from typing import Optional, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import Bloq, ConnectionT, DecomposeTypeError, QDType, QFxp, Register, Side, Signature
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.resource_counting.generalizers import _ignore_wrapper
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn


@attrs.frozen
class AcquirePhaseGradient(_BookkeepingBloq):
    r"""Acquire a phase gradient state, used to perform rotations.

    This is treated as a book-keeping bloq at the moment, because it should
    be accounted for as a one-time preparation cost.
    If we directly used allocations of `PhaseGradientState` each time in
    a large algorithm, their costs would add up, which is incorrect.

    To capture the required resource costs of an algorithm, we can use a separate
    cost key that only counts this bloq, and takes the max phase gradient size
    and computes the cost to synthesize it.

    Args:
        bitsize: number of bits of the phase gradient $b_\text{grad}$.
        eps: the precision to synthesize the state.

    Registers:
        phase_grad: A phase gradient state of `bitsize`, $|\phi_\text{grad}\rangle$.
    """

    bitsize: SymbolicInt
    eps: float = 1e-10

    @property
    def signature(self) -> 'Signature':
        return Signature([Register('phase_grad', self.phase_dtype, side=Side.RIGHT)])

    @property
    def phase_dtype(self) -> QDType:
        return QFxp(self.bitsize, self.bitsize)

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"cannot compute tensors for symbolic {self}")

        return [
            qtn.Tensor(
                data=np.array([1, np.exp(-1j * np.pi / 2**i)]) / np.sqrt(2),
                inds=[(outgoing['phase_grad'], i)],
                tags=[f'pg_{i}'],
            )
            for i in range(self.bitsize)
        ]

    @property
    def prepare(self) -> PhaseGradientState:
        """bloq to prepare the actual phase gradient state"""
        return PhaseGradientState(self.bitsize, eps=self.eps)

    def adjoint(self) -> 'ReleasePhaseGradient':
        return ReleasePhaseGradient(self.bitsize, self.eps)


@attrs.frozen
class ReleasePhaseGradient(_BookkeepingBloq):
    """Release a phase gradient resource state.

    Helper to make diagrams cleaner. See `AcquirePhaseGradient` for details.
    """

    bitsize: SymbolicInt
    eps: float = 1e-10

    @property
    def signature(self) -> 'Signature':
        return Signature([Register('phase_grad', self.phase_dtype, side=Side.LEFT)])

    @property
    def phase_dtype(self) -> QDType:
        return QFxp(self.bitsize, self.bitsize)

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"cannot compute tensors for symbolic {self}")

        return [
            qtn.Tensor(
                data=np.array([1, np.exp(1j * np.pi / 2**i)]) / np.sqrt(2),
                inds=[(incoming['phase_grad'], i)],
                tags=[f'pg_{i}'],
            )
            for i in range(self.bitsize)
        ]

    def adjoint(self) -> 'AcquirePhaseGradient':
        return AcquirePhaseGradient(self.bitsize, self.eps)


def ignore_resource_alloc_free(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores split and join operations."""
    if isinstance(b, (AcquirePhaseGradient, ReleasePhaseGradient)):
        return None

    return _ignore_wrapper(ignore_resource_alloc_free, b)
