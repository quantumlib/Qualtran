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
from typing import Any, Dict, Set, TYPE_CHECKING

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates.cnot import CNOT
from qualtran.bloqs.basic_gates.hadamard import Hadamard
from qualtran.bloqs.basic_gates.rotation import Rz
from qualtran.bloqs.basic_gates.s_gate import SGate
from qualtran.bloqs.basic_gates.t_gate import TGate
from qualtran.resource_counting import SympySymbolAllocator

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def _fkn_matrix(k: int, n: int) -> NDArray[np.complex128]:
    sqrt_two = np.sqrt(2)
    exp = np.exp(2 * np.pi * 1j * k / n)
    x = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / sqrt_two, exp / sqrt_two, 0],
            [0, 1 / sqrt_two, -exp / sqrt_two, 0],
            [0, 0, 0, -exp],
        ]
    )
    return x


@frozen
class TwoBitFFFT(Bloq):
    r"""Two-qubit fermionic Fourier transform gate.

    Args:
        k: An integer.
        n: The number of qubits the FFFT acts on.
        eps: The rotation precision.
        is_adjoint: If True, this bloq is $F^\dagger$ instead.

    References:
        [Improved Fault-Tolerant Quantum Simulation of Condensed-Phase Correlated Electrons
            via Trotterization](https://arxiv.org/abs/1902.10673). Eq 32 and Figure 8.
    """

    k: int
    n: int
    eps: float = 1e-10
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        if self.n == 0:
            raise ValueError("n must be greater than zero.")

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', QBit()), Register('y', QBit())])

    def short_name(self) -> str:
        return 'F(k, n)'

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        out_inds = [outgoing['x'], outgoing['y']]
        in_inds = [incoming['x'], incoming['y']]
        matrix = _fkn_matrix(self.k, self.n)
        matrix = matrix.conj().T if self.is_adjoint else matrix
        tn.add(
            qtn.Tensor(
                data=matrix.reshape((2,) * 4),
                inds=out_inds + in_inds,
                tags=[self.short_name(), tag],
            )
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (Rz(2 * np.pi * self.k / self.n, eps=self.eps), 1),
            (SGate(), 3),
            (Hadamard(), 6),
            (TGate(), 2),
            (CNOT(), 3),
        }

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        x = bb.add(Rz(2 * np.pi * self.k / self.n, eps=self.eps), q=x)
        y = bb.add(SGate(), q=y)
        x = bb.add(Hadamard(), q=x)
        y = bb.add(Hadamard(), q=y)
        x, y = bb.add(CNOT(), ctrl=x, target=y)
        x = bb.add(Hadamard(), q=x)
        x = bb.add(SGate(), q=x)
        y = bb.add(TGate().adjoint(), q=y)
        x, y = bb.add(CNOT(), ctrl=x, target=y)
        x = bb.add(Hadamard(), q=x)
        y = bb.add(TGate(), q=y)
        x, y = bb.add(CNOT(), ctrl=x, target=y)
        x = bb.add(Hadamard(), q=x)
        y = bb.add(Hadamard(), q=y)
        x = bb.add(SGate(), q=x)

        return {'x': x, 'y': y}


@bloq_example
def _two_bit_ffft() -> TwoBitFFFT:
    two_bit_ffft = TwoBitFFFT(2, 3)
    return two_bit_ffft


_TWO_BIT_FFFT_DOC = BloqDocSpec(
    bloq_cls=TwoBitFFFT,
    import_line='from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT',
    examples=[_two_bit_ffft],
)
