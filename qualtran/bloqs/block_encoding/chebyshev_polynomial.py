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
from typing import Dict, Set, Tuple, TYPE_CHECKING

import attrs

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class ChebyshevPolynomial(BlockEncoding):
    r"""Block encoding of $T_j[A]$ where $T_j$ is the $j$-th Chebyshev polynomial.

    Here $A$ is a Hermitian matrix with spectral norm $|A| \le 1$, we assume we have
    an $n_L$ qubit ancilla register, and assume that $j > 0$ to avoid block
    encoding the identity operator.

    Recall:

    \begin{align*}
        T_0[A] &= I \\
        T_1[A] &= A \\
        T_2[A] &= 2 A^2 - I \\
        T_3[A] &= 4 A^3 - 3 A \\
        &\dots
    \end{align*}

    See https://github.com/quantumlib/Qualtran/issues/984 for an alternative.

    Args:
        block_encoding: Block encoding of a Hermitian matrix $A$, $\mathcal{B}[A]$.
            Assumes the $|G\rangle$ state of the block encoding is the identity operator.
        order: order of Chebychev polynomial.

    References:
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            von Burg et al. 2007. Page 45; Theorem 1.
    """

    block_encoding: BlockEncoding
    order: int

    def __attrs_post_init__(self):
        if self.order < 1:
            raise ValueError(f"order must be greater >= 1. Found {self.order}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    def pretty_name(self) -> str:
        return f"B[T_{self.order}({self.block_encoding.pretty_name()})]"

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.block_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return self.block_encoding.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.block_encoding.alpha**self.order

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.block_encoding.epsilon * self.order

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.signature.rights())

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("resource"),) if self.resource_bitsize > 0 else ()

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),) if self.ancilla_bitsize > 0 else ()

    @property
    def signal_state(self) -> PrepareOracle:
        # This method will be implemented in the future after PrepareOracle
        # is updated for the BlockEncoding interface.
        # Github issue: https://github.com/quantumlib/Qualtran/issues/1104
        raise NotImplementedError

    @cached_property
    def reflection_bloq(self):
        return GlobalPhase(exponent=1).controlled(
            ctrl_spec=CtrlSpec(qdtypes=QAny(self.ancilla_bitsize), cvs=0)
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.ancilla_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        for _ in range(self.order // 2):
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, ctrl=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding, **soqs)
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, ctrl=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding.adjoint(), **soqs)
        if self.order % 2 == 1:
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, ctrl=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding, **soqs)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.order
        s: Set['BloqCountT'] = {
            (self.block_encoding, n // 2 + n % 2),
            (self.block_encoding.adjoint(), n // 2),
        }
        if is_symbolic(self.ancilla_bitsize) or self.ancilla_bitsize > 0:
            s.add((self.reflection_bloq, n - n % 2))
        return s


@bloq_example
def _chebyshev_poly_even() -> ChebyshevPolynomial:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary

    bloq = LinearCombination((Unitary(XGate()), Unitary(Hadamard())), (1.0, 1.0), lambd_bits=1)
    chebyshev_poly_even = ChebyshevPolynomial(bloq, order=4)
    return chebyshev_poly_even


@bloq_example
def _chebyshev_poly_odd() -> ChebyshevPolynomial:
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding import Unitary

    bloq = Unitary(Hadamard())
    chebyshev_poly_odd = ChebyshevPolynomial(bloq, order=5)
    return chebyshev_poly_odd


_CHEBYSHEV_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ChebyshevPolynomial, examples=(_chebyshev_poly_even, _chebyshev_poly_odd)
)
