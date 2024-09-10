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
from typing import Dict, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import (
        BloqCountDictT,
        MutableBloqCountDictT,
        SympySymbolAllocator,
    )


@attrs.frozen
class ChebyshevPolynomial(BlockEncoding):
    r"""Block encoding of $T_j(A / \alpha)$ where $T_j$ is the $j$-th Chebyshev polynomial.

    Given a Hermitian matrix $A$ with spectral norm $|A| \le 1$, recall:

    \begin{align*}
        T_0[A] &= I \\
        T_1[A] &= A \\
        T_2[A] &= 2 A^2 - I \\
        T_3[A] &= 4 A^3 - 3 A \\
        &\dots
    \end{align*}

    If `block_encoding` block encodes $A$ with normalization factor $\alpha$, i.e. it constructs
    $\mathcal{B}[A/\alpha]$, then this bloq constructs $\mathcal{B}[T_j(A/\alpha)]$ with
    normalization factor 1. Note that $\mathcal{B}[T_j(A/\alpha)]$ is not a multiple of
    $\mathcal{B}[T_j(A)]$ in general; use `ScaledChebyshevPolynomial` if $\alpha \neq 1$.

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
        if self.order < 0:
            raise ValueError(f"order must be greater >= 0. Found {self.order}.")
        if not isinstance(self.block_encoding.signal_state.prepare, PrepareIdentity):
            raise ValueError(
                "Cannot take Chebyshev polynomial of block encodings with non-zero signal state."
            )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    def pretty_name(self) -> str:
        return f"B[T_{self.order}({self.block_encoding})]"

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
        return 1

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.block_encoding.epsilon * self.order

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @cached_property
    def reflection_bloq(self):
        return ReflectionUsingPrepare.reflection_around_zero(
            bitsizes=(self.ancilla_bitsize,), global_phase=-1
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.ancilla_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        for _ in range(self.order // 2):
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, reg0_=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding, **soqs)
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, reg0_=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding.adjoint(), **soqs)
        if self.order % 2 == 1:
            if self.ancilla_bitsize > 0:
                soqs["ancilla"] = bb.add(self.reflection_bloq, reg0_=soqs["ancilla"])
            soqs |= bb.add_d(self.block_encoding, **soqs)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n = self.order
        s: 'MutableBloqCountDictT' = {
            self.block_encoding: n // 2 + n % 2,
            self.block_encoding.adjoint(): n // 2,
        }
        if is_symbolic(self.ancilla_bitsize) or self.ancilla_bitsize > 0:
            s[self.reflection_bloq] = n - n % 2
        return s


@bloq_example
def _chebyshev_poly_even() -> ChebyshevPolynomial:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary

    bloq = LinearCombination((Unitary(XGate()), Unitary(Hadamard())), (0.5, 0.5), lambd_bits=1)
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


@attrs.frozen
class ScaledChebyshevPolynomial(BlockEncoding):
    r"""Block encoding of $T_j(A)$ where $T_j$ is the $j$-th Chebyshev polynomial.

    Unlike `ChebyshevPolynomial`, this bloq accepts $\mathcal{B}[A/\alpha]$ with $\alpha \neq 1$
    and constructs $\mathcal{B}[T_j(A)]$ which is not a multiple of $\mathcal{B}[T_j(A/\alpha)]$
    in general. It does so by constructing $T_k(t)$ in terms of $T_j(t/\alpha)$ for $j \in [0, k]$.

    Args:
        block_encoding: Block encoding of a Hermitian matrix $A$, $\mathcal{B}[A/\alpha]$.
            Assumes the $|G\rangle$ state of the block encoding is the identity operator.
        order: order of Chebychev polynomial.
        lambd_bits: number of bits to represent coefficients of linear combination precisely.

    References:
        [Explicit Quantum Circuits for Block Encodings of Certain Sparse Matrices](https://arxiv.org/abs/2203.10236). Camps et al. (2023). Section 5.1.
    """

    block_encoding: BlockEncoding
    order: int
    lambd_bits: int = 5

    def __attrs_post_init__(self):
        if self.order < 0:
            raise ValueError(f"order must be greater >= 0. Found {self.order}.")
        if not isinstance(self.block_encoding.signal_state.prepare, PrepareIdentity):
            raise ValueError(
                "Cannot take Chebyshev polynomial of block encodings with non-zero signal state."
            )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.linear_combination.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.linear_combination.ancilla_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return self.linear_combination.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.linear_combination.alpha

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.linear_combination.epsilon

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
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @cached_property
    def linear_combination(self) -> Union[LinearCombination, ChebyshevPolynomial]:
        if self.order <= 1:
            return ChebyshevPolynomial(self.block_encoding, self.order)

        coeffs = np.polynomial.chebyshev.cheb2poly([0] * self.order + [1])
        for i in range(len(coeffs)):
            coeffs[i] *= self.block_encoding.alpha**i
        cheb_coeffs = np.polynomial.chebyshev.poly2cheb(coeffs)

        return LinearCombination(
            tuple(ChebyshevPolynomial(self.block_encoding, i) for i in range(self.order + 1)),
            cheb_coeffs,
            self.lambd_bits,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        return bb.add_d(self.linear_combination, **soqs)

    def __str__(self) -> str:
        return f"B[T_{self.order}({self.block_encoding})]"


@bloq_example
def _scaled_chebyshev_poly_even() -> ScaledChebyshevPolynomial:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary

    bloq = LinearCombination((Unitary(XGate()), Unitary(Hadamard())), (1.0, 1.0), lambd_bits=1)
    scaled_chebyshev_poly_even = ScaledChebyshevPolynomial(bloq, order=4)
    return scaled_chebyshev_poly_even


@bloq_example
def _scaled_chebyshev_poly_odd() -> ScaledChebyshevPolynomial:
    from attrs import evolve

    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding import Unitary

    bloq = evolve(Unitary(Hadamard()), alpha=3.14)
    scaled_chebyshev_poly_odd = ScaledChebyshevPolynomial(bloq, order=5)
    return scaled_chebyshev_poly_odd


_SCALED_CHEBYSHEV_BLOQ_DOC = BloqDocSpec(
    bloq_cls=ScaledChebyshevPolynomial,
    examples=(_scaled_chebyshev_poly_even, _scaled_chebyshev_poly_odd),
)
