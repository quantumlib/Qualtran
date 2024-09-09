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
from typing import Sequence, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, DecomposeTypeError, QAny, QUInt, Register, Signature
from qualtran.bloqs.arithmetic.permutation import Permutation
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
    _to_tuple_or_has_length,
    StatePreparationViaRotations,
)
from qualtran.symbolics import bit_length, HasLength, is_symbolic, slen, SymbolicInt

if TYPE_CHECKING:
    import scipy

    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class SparseStatePreparationViaRotations(Bloq):
    r"""Prepares a $d$-sparse state on $n$ qubits using rotations and basis permutation.

    This bloq prepares a $d$-sparse state on $n$ qubits:

        $$
            \ket{\psi} = \sum_{x \in S} \alpha_x \ket{x}
        $$

    where $S \subseteq [N]$ such that $|S| = d$.

    To achieve this, it first prepares a dense state on $\ceil{\log d}$ qubits, then
    permutes the basis s.t. $i \mapsto x_i$, where $x_i$ is the $i$-th element in S.

    References:
        [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309)
        Ramacciotti et. al. Section 4 "Permutation Grover-Rudolph".
    """
    sparse_indices: Union[tuple[int, ...], HasLength] = field(converter=_to_tuple_or_has_length)
    nonzero_coeffs: Union[tuple[complex, ...], HasLength] = field(converter=_to_tuple_or_has_length)
    N: SymbolicInt
    phase_bitsize: SymbolicInt

    def __attrs_post_init__(self):
        n_idx = slen(self.sparse_indices)
        n_coeff = slen(self.nonzero_coeffs)
        if not is_symbolic(n_idx, n_coeff) and n_idx != n_coeff:
            raise ValueError(f"Number of indices {n_idx} must equal number of coeffs {n_coeff}")

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            target_state=QUInt(self.target_bitsize), phase_gradient=QAny(self.phase_bitsize)
        )

    @property
    def target_bitsize(self) -> SymbolicInt:
        return bit_length(self.N - 1)

    @property
    def dense_bitsize(self) -> SymbolicInt:
        return bit_length(slen(self.sparse_indices) - 1)

    def is_symbolic(self):
        return is_symbolic(self.sparse_indices, self.nonzero_coeffs, self.N)

    @classmethod
    def from_coefficient_map(
        cls, N: SymbolicInt, coeff_map: dict[int, complex], phase_bitsize: SymbolicInt
    ):
        """Factory to construct sparse state from a dictionary of coefficients.

        Args:
            N: the total size of the state `N`.
            coeff_map: dictionary mapping index `i` to non-zero coefficient `c_i`
            phase_bitsize: size of the phase-gradient register.
        """
        return cls(tuple(coeff_map.keys()), tuple(coeff_map.values()), N, phase_bitsize)

    @classmethod
    def from_sparse_array(
        cls,
        coeffs: Union[Sequence[complex], NDArray[np.complex128], 'scipy.sparse.sparray'],
        phase_bitsize: SymbolicInt,
    ):
        """Factory to construct sparse state given the coefficients.

        Args:
            coeffs: A vector of coefficients `c_i`, either as a sequence/numpy array,
                or a scipy sparse array.
            phase_bitsize: size of the phase-gradient register.
        """
        import scipy

        N = len(coeffs)
        sparse_coeffs = scipy.sparse.dok_array(np.atleast_2d(coeffs))
        sparse_coeffs_as_dict = {i: c for ((_, i), c) in sparse_coeffs.items()}

        return cls.from_coefficient_map(N, sparse_coeffs_as_dict, phase_bitsize)

    @classmethod
    def from_n_coeffs(
        cls, n_coeffs: SymbolicInt, n_nonzero_coeffs: SymbolicInt, phase_bitsize: SymbolicInt
    ):
        """Factory to construct a sparse state of `d` non-zero coefficients over `[0, N - 1]`.

        Args:
            n_coeffs: the total size of the state `N`
            n_nonzero_coeffs: the number of non-zero coefficients `d`.
            phase_bitsize: size of the phase-gradient register.
        """
        return cls(
            HasLength(n_nonzero_coeffs), HasLength(n_nonzero_coeffs), n_coeffs, phase_bitsize
        )

    @property
    def _extract_first_d_qubits(self) -> Partition:
        """Bloq to extract the first `d` qubits to prepare the dense state on."""
        if is_symbolic(self.target_bitsize):
            raise ValueError(f"cannot partition with symbolic {self.target_bitsize=}")

        assert not isinstance(self.target_bitsize, sympy.Expr)

        return Partition(
            self.target_bitsize,
            (
                Register('high_bits', QAny(self.target_bitsize - self.dense_bitsize)),
                Register('low_bits', QUInt(self.dense_bitsize)),
            ),
        )

    @property
    def _dense_stateprep_bloq(self) -> StatePreparationViaRotations:
        if is_symbolic(self.nonzero_coeffs):
            return StatePreparationViaRotations(
                HasLength(slen(self.nonzero_coeffs)), self.phase_bitsize
            )

        assert isinstance(self.nonzero_coeffs, tuple)

        # pad the list of nonzero coeffs to a power of 2.
        dense_coeffs_padded = np.pad(
            list(self.nonzero_coeffs), (0, 2**self.dense_bitsize - len(self.nonzero_coeffs))
        )
        return StatePreparationViaRotations(tuple(dense_coeffs_padded.tolist()), self.phase_bitsize)

    @property
    def _basis_permutation_bloq(self) -> Permutation:
        if is_symbolic(self.sparse_indices):
            # worst case: single cycle of length 2*d
            return Permutation.from_cycle_lengths(self.N, [2 * slen(self.sparse_indices)])

        assert isinstance(self.sparse_indices, tuple)

        return Permutation.from_partial_permutation_map(
            self.N, dict(enumerate(self.sparse_indices))
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', target_state: 'SoquetT', phase_gradient: 'SoquetT'
    ) -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic data")

        # prepare the dense state on log(d) bits
        high_bits, low_bits = bb.add(self._extract_first_d_qubits, x=target_state)
        low_bits, phase_gradient = bb.add(
            self._dense_stateprep_bloq, target_state=low_bits, phase_gradient=phase_gradient
        )
        target_state = bb.add(
            self._extract_first_d_qubits.adjoint(), high_bits=high_bits, low_bits=low_bits
        )

        # permute the basis to obtain the sparse state
        target_state = bb.add(self._basis_permutation_bloq, x=target_state)

        return {'target_state': target_state, 'phase_gradient': phase_gradient}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self._dense_stateprep_bloq: 1, self._basis_permutation_bloq: 1}


@bloq_example
def _sparse_state_prep_via_rotations() -> SparseStatePreparationViaRotations:
    sparse_state_prep_via_rotations = SparseStatePreparationViaRotations.from_sparse_array(
        [0.70914953, 0, 0, 0, 0.46943701, 0, 0.2297245, 0, 0, 0.32960471, 0, 0, 0.33959273, 0, 0],
        phase_bitsize=2,
    )
    return sparse_state_prep_via_rotations
