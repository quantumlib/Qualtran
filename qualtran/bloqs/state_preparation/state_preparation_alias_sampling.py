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

"""Gates for preparing coefficient states.

In section III.D. of the [Linear T paper](https://arxiv.org/abs/1805.03662) the authors introduce
a technique for initializing a state with $L$ unique coefficients (provided by a classical
database) with a number of T gates scaling as 4L + O(log(1/eps)) where eps is the
largest absolute error that one can tolerate in the prepared amplitudes.
"""
from functools import cached_property
from typing import Sequence, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, BQUInt, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates import CSwap, Hadamard, OnEach
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.linalg.lcu_util import (
    preprocess_probabilities_for_reversible_sampling,
    sub_bit_prec_from_epsilon,
)
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.symbolics import bit_length, is_symbolic, Shaped, slen, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _data_or_shape_to_tuple(data_or_shape: Union[NDArray, Shaped]) -> Tuple:
    return (
        tuple(data_or_shape.flatten())
        if isinstance(data_or_shape, np.ndarray)
        else (data_or_shape,)
    )


@attrs.frozen
class StatePreparationAliasSampling(PrepareOracle):
    r"""Initialize a state with $L$ coefficients using coherent alias sampling.

    In particular, given coefficients $w_\ell$, we take the zero state to:

    $$
    \sum_{\ell=0}^{L-1} \sqrt{p_\ell} |\ell\rangle |\mathrm{temp}_\ell\rangle
    $$

    where the probabilities $p_\ell$ are $\mu$-bit binary approximations to the true values
    $w_\ell / \lambda$ (where $\lambda = \sum_\ell w_\ell$).
    Note that the temporary register must be treated with care, see the details in Section III.D.
    of the reference.

    This construction is designed to work specifically when you don't require specific phases,
    and the problem is reduced to [classical alias sampling]
    (https://en.wikipedia.org/wiki/Alias_method). We sample `l` with probability `p[l]` by first
    selecting `l` uniformly at random and then returning it with probability `keep[l] / 2**mu`;
    otherwise returning `alt[l]`.

    This gate corresponds to the following operations:
     - UNIFORM_L on the selection register
     - H^mu on the sigma register
     - QROM addressed by the selection register into the alt and keep signature.
     - LessThanEqualGate comparing the keep and sigma signature.
     - Coherent swap between the selection register and alt register if the comparison
       returns True.

    Total space will be (2 * log(L) + 2 mu + 1) work qubits + log(L) ancillas for QROM.
    The 1 ancilla in work qubits is for the `LessThanEqualGate` followed by coherent swap.

    Registers:
        selection: The input/output register $|\mathrm{ind}_l\rangle$ of size lg(L) where the desired
            coefficient state is prepared. Default name is 'selection' if the builder methods on
            the class are used. Or else, users can specify custom named registers
        sigma_mu: A mu-sized register containing uniform probabilities for comparison against `keep`.
        alt: A lg(L)-sized register of alternate indices
        keep: a mu-sized register of probabilities of keeping the initially sampled index.
        less_than_equal: one bit for the result of the comparison.

    Args:
        selection_registers: The input/output registers to prepare the state on (see Registers section).
        keep: The discretized `keep` probabilities for alias sampling.
        alt: The alternate/alias values to swap.
        mu: The number of bits to approximate the `keep` probabilities.
        sum_of_unnormalized_probabilities: The total of the input unnormalized probabilities,
            i.e., $\lambda$. This is used as the `PrepareOracle.l1_norm_of_coeffs` property.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 11.
    """

    selection_registers: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    alt: Union[Shaped, NDArray[np.int_]] = attrs.field(eq=_data_or_shape_to_tuple)
    keep: Union[Shaped, NDArray[np.int_]] = attrs.field(eq=_data_or_shape_to_tuple)
    mu: SymbolicInt
    sum_of_unnormalized_probabilities: SymbolicFloat

    def __attrs_post_init__(self):
        if not is_symbolic(self.mu) and self.mu <= 0:
            raise ValueError(f"{self.mu=} must be greater than 0.")
        if len(self.selection_registers) != 1:
            raise ValueError(
                f"{type(self)} only supports 1D state preparation. Found multiple {self.selection_registers=}."
            )

    @classmethod
    def from_probabilities(
        cls, unnormalized_probabilities: Sequence[float], *, precision: float = 1.0e-5
    ) -> 'StatePreparationAliasSampling':
        r"""Factory to construct the state preparation gate for a given set of unnormalized probabilities.

        Given input `unnormalized_probabilities` $w_l$, with sum $\lambda = \sum_l w_l$, this prepares
        a state s.t. $p_l = \tilde{w_l} / \lambda$ such that the input `precision` $\epsilon$ satisfies:

            $$
                |w_l - \tilde{w}_l| \le \epsilon
            $$

        That is, the value `precision` is the absolute error in approximating the input values.

        Args:
            unnormalized_probabilities: The LCU coefficients $w_l$.
            precision: The desired accuracy $\epsilon$ to represent each input value
                (which sets mu size and keep/alt integers).
                See `qualtran.linalg.lcu_util.preprocess_probabilities_for_reversible_sampling`
                for more information.
        """
        if not all(x >= 0 for x in unnormalized_probabilities):
            raise ValueError(f"{cls} expects only non-negative probabilities")

        qlambda = sum(x for x in unnormalized_probabilities)
        alt, keep, mu = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=unnormalized_probabilities, epsilon=precision / qlambda
        )
        N = len(unnormalized_probabilities)
        return StatePreparationAliasSampling(
            selection_registers=Register('selection', BQUInt((N - 1).bit_length(), N)),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            sum_of_unnormalized_probabilities=qlambda,
        )

    @classmethod
    def from_n_coeff(
        cls,
        n_coeff: SymbolicInt,
        sum_of_unnormalized_probabilites: SymbolicFloat,
        *,
        precision: SymbolicFloat = 1.0e-5,
    ) -> 'StatePreparationAliasSampling':
        r"""Factory to construct the state preparation gate for symbolic number of unnormalized probabilities.

        See docstring for :meth:`StatePreparationAliasSampling.from_probabilities` for details

        Args:
            n_coeff: Symbolic number of LCU coefficients in the prepared state.
            sum_of_unnormalized_probabilites: Sum of absolute values of input unnormalized probabilities.
            precision: The desired accuracy $\epsilon$ to represent each input value
                (which sets mu size and keep/alt integers).
                See `qualtran.linalg.lcu_util.preprocess_probabilities_for_reversible_sampling`
                for more information.
        """
        mu = sub_bit_prec_from_epsilon(n_coeff, precision / sum_of_unnormalized_probabilites)
        selection_bitsize = bit_length(n_coeff - 1)
        alt, keep = Shaped((n_coeff,)), Shaped((n_coeff,))
        return StatePreparationAliasSampling(
            selection_registers=Register('selection', BQUInt(selection_bitsize, n_coeff)),
            alt=alt,
            keep=keep,
            mu=mu,
            sum_of_unnormalized_probabilities=sum_of_unnormalized_probabilites,
        )

    @property
    def n_coeff(self) -> SymbolicInt:
        return self.selection_registers[0].dtype.iteration_length_or_zero()

    @cached_property
    def l1_norm_of_coeffs(self) -> 'SymbolicFloat':
        return self.sum_of_unnormalized_probabilities

    @cached_property
    def sigma_mu_bitsize(self) -> SymbolicInt:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> SymbolicInt:
        return total_bits(self.selection_registers)

    @cached_property
    def keep_bitsize(self) -> SymbolicInt:
        return self.mu

    @cached_property
    def selection_bitsize(self) -> SymbolicInt:
        return total_bits(self.selection_registers)

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Signature.build(
                sigma_mu=self.sigma_mu_bitsize,
                alt=self.alternates_bitsize,
                keep=self.keep_bitsize,
                less_than_equal=1,
            )
        )

    @cached_property
    def qrom_bloq(self) -> QROM:
        return QROM(
            (self.alt, self.keep),
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        sigma_mu: 'SoquetT',
        alt: 'SoquetT',
        keep: 'SoquetT',
        less_than_equal: 'Soquet',
        **soqs: 'SoquetT',
    ):
        selection = soqs.pop(self.selection_registers[0].name)
        assert not soqs
        selection = bb.add(PrepareUniformSuperposition(self.n_coeff), target=selection)
        sigma_mu = bb.add(OnEach(self.mu, Hadamard()), q=sigma_mu)
        selection, alt, keep = bb.add(
            self.qrom_bloq, selection=selection, target0_=alt, target1_=keep
        )
        keep, sigma_mu, less_than_equal = bb.add(
            LessThanEqual(self.mu, self.mu), x=keep, y=sigma_mu, target=less_than_equal
        )
        less_than_equal, alt, selection = bb.add(
            CSwap(self.selection_bitsize), ctrl=less_than_equal, x=alt, y=selection
        )
        return {
            self.selection_registers[0].name: selection,
            'less_than_equal': less_than_equal,
            'sigma_mu': sigma_mu,
            'alt': alt,
            'keep': keep,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            PrepareUniformSuperposition(self.n_coeff): 1,
            self.qrom_bloq: 1,
            LessThanEqual(self.mu, self.mu): 1,
            CSwap(self.selection_bitsize): 1,
            Hadamard(): self.mu,
        }


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _state_prep_alias() -> StatePreparationAliasSampling:
    coeffs = [1.0, 1, 3, 2]
    mu = 3
    state_prep_alias = StatePreparationAliasSampling.from_probabilities(
        coeffs, precision=2**-mu / len(coeffs) * sum(coeffs)
    )
    return state_prep_alias


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _state_prep_alias_symb() -> StatePreparationAliasSampling:
    import sympy

    n_coeffs, sum_coeff, eps = sympy.symbols(r"L \lambda \epsilon")
    state_prep_alias_symb = StatePreparationAliasSampling.from_n_coeff(
        n_coeffs, sum_coeff, precision=eps
    )
    return state_prep_alias_symb


_STATE_PREP_ALIAS_DOC = BloqDocSpec(
    bloq_cls=StatePreparationAliasSampling, examples=(_state_prep_alias, _state_prep_alias_symb)
)


@attrs.frozen
class SparseStatePreparationAliasSampling(PrepareOracle):
    r"""Initialize a $d$-sparse state over $L$ indices using coherent alias sampling.

    In particular, we take the zero state to:

    $$
        \sum_{j=0}^{d-1} \sqrt{p_{\mathrm{ind}_j}} |\mathrm{ind}_j\rangle |\mathrm{temp}_j\rangle
    $$

    where $\mathrm{ind}_j \in [0, L)$ is the index of the $j$-th non-zero coefficient,
    and the probabilities $p_l$ are $\mu$-bit binary approximations to the true values,
    and the register $|\mathrm{temp}_j\rangle$ may be entangled with the index register.

    This bloq is nearly identical to :class:`StatePreparationByAliasSampling`, except
    that it loads the non-zero indices from the QROM and prepares a dense state on them.
    In comparison, this uses $\lceil \log d \rceil$ extra ancilla qubits, and reduces
    the iteration length to $d$ from $L$.

    See :class:`StatePreparationAliasSampling` for an exposition on alias sampling.


    Registers:
        selection: The input/output register $|\mathrm{ind}_l\rangle$ of size lg(L) where the desired
            coefficient state is prepared.
        sigma_mu: A mu-sized register containing uniform probabilities for comparison against `keep`.
        sparse_index: A lg(d)-sized register storing the sparse index $j \in [0, d)$.
        alt: A lg(L)-sized register of alternate indices
        keep: a mu-sized register of probabilities of keeping the initially sampled index.
        less_than_equal: one bit for the result of the comparison.

    This gate corresponds to the following operations:
     - UNIFORM_d on the `sparse_index` register.
     - H^mu on the `sigma` register.
     - QROM addressed by the `sparse_index` register into the `selection`, `alt`, and `keep` signature.
     - LessThanEqualGate comparing the `keep` and `sigma` registers.
     - Coherent swap between the `selection` and `alt` registers if the comparison returns True.

    Total space will be $(2 \log(L) + \log(d) + 2 \mu + 1)$ work qubits + $log(L)$ ancillas for QROM.

    References:
        [1] [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization](https://arxiv.org/pdf/1902.02134#page=15.30)
        Berry et al. (2019). Section 5, Eqs. 43, 44.
        [2] [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et al. (2018). Section III.D. and Figure 11.
    """
    selection_registers: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    index: Union[Shaped, NDArray[np.int_]] = attrs.field(eq=_data_or_shape_to_tuple)
    alt: Union[Shaped, NDArray[np.int_]] = attrs.field(eq=_data_or_shape_to_tuple)
    keep: Union[Shaped, NDArray[np.int_]] = attrs.field(eq=_data_or_shape_to_tuple)
    mu: SymbolicInt
    sum_of_unnormalized_probabilities: SymbolicFloat

    def __attrs_post_init__(self):
        if not is_symbolic(self.mu) and self.mu <= 0:
            raise ValueError(f"{self.mu=} must be at least 1")

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Signature.build(
                sigma_mu=self.mu,
                sparse_index=self.sparse_index_bitsize,
                alt=self.selection_bitsize,
                keep=self.mu,
                less_than_equal=1,
            )
        )

    @classmethod
    def from_sparse_dict(
        cls, unnormalized_probabilities: dict[int, float], N: int, *, precision: float = 1.0e-5
    ) -> 'SparseStatePreparationAliasSampling':
        """Construct the state preparation gate for a given dictionary of non-zero probabilities.

        Args:
            unnormalized_probabilities: a dictionary mapping indices to non-zero probabilities.
            N: the maximum index (i.e. prepares a state with basis in [0, N))
            precision: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `qualtran.linalg.lcu_util.preprocess_probabilities_for_reversible_sampling`
                for more information.
        """
        if not all(x >= 0 for x in unnormalized_probabilities.values()):
            raise ValueError(f"{cls} expects only non-negative probabilities")
        if not all(0 <= ix < N for ix in unnormalized_probabilities.keys()):
            raise ValueError(
                f"Sparse indices not in range [0, {N}): {unnormalized_probabilities.keys()}"
            )

        alt_compressed, keep, mu = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=list(unnormalized_probabilities.values()), epsilon=precision
        )

        index = list(unnormalized_probabilities.keys())
        alt = [index[idx] for idx in alt_compressed]

        return cls(
            selection_registers=Register('selection', BQUInt((N - 1).bit_length(), N)),
            index=np.array(index),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            sum_of_unnormalized_probabilities=sum(unnormalized_probabilities.values()),
        )

    @classmethod
    def from_dense_probabilities(
        cls,
        unnormalized_probabilities: Sequence[float],
        *,
        precision: float = 1.0e-5,
        nonzero_threshold: float = 1e-6,
    ) -> 'SparseStatePreparationAliasSampling':
        """Factory to construct the state preparation gate for a given set of probability coefficients.

        Args:
            unnormalized_probabilities: A dense list of all probabilities (i.e. including 0s)
            precision: The desired accuracy to represent each probability
            nonzero_threshold: minimum value for a probability entry to be considered non-zero.
        """
        nonzero_value_map: dict[int, float] = {
            ix: prob
            for ix, prob in enumerate(unnormalized_probabilities)
            if not np.isclose(prob, 0, atol=nonzero_threshold)
        }

        return cls.from_sparse_dict(
            nonzero_value_map, len(unnormalized_probabilities), precision=precision
        )

    @classmethod
    def from_n_coeff(
        cls,
        n_coeff: SymbolicInt,
        n_nonzero_coeff: SymbolicInt,
        sum_of_terms: SymbolicFloat,
        *,
        precision: SymbolicFloat = 1.0e-5,
    ) -> 'SparseStatePreparationAliasSampling':
        """Factory to construct sparse state preparation for symbolic number of input probabilities.

        Args:
            n_coeff: Symbolic number of LCU coefficients in the prepared state.
            n_nonzero_coeff: Symbolic number of non-zero LCU coefficients in the prepared state.
            sum_of_terms: Sum of absolute values of the input probabilities.
            precision: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `qualtran.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        """
        mu = sub_bit_prec_from_epsilon(n_coeff, precision)
        selection_bitsize = bit_length(n_coeff - 1)
        return cls(
            selection_registers=Register('selection', BQUInt(selection_bitsize, n_coeff)),
            index=Shaped((n_nonzero_coeff,)),
            alt=Shaped((n_nonzero_coeff,)),
            keep=Shaped((n_nonzero_coeff,)),
            mu=mu,
            sum_of_unnormalized_probabilities=sum_of_terms,
        )

    @property
    def n_coeff(self) -> SymbolicInt:
        return self.selection_registers[0].dtype.iteration_length_or_zero()

    @property
    def n_nonzero_coeff(self) -> SymbolicInt:
        return slen(self.index)

    @cached_property
    def l1_norm_of_coeffs(self) -> 'SymbolicFloat':
        return self.sum_of_unnormalized_probabilities

    @cached_property
    def selection_bitsize(self) -> SymbolicInt:
        return total_bits(self.selection_registers)

    @cached_property
    def sparse_index_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_nonzero_coeff - 1)

    @cached_property
    def qrom_bloq(self) -> QROM:
        return QROM(
            (self.index, self.alt, self.keep),
            (self.sparse_index_bitsize,),
            (self.selection_bitsize, self.selection_bitsize, self.mu),
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        soqs['sparse_index'] = bb.add(
            PrepareUniformSuperposition(self.n_nonzero_coeff), target=soqs['sparse_index']
        )
        soqs['sigma_mu'] = bb.add(OnEach(self.mu, Hadamard()), q=soqs['sigma_mu'])
        soqs['sparse_index'], soqs['selection'], soqs['alt'], soqs['keep'] = bb.add_t(
            self.qrom_bloq,
            selection=soqs['sparse_index'],
            target0_=soqs['selection'],
            target1_=soqs['alt'],
            target2_=soqs['keep'],
        )
        soqs['keep'], soqs['sigma_mu'], soqs['less_than_equal'] = bb.add_t(
            LessThanEqual(self.mu, self.mu),
            x=soqs['keep'],
            y=soqs['sigma_mu'],
            target=soqs['less_than_equal'],
        )
        soqs['less_than_equal'], soqs['alt'], soqs['selection'] = bb.add_t(
            CSwap(self.selection_bitsize),
            ctrl=soqs['less_than_equal'],
            x=soqs['alt'],
            y=soqs['selection'],
        )

        return soqs


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _sparse_state_prep_alias() -> SparseStatePreparationAliasSampling:
    coeff_map = {0: 1.0, 3: 1.0, 5: 3.0, 7: 2.0}
    N = 9
    mu = 3
    sparse_state_prep_alias = SparseStatePreparationAliasSampling.from_sparse_dict(
        coeff_map, N, precision=2**-mu / len(coeff_map)
    )
    return sparse_state_prep_alias


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _sparse_state_prep_alias_from_list() -> SparseStatePreparationAliasSampling:
    coeffs = [1.0, 0, 0, 1, 0, 3, 0, 2, 0]
    mu = 3
    sparse_state_prep_alias_from_list = (
        SparseStatePreparationAliasSampling.from_dense_probabilities(coeffs, precision=2**-mu / 4)
    )
    return sparse_state_prep_alias_from_list


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _sparse_state_prep_alias_symb() -> SparseStatePreparationAliasSampling:
    import sympy

    n_coeffs, n_nonzero_coeffs, sum_coeff, eps = sympy.symbols(r"L d \lambda \epsilon")
    sparse_state_prep_alias_symb = SparseStatePreparationAliasSampling.from_n_coeff(
        n_coeffs, n_nonzero_coeffs, sum_coeff, precision=eps
    )
    return sparse_state_prep_alias_symb


_SPARSE_STATE_PREP_ALIAS_DOC = BloqDocSpec(
    bloq_cls=SparseStatePreparationAliasSampling,
    examples=(
        _sparse_state_prep_alias,
        _sparse_state_prep_alias_from_list,
        _sparse_state_prep_alias_symb,
    ),
)
