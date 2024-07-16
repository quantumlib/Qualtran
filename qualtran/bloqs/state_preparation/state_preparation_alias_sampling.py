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
from typing import Iterator, Sequence, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, BoundedQUInt, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates import CSwap, Hadamard
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.data_loading.qrom import QROM
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
from qualtran.symbolics import bit_length, Shaped, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@cirq.value_equality()
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

    Args:
        selection_registers: The input/output registers to prepare the state on (see Signature).
        keep: The discretized `keep` probabilities for alias sampling.
        alt: The alternate/alias values to swap.
        mu: The number of bits to approximate the `keep` probabilities.
        sum_of_unnormalized_probabilities: The total of the input unnormalized probabilities,
            i.e., $\lambda$. This is used as the `PrepareOracle.l1_norm_of_coeffs` property.

    Signature:
        selection: The input/output register $|\ell\rangle$ of size lg(L) where the desired
            coefficient state is prepared.
        temp: Work space comprised of sub signature:
            - sigma: A mu-sized register containing uniform probabilities for comparison against
                `keep`.
            - alt: A lg(L)-sized register of alternate indices
            - keep: a mu-sized register of probabilities of keeping the initially sampled index.
            - one bit for the result of the comparison.

    This gate corresponds to the following operations:
     - UNIFORM_L on the selection register
     - H^mu on the sigma register
     - QROM addressed by the selection register into the alt and keep signature.
     - LessThanEqualGate comparing the keep and sigma signature.
     - Coherent swap between the selection register and alt register if the comparison
       returns True.

    Total space will be (2 * log(L) + 2 mu + 1) work qubits + log(L) ancillas for QROM.
    The 1 ancilla in work qubits is for the `LessThanEqualGate` followed by coherent swap.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 11.
    """

    selection_registers: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    alt: Union[Shaped, NDArray[np.int_]]
    keep: Union[Shaped, NDArray[np.int_]]
    mu: SymbolicInt
    sum_of_unnormalized_probabilities: SymbolicFloat

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
            selection_registers=Register('selection', BoundedQUInt((N - 1).bit_length(), N)),
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
            selection_registers=Register('selection', BoundedQUInt(selection_bitsize, n_coeff)),
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

    def _value_equality_values_(self):
        return (
            self.selection_registers,
            self.alt if isinstance(self.alt, Shaped) else tuple(self.alt.ravel()),
            self.keep if isinstance(self.keep, Shaped) else tuple(self.keep.ravel()),
            self.mu,
        )

    @cached_property
    def qrom_bloq(self) -> QROM:
        return QROM(
            (self.alt, self.keep),
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        yield PrepareUniformSuperposition(self.n_coeff).on(*quregs['selection'])
        if self.mu == 0:
            return
        selection, less_than_equal = quregs['selection'], quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        yield cirq.H.on_each(*sigma_mu)
        yield self.qrom_bloq.on_registers(selection=selection, target0_=alt, target1_=keep)
        yield LessThanEqual(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
        yield CSwap.make_on(ctrl=less_than_equal, x=alt, y=selection)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (PrepareUniformSuperposition(self.n_coeff), 1),
            (self.qrom_bloq, 1),
            (LessThanEqual(self.mu, self.mu), 1),
            (CSwap(self.selection_bitsize), 1),
            (Hadamard(), self.mu),
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
    bloq_cls=StatePreparationAliasSampling,
    import_line='from qualtran.bloqs.state_preparation import StatePreparationAliasSampling',
    examples=(_state_prep_alias, _state_prep_alias_symb),
)
