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
from typing import Optional

import cirq
import numpy as np

from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
from qualtran.bloqs.chemistry.ising import get_1d_ising_hamiltonian
from qualtran.bloqs.chemistry.ising.hamiltonian import get_1d_ising_hamiltonian_norm_upper_bound
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt


def get_prepare_precision_from_eigenphase_precision(
    eps_eigenphase: SymbolicFloat,
    num_coeffs: SymbolicInt,
    sum_of_coeffs: SymbolicFloat,
    hamiltonian_l2_norm: SymbolicFloat,
) -> SymbolicFloat:
    r"""Precision of LCU coefficients to get an `eps_eigenphase` approx. of eigenphases.

    For the overall block-encoding to have precision $\epsilon$, it requires approximating
    $w_l$ to a precision of $\delta$ such that

    $$
        \delta = \frac{\epsilon}{(1 + \epsilon^2)L}
        \left( 1 - \frac{\norm{H}^2}{\lambda^2} \right)
    $$

    as described in Eq. A9 of [1].

    Args:
        num_coeffs: number of LCU coefficients $L$.
        eps_eigenphase: precision to approximate the eigenphases $\epsilon$.
        sum_of_coeffs: sum of LCU coefficients $\lambda$.
        hamiltonian_l2_norm: upper bound on $\norm{H}$.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Eq (A9).
    """
    return ((eps_eigenphase * sum_of_coeffs) / ((1 + eps_eigenphase**2) * num_coeffs)) * (
        1 - (hamiltonian_l2_norm / sum_of_coeffs) ** 2
    )


def upper_bound_norm_for_pauli_hamiltonian(ham: cirq.PauliSum) -> float:
    r"""Compute a weak upper-bound of the norm of a PauliSum Hamiltonian.

    Given a Pauli Hamiltonian $H = \sum_{l = 0}^{L-1} w_l H_l$, where each $H_l$
    is a PauliString, compute an upper bound on $\norm{H}$ which is slightly
    better than $\lambda = \sum_l w_l$, if possible.

    If all the terms commute, this just returns $\lambda$.

    Proof:

    As all the $H_l$ are Pauli strings, we know
    - $H_l^2 = I$
    - $H_l H_{l'} = \pm H_{l'} H_l$ (i.e. they either commute/anti-commute)

    $$
        H^2
        = \sum_{l, l'} w_l w_{l'} H_l H_{l'}
        = \sum_{l} w_l^2 I + \sum_{l < l'; [H_l, H_{l'}] = 0} 2 w_l w_{l'} H_l H_{l'}
    $$

    $$
        \norm{H^2}
        \le \sum_{l} w_l^2 + \sum_{l < l'; [H_l, H_{l'}] = 0} 2 w_l w_{l'}
        = \lambda^2 - \sum_{l < l'; [H_l, H_{l'}] \ne 0} 2 w_l w_{l'}
    $$

    Notes:
        Takes time $L^2 n$, for $L$ terms and each PauliString acting on $n$ qubits.
    """
    L = len(ham)
    ham_ps = [ps for ps in ham]
    qlambda = sum(ps.coefficient for ps in ham)

    anticommute_sum = 0
    for i in range(L):
        for j in range(i):
            if not cirq.commutes(ham_ps[i], ham_ps[j]):
                anticommute_sum += np.abs(ham_ps[i].coefficient * ham_ps[j].coefficient)

    return np.sqrt(qlambda**2 - 2 * anticommute_sum)


def walk_operator_for_pauli_hamiltonian(
    ham: cirq.PauliSum, eps: SymbolicFloat, *, ham_norm_upper_bound: Optional[SymbolicFloat] = None
) -> QubitizationWalkOperator:
    r"""Get the QubitizationWalkOperator for a Hamiltonian with Pauli terms.

    For a Hamiltonian $H = \sum_{l=0}^{L-1} w_l H_l$, this returns a qubitized walk operator
    with eigenphases approximated to a precision of `eps` ($\epsilon$). I.e. this guarantees

    $$
        \norm{
            e^{i\arccos(H/\lambda)} - e^{i\arccos(\tilde{H}/\lambda)}
        } \le \epsilon
    $$

    Uses :class:`StatePreparationAliasSampling` to implement an approximate prepare for $w_l$s.

    Args:
        ham: Hamiltonian described as a sum of Pauli terms.
        eps: precision $\epsilon$ of the eigenphases of the walk operator.
        ham_norm_upper_bound: an upper bound on $\norm{H}$. If not provided,
            it is computed using :meth:`upper_bound_norm_for_pauli_hamiltonian`

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Eq (A9).
    """
    q = sorted(ham.qubits)
    ham_dps = [ps.dense(q) for ps in ham]
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]

    if ham_norm_upper_bound is None:
        ham_norm_upper_bound = upper_bound_norm_for_pauli_hamiltonian(ham)
    delta = get_prepare_precision_from_eigenphase_precision(
        eps, len(ham_coeff), sum(ham_coeff), ham_norm_upper_bound
    )
    if is_symbolic(delta):
        prepare = StatePreparationAliasSampling.from_n_coeff(
            len(ham_coeff), sum(ham_coeff), precision=delta
        )
    else:
        prepare = StatePreparationAliasSampling.from_probabilities(
            ham_coeff, precision=float(delta)
        )

    select = SelectPauliLCU(
        total_bits(prepare.selection_registers), select_unitaries=ham_dps, target_bitsize=len(q)
    )
    block_encoding = SelectBlockEncoding(select=select, prepare=prepare)

    return QubitizationWalkOperator(block_encoding=block_encoding)


def get_walk_operator_for_1d_ising_model(
    num_sites: int, eps: SymbolicFloat, *, j_zz_strength: float = 1, gamma_x_strength: float = -1
) -> tuple[QubitizationWalkOperator, cirq.PauliSum]:
    r"""Get the QubitizationWalkOperator for a 1d Ising Hamiltonian on n sites.

    Returns an $(\lambda, \cdot, \epsilon)$-block-encoding of the 1d Ising Hamiltonian.

    Args:
        num_sites: number of spins $n$.
        eps: precision $\epsilon$ of the block-encoding of the hamiltonian.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\Gamma$.

    Returns:
        The walk operator $W$ and the hamiltonian $H$.
    """
    ham = get_1d_ising_hamiltonian(cirq.LineQubit.range(num_sites), j_zz_strength, gamma_x_strength)
    ham_norm_upper_bound = get_1d_ising_hamiltonian_norm_upper_bound(
        num_sites, j_zz_strength, gamma_x_strength
    )
    walk = walk_operator_for_pauli_hamiltonian(ham, eps, ham_norm_upper_bound=ham_norm_upper_bound)
    return walk, ham
