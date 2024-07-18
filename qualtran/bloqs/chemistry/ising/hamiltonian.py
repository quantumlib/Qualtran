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
"""Functions for specifying the Ising model and building LCU coefficients."""
from typing import List, Sequence

import cirq
import numpy as np

from qualtran.symbolics import ssqrt, SymbolicFloat, SymbolicInt


def get_1d_ising_pauli_terms(
    qubits: Sequence[cirq.Qid], j_zz_strength: float = 1.0, gamma_x_strength: float = -1
):
    r"""Get the pauli terms for a 1d ising model with periodic boundaries.

    $$
    H = -J\sum_{k=0}^{L-1}\sigma_{k}^{Z}\sigma_{(k+1)\%L}^{Z} - \Gamma\sum_{k=0}^{L-1}\sigma_{k}^{X}
    $$

    Args:
        qubits: One qubit for each spin site.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\Gamma$.

    Returns:
        zz_terms: The list of PauliStrings for the ZZ terms.
        x_terms: The list of PauliStrings for the X terms.
    """
    n_sites = len(qubits)
    zz_terms: List[cirq.PauliString] = []
    x_terms: List[cirq.PauliString] = []
    for k in range(n_sites):
        zz_terms.append(
            cirq.PauliString(
                {qubits[k]: cirq.Z, qubits[(k + 1) % n_sites]: cirq.Z}, coefficient=j_zz_strength
            )
        )
        x_terms.append(cirq.PauliString({qubits[k]: cirq.X}, coefficient=gamma_x_strength))

    return zz_terms, x_terms


def get_1d_ising_hamiltonian(
    qubits: Sequence[cirq.Qid], j_zz_strength: float = 1.0, gamma_x_strength: float = -1
) -> cirq.PauliSum:
    r"""A one dimensional ising model with periodic boundaries.

    $$
    H = -J\sum_{k=0}^{L-1}\sigma_{k}^{Z}\sigma_{(k+1)\%L}^{Z} - \Gamma\sum_{k=0}^{L-1}\sigma_{k}^{X}
    $$

    Args:
        qubits: One qubit for each spin site.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\Gamma$.

    Returns:
        cirq.PauliSum representing the Hamiltonian
    """
    zz_terms, x_terms = get_1d_ising_pauli_terms(qubits, j_zz_strength, gamma_x_strength)
    return cirq.PauliSum.from_pauli_strings(zz_terms + x_terms)


def get_1d_ising_lcu_coeffs(
    n_spins: int, j_zz_strength: float = np.pi / 3, gamma_x_strength: float = np.pi / 7
) -> np.ndarray:
    r"""Get LCU coefficients for a 1d ising Hamiltonian.

    The order of the terms is according to `get_1d_ising_hamiltonian`, namely: ZZ's and X's
    interleaved.

    Args:
        n_spins: The number of lattice sites / spins.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\Gamma$.

    Returns:
        The LCU coefficients.
    """
    spins = cirq.LineQubit.range(n_spins)
    ham = get_1d_ising_hamiltonian(spins, j_zz_strength, gamma_x_strength)
    coeffs = np.array([term.coefficient.real for term in ham])
    return coeffs


def get_1d_ising_hamiltonian_norm_upper_bound(
    n_sites: SymbolicInt, j_zz_strength: SymbolicFloat, gamma_x_strength: SymbolicFloat
) -> SymbolicFloat:
    r"""Weak upperbound on the norm of the 1d Ising Hamiltonian.

    Recall that the 1d Ising Hamiltonian on $L$ sites is

        $$
        H = -J\sum_{k=0}^{L-1}\Z_{k}\Z_{(k+1)\%L} - \Gamma\sum_{k=0}^{L-1}X_{k}
        $$

    We can therefore upperbound the norm by grouping $Z_k Z_{k+1}$ and $X_k$:

    $$
        \norm{H} \le \sum_{k=0}^{L-1} \norm{J Z_k Z_{(k + 1)\%L} + \Gamma X_k}
    $$

    As each term is equal, this is effectively $L \norm{J Z_k Z_{(k + 1)\%L} + \Gamma X_k}$.
    And the square of this term is

    $$
        (J Z_k Z_{(k + 1)\%L} + \Gamma X_k)^2 = (J^2 + \Gamma^2) I
    $$

    and therefore we obtain $\norm{H} \le L \sqrt{J^2 + \Gamma^2}$.

    See :meth:`get_1d_ising_hamiltonian` to get the exact hamiltonian.
    """
    return n_sites * ssqrt(j_zz_strength**2 + gamma_x_strength**2)
