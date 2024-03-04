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
from typing import Dict, List, Sequence

import attrs
import cirq
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Rx, Rz


def get_1d_ising_pauli_terms(
    qubits: Sequence[cirq.Qid], j_zz_strength: float = 1.0, gamma_x_strength: float = -1
):
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


def get_1d_Ising_hamiltonian(
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


def get_1d_Ising_lcu_coeffs(
    n_spins: int, j_zz_strength: float = np.pi / 3, gamma_x_strength: float = np.pi / 7
) -> np.ndarray:
    """Get LCU coefficients for a 1d ising Hamiltonian.

    The order of the terms is according to `get_1d_Ising_hamiltonian`, namely: ZZ's and X's
    interleaved.
    """
    spins = cirq.LineQubit.range(n_spins)
    ham = get_1d_Ising_hamiltonian(spins, j_zz_strength, gamma_x_strength)
    coeffs = np.array([term.coefficient.real for term in ham])
    lcu_coeffs = coeffs / np.sum(abs(coeffs))
    return lcu_coeffs


@attrs.frozen
class IsingXUnitary(Bloq):
    nsites: int
    angle: float
    eps: float = 1e-10

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(system=self.nsites)

    def short_name(self) -> str:
        return 'U_X'

    def build_composite_bloq(self, bb: 'BloqBuilder', system: 'SoquetT') -> Dict[str, 'Soquet']:
        system = bb.split(system)
        for iq in range(self.nsites):
            system[iq] = bb.add(Rx(self.angle), q=system[iq])
        return {'system': bb.join(system)}


@attrs.frozen
class IsingZZUnitary(Bloq):
    nsites: int
    angle: float
    eps: float = 1e-10

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(system=self.nsites)

    def short_name(self) -> str:
        return 'U_ZZ'

    def build_composite_bloq(self, bb: 'BloqBuilder', system: 'SoquetT') -> Dict[str, 'Soquet']:
        system = bb.split(system)
        for iq_a in range(self.nsites):
            iq_b = (iq_a + 1) % self.nsites
            system[iq_a], system[iq_b] = bb.add(CNOT(), ctrl=system[iq_a], target=system[iq_b])
            system[iq_b] = bb.add(Rz(self.angle, self.eps), q=system[iq_b])
            system[iq_a], system[iq_b] = bb.add(CNOT(), ctrl=system[iq_a], target=system[iq_b])
        return {'system': bb.join(system)}
