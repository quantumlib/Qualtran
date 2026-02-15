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
"""Function for building a walk operator for the THC hamiltonian."""
import attrs
import numpy as np
from numpy.typing import NDArray
from openfermion.resource_estimates.utils import QI

from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
from qualtran.bloqs.chemistry.thc import PrepareTHC, SelectTHC
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator


def get_walk_operator_for_thc_ham(
    t_l: NDArray,
    eta: NDArray,
    zeta: NDArray,
    num_bits_state_prep: int,
    num_bits_theta: int,
    kr1: int = 1,
    kr2: int = 1,
) -> QubitizationWalkOperator:
    r"""Build a QubitizationWalkOperator for the THC hamiltonian.

    Args:
        t_l: Eigenvalues of modified one-body hamiltonian.
        eta: THC leaf tensors.
        zeta: THC central tensor.
        num_bits_state_prep: The number of bits for the state prepared during alias sampling.
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)

    Returns:
        walk_op: Walk operator for THC hamiltonian.
    """
    prep = PrepareTHC.from_hamiltonian_coeffs(t_l, eta, zeta, num_bits_state_prep)
    num_mu = zeta.shape[-1]
    num_spin_orb = 2 * len(t_l)
    sel = SelectTHC(num_mu, num_spin_orb, num_bits_theta, prep.keep_bitsize, kr1=kr1, kr2=kr2)
    block_encoding = SelectBlockEncoding(select=sel, prepare=prep)
    walk_op = QubitizationWalkOperator(block_encoding=block_encoding)
    return walk_op


def get_reiher_thc_walk_operator(
    num_bits_theta: int = 16, num_bits_state_prep: int = 10
) -> QubitizationWalkOperator:
    """Build the THC walk operator for the Reiher hamiltoninan

    Note currently we spoof the Hamiltonian by over writing prepare's 1-norm
    value with the correct value and use random THC factors for expediency.

    Parameters are taken from openfermion compute_cost_thc_test.py.

    Args:
        num_bits_theta: the number of bits of precision for the givens rotations
        num_bits_state_prep: The number of bits of precision for the preparation
            of the LCU coefficients using alias sampling.

    Returns:
        walk_op: A constructed Reiher Hamiltonian walk operator.
    """
    # Let's just generate some random coefficients for the moment with parameters
    # corresponding to the FeMoCo model complex.
    num_spin_orb = 108
    num_mu = 350
    num_bits_theta = 16
    num_bits_state_prep = 10
    tpq = np.random.normal(0, 1, size=(num_spin_orb // 2, num_spin_orb // 2))
    zeta = np.random.normal(0, 1, size=(num_mu, num_mu))
    zeta = 0.5 * (zeta + zeta.T)
    eta = np.random.normal(0, 1, size=(num_mu, num_spin_orb // 2))
    eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", eta, eta, eta, eta, zeta, optimize=True)
    # In practice one typically uses the exact ERI tensor instead of that from
    # THC, but that's a minor detail.
    tpq_prime = (
        tpq
        - 0.5 * np.einsum("illj->ij", eri_thc, optimize=True)
        + np.einsum("llij->ij", eri_thc, optimize=True)
    )
    t_l = np.linalg.eigvalsh(tpq_prime)
    qroam_blocking_factor = QI(num_mu + num_spin_orb // 2)[0]
    walk_op = get_walk_operator_for_thc_ham(
        t_l,
        eta,
        zeta,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_theta=num_bits_theta,
        kr1=qroam_blocking_factor,
        kr2=qroam_blocking_factor,
    )
    # GIANT HACK: overwrite the lambda value directly
    # TODO: maybe parse THC hamiltonian files from openfermion directly
    block_encoding = attrs.evolve(
        walk_op.block_encoding, prepare=attrs.evolve(walk_op.prepare, sum_of_l1_coeffs=306.3)
    )
    walk_op = attrs.evolve(walk_op, block_encoding=block_encoding)
    return walk_op
