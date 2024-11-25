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
from numpy.typing import NDArray

from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator


def get_walk_operator_for_sparse_chem_ham(
    tpq: NDArray, eris: NDArray, num_bits_state_prep: int, num_bits_rot_aa: int
) -> QubitizationWalkOperator:
    """Build a walk operator for the sparse second quantized chemistry hamiltonian.

    Args:
        tpq: The modified one-electron integrals in the spatial orbital basis.
        eris: The two electron integrals in Chemist's notation in the spatial orbital basis.
        num_bits_state_prep: The number of bits of precision for state
            preparation via alias sampling.
        num_bits_rot_aa: The number of bits of precision for the single-qubit
            rotation for state preparation.

    Returns:
        walk_op: The qubitization walk operator.
    """
    assert len(tpq.shape) == 2, "tpq must be a two-dimensional array."
    assert len(eris.shape) == 4, "eris must be a four-index tensor."
    num_spin_orb = 2 * tpq.shape[-1]
    prepare = PrepareSparse.from_hamiltonian_coeffs(
        num_spin_orb,
        tpq,
        eris,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
    )
    select = SelectSparse(num_spin_orb)
    block_encoding = SelectBlockEncoding(select=select, prepare=prepare)
    walk_op = QubitizationWalkOperator(block_encoding=block_encoding)
    return walk_op
