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

from typing import Optional

import numpy as np
import pytest

from qualtran import Bloq
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sparse.prepare import _prep_sparse, get_sparse_inputs_from_integrals


def test_prep_sparse(bloq_autotester):
    bloq_autotester(_prep_sparse)


def test_prep_sparse_adj():
    bloq: Bloq = _prep_sparse.make()
    bloq.adjoint().decompose_bloq()


def reconstruct_eris(eris, indx, nb):
    """Fill back the eri tensor given the non-zero eris and the corresponding 4-index indices.

    Slight overkill, but written to mimic the swaps in the quantum algorithm
    """
    eris_recon = np.zeros((nb,) * 4)
    # let controls be a tuple of 3 cvs
    for ix, pqrs in enumerate(indx):
        p, q, r, s = pqrs
        # 1. Swap pq, rs
        # controls (0, 0, 0), no swap, base case
        eris_recon[p, q, r, s] = eris[ix]
        # ctrl = (1, 0, 0), swap
        eris_recon[r, s, p, q] = eris[ix]
        # 2.a Swap p, q, given zero swap for Step 1.
        # ctrl = (0, 1, 0)
        eris_recon[q, p, r, s] = eris[ix]
        # 2.b Swap p, q (first two registers), given swap for Step 1.
        # ctrl = (1, 1, 0)
        eris_recon[s, r, p, q] = eris[ix]
        # 3.a swap r, s, ctrl = (0, 0, 1)
        eris_recon[p, q, s, r] = eris[ix]
        # 3.b swap r, s, and p, q ctrl = (0, 1, 1)
        eris_recon[q, p, s, r] = eris[ix]
        # 3.c swap r, s, ctrl = (1, 0, 1)
        eris_recon[r, s, q, p] = eris[ix]
        # 3.d swap r, s, ctrl = (1, 1, 1)
        eris_recon[s, r, q, p] = eris[ix]
    return eris_recon


def test_decompose_bloq_counts():
    prep = _prep_sparse()
    cost_decomp = prep.decompose_bloq().call_graph()[1][TGate()]
    cost_call = prep.call_graph()[1][TGate()]
    assert cost_decomp == cost_call


def build_random_test_integrals(nb: int, seed: Optional[int] = 7):
    """Build random one- and two-electron integrals of the correct symmetry.

    Args:
        nb: The number of spatial orbitals.
        seed: If set then set the random number seed to this value. Otherwise it is not set here.

    Returns:
        tpq: The one-body matrix elements.
        eris: Chemist ERIs (pq|rs).
    """
    rs = np.random.RandomState(seed)
    tpq = rs.normal(size=(nb, nb))
    tpq = 0.5 * (tpq + tpq.T)
    eris = rs.normal(scale=4, size=(nb,) * 4)
    eris += np.transpose(eris, (0, 1, 3, 2))
    eris += np.transpose(eris, (1, 0, 2, 3))
    eris += np.transpose(eris, (2, 3, 0, 1))
    return tpq, eris


@pytest.mark.parametrize('sparsity', [0.0, 1e-2])
@pytest.mark.parametrize('nb', [4, 5, 6, 7])
def test_get_sparse_inputs_from_integrals(nb, sparsity):
    tpq, eris = build_random_test_integrals(nb, seed=7)
    pqrs_indx, eris_eight = get_sparse_inputs_from_integrals(
        tpq, eris, drop_element_thresh=sparsity
    )
    # eq A20 + one-body bit
    if sparsity < 1e-12:
        assert len(pqrs_indx) == nb * (nb + 1) * (nb**2 + nb + 2) // 8 + nb * (nb + 1) // 2
    num_lt = nb * (nb + 1) // 2
    num_lt_mat = num_lt * (num_lt + 1) // 2
    if sparsity < 1e-12:
        assert len(eris_eight) == num_lt_mat + num_lt
    tpq_recon = np.zeros_like(tpq)
    for ix, pq in enumerate(pqrs_indx[:num_lt]):
        p, q, _, _ = pq
        tpq_recon[p, q] = eris_eight[ix]
    tpq_recon += np.tril(tpq_recon, k=-1).T
    assert np.allclose(tpq_recon, tpq)
    eris[np.where(np.abs(eris) < sparsity)] = 0.0
    eris_recon = reconstruct_eris(eris_eight[num_lt:], pqrs_indx[num_lt:], nb)
    assert np.allclose(eris_recon, eris)
