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

from typing import List

from numpy.typing import NDArray

from qualtran import Bloq, Signature, Soquet
from qualtran._infra.composite_bloq import _get_flat_dangling_soqs

from ._quimb import cbloq_to_quimb


def get_right_and_left_inds(signature: Signature) -> List[List[Soquet]]:
    """Return right and left tensor indices.

    In general, this will be returned as a list of length-2 corresponding
    to the right and left indices, respectively. If there *are* no right
    or left indices, that entry will be omitted from the returned list.

    Right indices come first to match the quantum computing / matrix multiplication
    convention where U_tot = U_n ... U_2 U_1.
    """
    inds = []
    rsoqs = _get_flat_dangling_soqs(signature, right=True)
    if rsoqs:
        inds.append(rsoqs)
    lsoqs = _get_flat_dangling_soqs(signature, right=False)
    if lsoqs:
        inds.append(lsoqs)
    return inds


def bloq_to_dense(bloq: Bloq) -> NDArray:
    """Return a contracted, dense ndarray representing the composite bloq.

    The public version of this function is available as the `Bloq.tensor_contract()`
    method.

    This constructs a tensor network and then contracts it according to the cbloq's registers,
    i.e. the dangling indices. The returned array will be 0-, 1- or 2- dimensional. If it is
    a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
    of (right, left) indices.

    For more fine grained control over the final shape of the tensor, use
    `cbloq_to_quimb` and `TensorNetwork.to_dense` directly.
    """
    cbloq = bloq.as_composite_bloq()
    tn, _ = cbloq_to_quimb(cbloq)
    inds = get_right_and_left_inds(cbloq.signature)

    if inds:
        return tn.to_dense(*inds)

    return tn.contract()
