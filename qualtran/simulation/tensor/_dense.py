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

import logging
from typing import Dict, List, Tuple, TYPE_CHECKING

from numpy.typing import NDArray

from qualtran import Bloq, Connection, ConnectionT, LeftDangle, RightDangle, Signature, Soquet

from ._flattening import flatten_for_tensor_contraction
from ._quimb import cbloq_to_quimb

if TYPE_CHECKING:
    import quimb.tensor as qtn

logger = logging.getLogger(__name__)


def _order_incoming_outgoing_indices(
    signature: Signature, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
) -> List[Tuple[Connection, int]]:
    """Order incoming and outgoing indices provided by the tensor protocol according to `signature`.

    This can be used if you have a well-ordered, dense numpy array.

    >>> inds = _order_incoming_outgoing_indices(signature, incoming, outgoing)
    >>> data = unitary.reshape((2,) * len(inds))
    >>> return [qtn.Tensor(data=data, inds=inds)]
    """

    inds: List[Tuple[Connection, int]] = []

    # Nested for loops:
    #   reg: each register in the signature
    #   idx: each index into a shaped register, or () for a non-shaped register
    #   j:   each qubit (sub-)index for a given data type
    for reg in signature.rights():
        for idx in reg.all_idxs():
            for j in range(reg.dtype.num_qubits):
                if idx:
                    inds.append((outgoing[reg.name][idx], j))  # type: ignore[index]
                else:
                    inds.append((outgoing[reg.name], j))  # type: ignore[arg-type]
    for reg in signature.lefts():
        for idx in reg.all_idxs():
            for j in range(reg.dtype.num_qubits):
                if idx:
                    inds.append((incoming[reg.name][idx], j))  # type: ignore[index]
                else:
                    inds.append((incoming[reg.name], j))  # type: ignore[arg-type]

    return inds


def get_right_and_left_inds(tn: 'qtn.TensorNetwork', signature: Signature) -> List[List[Soquet]]:
    """Return right and left tensor indices.

    In general, this will be returned as a list of length-2 corresponding
    to the right and left indices, respectively. If there *are no* right
    or left indices, that entry will be omitted from the returned list.

    Right indices come first to match the quantum computing / matrix multiplication
    convention where U_tot = U_n ... U_2 U_1.

    Args:
        tn: The tensor network to fetch the outer indices, which won't necessarily be ordered.
        signature: The signature of the bloq used to order the indices.
    """
    left_inds = {}
    right_inds = {}
    cxn: Connection
    j: int
    for ind in tn.outer_inds():
        cxn, j = ind
        if cxn.left.binst is LeftDangle:
            soq = cxn.left
            left_inds[soq.reg, soq.idx, j] = ind
        elif cxn.right.binst is RightDangle:
            soq = cxn.right
            right_inds[soq.reg, soq.idx, j] = ind
        else:
            raise ValueError(
                "Outer indices of a tensor network should be "
                "connections to LeftDangle or RightDangle"
            )

    left_ordered_inds = []
    for reg in signature.lefts():
        for idx in reg.all_idxs():
            for j in range(reg.dtype.num_qubits):
                left_ordered_inds.append(left_inds[reg, idx, j])

    right_ordered_inds = []
    for reg in signature.rights():
        for idx in reg.all_idxs():
            for j in range(reg.dtype.num_qubits):
                right_ordered_inds.append(right_inds[reg, idx, j])

    inds = []
    if right_ordered_inds:
        inds.append(right_ordered_inds)
    if left_ordered_inds:
        inds.append(left_ordered_inds)

    return inds


def quimb_to_dense(tn: 'qtn.TensorNetwork', signature: Signature) -> NDArray:
    """Contract a quimb tensor network `tn` to a dense matrix consistent with `signature`."""
    inds = get_right_and_left_inds(tn, signature)
    if tn.contraction_width() > 8:
        tn.full_simplify(inplace=True)

    if inds:
        data = tn.to_dense(*inds)
    else:
        data = tn.contract()

    return data


def bloq_to_dense(bloq: Bloq, full_flatten: bool = True) -> NDArray:
    """Return a contracted, dense ndarray representing the composite bloq.

    This function is also available as the `Bloq.tensor_contract()` method.

    This function decomposes and flattens a given bloq into a factorized CompositeBloq,
    turns that composite bloq into a Quimb tensor network, and contracts it into a dense
    matrix.

    The returned array will be 0-, 1- or 2- dimensional with indices arranged according to the
    bloq's signature. In the case of a 2-dimensional matrix, we follow the
    quantum computing / matrix multiplication convention of (right, left) order of dimensions.

    For fine-grained control over the tensor contraction, use
    `cbloq_to_quimb` and `TensorNetwork.to_dense` directly.

    Args:
        bloq: The bloq
        full_flatten: Whether to completely flatten the bloq into the smallest possible
            bloqs. Otherwise, stop flattening if custom tensors are encountered.
    """
    logging.info("bloq_to_dense() on %s", bloq)
    flat_cbloq = flatten_for_tensor_contraction(bloq, full_flatten=full_flatten)
    tn = cbloq_to_quimb(flat_cbloq)
    return quimb_to_dense(tn, bloq.signature)
