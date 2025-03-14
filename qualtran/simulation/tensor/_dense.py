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
from collections import defaultdict
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from numpy.typing import NDArray

from qualtran import Bloq, Connection, ConnectionT, Signature

from ._flattening import flatten_for_tensor_contraction
from ._quimb import _IndT, cbloq_to_quimb, cbloq_to_superquimb

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
            for j in range(reg.dtype.num_bits):
                if idx:
                    inds.append((outgoing[reg.name][idx], j))  # type: ignore[index]
                else:
                    inds.append((outgoing[reg.name], j))  # type: ignore[arg-type]
    for reg in signature.lefts():
        for idx in reg.all_idxs():
            for j in range(reg.dtype.num_bits):
                if idx:
                    inds.append((incoming[reg.name][idx], j))  # type: ignore[index]
                else:
                    inds.append((incoming[reg.name], j))  # type: ignore[arg-type]

    return inds


def _group_outer_inds(
    tn: 'qtn.TensorNetwork', signature: Signature, superoperator: bool = False
) -> List[List[_IndT]]:
    """Group outer indices of a tensor network.

    This is used by 'bloq_to_dense` and `quimb_to_dense` to return a 1-, 2-, or 4-dimensional
    array depending on the quantity and type of outer indices in the tensor network. See
    the docstring for `Bloq.tensor_contract()` for more informaiton.

    Args:
        tn: The tensor network to fetch the outer indices, which won't necessarily be ordered.
        signature: The signature of the bloq used to order the indices.
        superoperator: Whether `tn` is a pure-state or open-system tensor network.
    """
    reg_name: str
    idx: Tuple[int, ...]
    j: int
    group: str
    _KeyT = Tuple[str, Tuple[int, ...], int]
    ind_groups_d: Dict[str, Dict[_KeyT, _IndT]] = defaultdict(dict)

    for ind in tn.outer_inds():
        reg_name, idx, j, group = ind
        ind_groups_d[group][reg_name, idx, j] = ind

    ind_groups_l: Dict[str, List[_IndT]] = defaultdict(list)

    def _sort_group(regs, group_name):
        for reg in regs:
            for idx in reg.all_idxs():
                for j in range(reg.dtype.num_bits):
                    ind_groups_l[group_name].append(ind_groups_d[group_name][reg.name, idx, j])

    if superoperator:
        _sort_group(signature.lefts(), 'lf')
        _sort_group(signature.lefts(), 'lb')
        _sort_group(signature.rights(), 'rf')
        _sort_group(signature.rights(), 'rb')
        group_names = ['rf', 'rb', 'lf', 'lb']
    else:
        _sort_group(signature.lefts(), 'l')
        _sort_group(signature.rights(), 'r')
        group_names = ['r', 'l']

    inds = [ind_groups_l[groupname] for groupname in group_names if ind_groups_l[groupname]]
    return inds


def quimb_to_dense(
    tn: 'qtn.TensorNetwork', signature: Signature, superoperator: bool = False
) -> NDArray:
    """Contract a quimb tensor network `tn` to a dense matrix consistent with `signature`."""
    inds = _group_outer_inds(tn, signature, superoperator=superoperator)
    if tn.contraction_width() > 8:
        tn.full_simplify(inplace=True)

    if inds:
        data = tn.to_dense(*inds)
    else:
        data = tn.contract()

    return data


def bloq_to_dense(bloq: Bloq, full_flatten: bool = True, superoperator: bool = False) -> NDArray:
    """Return a contracted, dense ndarray representing the composite bloq.

    This function is also available as the `Bloq.tensor_contract()` method.

    This function decomposes and flattens a given bloq into a factorized CompositeBloq,
    turns that composite bloq into a Quimb tensor network, and contracts it into a dense
    ndarray.

    The returned array will be 0-, 1-, 2-, or 4-dimensional with indices arranged according to the
    bloq's signature and the type of simulation requested via the `superoperator` flag.

    If `superoperator` is set to False (the default), a pure-state tensor network will be
    constructed.
     - If `bloq` has all thru-registers, the dense tensor will be 2-dimensional with shape `(n, n)`
       where `n` is the number of bits in the signature. We follow the linear algebra convention
       and order the indices as (right, left) so the matrix-vector product can be used to evolve
       a state vector.
     - If `bloq` has all left- or all right-registers, the tensor will be 1-dimensional with
       shape `(n,)`. Note that we do not distinguish between 'row' and 'column' vectors in this
       function.
     - If `bloq` has no external registers, the contracted form is a 0-dimensional complex number.

    If `superoperator` is set to True, an open-system tensor network will be constructed.
     - States result in a 2-dimensional density matrix with indices (right_forward, right_backward)
       or (left_forward, left_backward) depending on whether they're input or output states.
     - Operations result in a 4-dimensional tensor with indices (right_forward, right_backward,
       left_forward, left_backward).

    For fine-grained control over the tensor contraction, use
    `cbloq_to_quimb` and `TensorNetwork.to_dense` directly.

    Args:
        bloq: The bloq
        full_flatten: Whether to completely flatten the bloq into the smallest possible
            bloqs. Otherwise, stop flattening if custom tensors are encountered.
        superoperator: If toggled to True, do an open-system simulation. This supports
            non-unitary operations like measurement, but is more costly and results in
            higher-dimension resultant tensors.
    """
    logging.info("bloq_to_dense() on %s", bloq)
    flat_cbloq = flatten_for_tensor_contraction(bloq, full_flatten=full_flatten)
    if superoperator:
        tn = cbloq_to_superquimb(flat_cbloq)
    else:
        tn = cbloq_to_quimb(flat_cbloq)
    return quimb_to_dense(tn, bloq.signature, superoperator=superoperator)
