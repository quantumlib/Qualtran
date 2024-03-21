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
import itertools
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import quimb.tensor as qtn

from qualtran import (
    Bloq,
    BloqInstance,
    CompositeBloq,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
    SoquetT,
)
from qualtran._infra.composite_bloq import (
    _cxn_to_soq_dict,
    _flatten_soquet_collection,
    _get_flat_dangling_soqs,
)


def cbloq_to_quimb(
    cbloq: CompositeBloq, pos: Optional[Dict[BloqInstance, Tuple[float, float]]] = None
) -> Tuple[qtn.TensorNetwork, Dict]:
    """Convert a composite bloq into a Quimb tensor network.

    External indices are the dangling soquets of the compute graph.

    Args:
        cbloq: The composite bloq. A composite bloq is a container class analogous to a
            `TensorNetwork`. This function will simply add the tensor(s) for each Bloq
            that constitutes the `CompositeBloq`.
        pos: Optional mapping of each `binst` to (x, y) coordinates which will be converted
            into a `fix` dictionary appropriate for `qtn.TensorNetwork.draw()`.

    Returns:
        tn: The `qtn.TensorNetwork` representing the quantum graph. This is constructed
            by delegating to each bloq's `add_my_tensors` method.
        fix: A version of `pos` suitable for `TensorNetwork.draw()`
    """
    tn = qtn.TensorNetwork([])
    fix = {}

    def _assign_outgoing(cxn: Connection) -> Soquet:
        """Logic for naming outgoing indices in quimb-land.

        In our representation, a `Connection` is a tuple of soquets. In quimb, connections are
        made between nodes with indices having the same name. Conveniently, the indices
        don't have to have string names, so we use a Soquet.

        Each binst makes a qtn.Tensor, and we use a soquet to name each index. We choose
        the convention that each binst will respect its predecessors' outgoing index names but
        is in charge of its own outgoing index names.

        This convention breaks down at the end of our graph because we wish our external quimb
        indices match the composite bloq's dangling soquets. Therefore: when the successor
        is `RightDangle` the binst will respect the dangling soquets for naming its outgoing
        indices.
        """
        if isinstance(cxn.right.binst, DanglingT):
            return cxn.right
        return cxn.left

    visited_bloqs: Set[BloqInstance] = set()
    for binst, incoming, outgoing in cbloq.iter_bloqnections():
        bloq = binst.bloq
        visited_bloqs.add(binst)
        assert isinstance(bloq, Bloq)

        inc_d = _cxn_to_soq_dict(
            bloq.signature.lefts(),
            incoming,
            get_me=lambda cxn: cxn.right,
            get_assign=lambda cxn: cxn.left,
        )
        out_d = _cxn_to_soq_dict(
            bloq.signature.rights(),
            outgoing,
            get_me=lambda cxn: cxn.left,
            get_assign=_assign_outgoing,
        )

        bloq.add_my_tensors(tn, binst, incoming=inc_d, outgoing=out_d)
        if pos is not None:
            fix[tuple([binst])] = pos[binst]

    # Special case: Add variables corresponding to all registers that don't connect to any Bloq.
    # This is needed because `CompositeBloq.iter_bloqnections` ignores `LeftDangle/RightDangle`
    # bloqs, and therefore we never see connections that exist only between LeftDangle and
    # RightDangle bloqs.
    for cxn in cbloq.connections:
        if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:
            # This register has no Bloq acting on it, and thus it would not have a variable in
            # the tensor network. Add an identity tensor acting on this register to make sure the
            # tensor network has variables corresponding to all input / output registers.
            tn.add(qtn.Tensor(data=np.eye(2**cxn.left.reg.bitsize), inds=[cxn.right, cxn.left]))

    return tn, fix


def _get_index(soq: Soquet, d: Dict[str, SoquetT]) -> Soquet:
    # Helper function to index into `d` according to soq.reg.name and soq.idx.
    soq_or_arr = d[soq.reg.name]
    if soq.idx:
        return soq_or_arr[soq.idx]
    return soq_or_arr


def cbloq_as_contracted_tensor(
    cbloq: CompositeBloq,
    incoming: Dict[str, SoquetT],
    outgoing: Dict[str, SoquetT],
    tags: Optional[Sequence[Any]] = None,
) -> qtn.Tensor:
    """`add_my_tensors` helper for contracting `cbloq` and adding it as a dense tensor.

    First, we turn the composite bloq into a TensorNetwork with `cbloq_to_quimb`. Then
    we contract it to a single tensor. We then make sure the indices match the desired
    `incoming` and `outgoing` indices.
    """
    # Contract into one tensor.
    tn, _ = cbloq_to_quimb(cbloq)
    tensor: qtn.Tensor = tn.contract(preserve_tensor=True)
    assert isinstance(tensor, qtn.Tensor), tensor

    # TODO: choose one way of doing this.
    signature = cbloq.signature
    rsoqs = _get_flat_dangling_soqs(signature, right=True)
    lsoqs = _get_flat_dangling_soqs(signature, right=False)
    lsoqs2 = [soq for soq in tensor.inds if soq.binst is LeftDangle]
    rsoqs2 = [soq for soq in tensor.inds if soq.binst is RightDangle]
    assert set(lsoqs) == set(lsoqs2), f'{lsoqs} != {lsoqs2}'
    assert set(rsoqs) == set(rsoqs2)
    # TODO: end duplicate region

    # Now we just need to make sure the indices (soquets) are the ones requested
    # by the caller of this function.
    ind_map = {soq: _get_index(soq, incoming) for soq in lsoqs}
    ind_map |= {soq: _get_index(soq, outgoing) for soq in rsoqs}
    tensor.reindex(ind_map, inplace=True)

    # We'll also set the tags.
    tensor.modify(tags=tags)
    return tensor
