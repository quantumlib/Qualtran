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
from collections.abc import Iterable
from typing import Any, cast, TypeAlias, Union

import attrs
import numpy as np
import quimb.tensor as qtn

from qualtran import (
    Bloq,
    CompositeBloq,
    Connection,
    ConnectionT,
    LeftDangle,
    QBit,
    Register,
    RightDangle,
    Soquet,
    SoquetT,
)
from qualtran._infra.composite_bloq import _cxns_to_cxn_dict, BloqBuilder

logger = logging.getLogger(__name__)

_IndT: TypeAlias = Any


def cbloq_to_quimb(cbloq: CompositeBloq, friendly_indices: bool = False) -> qtn.TensorNetwork:
    """Convert a composite bloq into a tensor network.

    This function will call `Bloq.my_tensors` on each subbloq in the composite bloq to add
    tensors to a quimb tensor network. This method has no default fallback, so you likely want to
    call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
    smallest form first. The small bloqs that result from a flattening 1) likely already have
    their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
    path.

    Args:
        cbloq: The composite bloq.
        friendly_indices: If set to True, the outer indices of the tensor network will be renamed
            from their Qualtran-computer-readable form to human-friendly strings. This may be
            useful if you plan on manually manipulating the resulting tensor network but will
            preclude any further processing by Qualtran functions. The indices are named
            {soq.reg.name}{soq.idx}_{j}{side}, where j is the individual bit index and side is 'l'
            or 'r' for left or right, respectively.

    Returns:
        The tensor network
    """
    tn = qtn.TensorNetwork([])

    logging.info(
        "Constructing a tensor network for composite bloq of size %d", len(cbloq.bloq_instances)
    )

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxns_to_cxn_dict(bloq.signature.lefts(), pred_cxns, get_me=lambda cxn: cxn.right)
        out_d = _cxns_to_cxn_dict(bloq.signature.rights(), succ_cxns, get_me=lambda cxn: cxn.left)

        for tensor in bloq.my_tensors(inc_d, out_d):
            if isinstance(tensor, DiscardInd):
                raise ValueError(
                    f"During tensor simulation, {bloq} tried to discard information. This requires using `tensor_contract(superoperator=True)` or `cbloq_to_superquimb`."
                )
            tn.add(tensor)

    # Special case: Add indices corresponding to unused wires
    for cxn in cbloq.connections:
        if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:
            # Connections that directly tie LeftDangle to RightDangle
            for id_tensor in _get_placeholder_tensors(cxn):
                tn.add(id_tensor)

    return tn.reindex(_get_outer_indices(tn, friendly_indices=friendly_indices))


def _get_placeholder_tensors(cxn):
    """Get identity placeholder tensors to directly connect LeftDangle to RightDangle.

    This function is used in `cbloq_to_quimb` and `cbloq_to_superquimb` for the following
    contingency:

    >>> for cxn in cbloq.connections:
    >>>     if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:

    This is needed because `CompositeBloq.iter_bloqnections` ignores `LeftDangle/RightDangle`
    bloqs, and therefore we never see connections that exist only between LeftDangle and
    RightDangle sentinel values.
    """
    for j in range(cxn.left.reg.bitsize):
        placeholder = Soquet(None, Register('simulation_placeholder', QBit()))  # type: ignore
        Connection(cxn.left, placeholder)
        yield qtn.Tensor(
            data=np.eye(2),
            inds=[(Connection(cxn.left, placeholder), j), (Connection(placeholder, cxn.right), j)],
        )


_OuterIndT = tuple[str, tuple[int, ...], int, str]


def _get_outer_indices(
    tn: 'qtn.TensorNetwork', friendly_indices: bool = False
) -> dict[_IndT, Union[str, _OuterIndT]]:
    """Provide a mapping for a tensor network's outer indices.

    Internal indices effectively use `qualtran.Connection` objects as their indices. The
    outer indices correspond to connections to `DanglingT` items, and you end up having to
    do logic to disambiguate left dangling indices from right dangling indices. This function
    facilitates re-indexing the tensor network's outer indices to better match a bloq's signature.

    In particular, we map each outer index to a tuple (reg.name, soq.idx, j, group) where
    group is 'l' or 'r' for left or right indices.

    If `friendly_indices` is set to True, the tuple of items is converted to a string.

    This function is called at the end of `cbloq_to_quimb` as part of a `tn.reindex(...) operation.
    """
    ind_name_map: dict[_IndT, Union[str, _OuterIndT]] = {}

    # Each index is a (cxn: Connection, j: int) tuple.
    cxn: Connection
    j: int

    for ind in tn.outer_inds():
        cxn, j = ind
        if cxn.left.binst is LeftDangle:
            soq = cxn.left
            group = 'l'
        elif cxn.right.binst is RightDangle:
            soq = cxn.right
            group = 'r'
        else:
            raise ValueError(
                f"Outer indices of a tensor network should be "
                f"connections to LeftDangle or RightDangle, not {cxn}"
            )

        if friendly_indices:
            # Turn everything to strings
            idx_str = f'{soq.idx}' if soq.idx else ''
            ind_name_map[ind] = f'{soq.reg.name}{idx_str}_{j}{group}'
        else:
            # Keep as tuple
            ind_name_map[ind] = (soq.reg.name, soq.idx, j, group)

    return ind_name_map


@attrs.frozen
class DiscardInd:
    """Return `DiscardInd` in `Bloq.my_tensors()` to indicate an index should be discarded.

    We cannot discard an index from a state-vector pure-state simulation, so any bloq that
    returns `DiscardInd` in its `my_tensors` method will cause an error in the ordinary
    tensor contraction simulator.

    We can discard indices in open-system simulations by tracing out the index. When using
    `Bloq.tensor_contract(superoperator=True)`, the index contained in a `DiscardInd` will be
    traced out of the superoperator tensor network.

    Args:
        ind_tuple: The index to trace out, of the form (cxn, j) where `j` addresses
            individual bits.
    """

    ind_tuple: tuple['ConnectionT', int]


def make_forward_tensor(t: qtn.Tensor):
    new_inds = [(*ind, True) for ind in t.inds]

    t2 = t.copy()
    t2.modify(inds=new_inds)
    return t2


def make_backward_tensor(t: qtn.Tensor):
    new_inds = []
    for ind in t.inds:
        new_inds.append((*ind, False))

    t2 = t.H
    t2.modify(inds=new_inds, tags=t.tags | {'dag'})
    return t2


def cbloq_to_superquimb(cbloq: CompositeBloq, friendly_indices: bool = False) -> qtn.TensorNetwork:
    """Convert a composite bloq into a superoperator tensor network.

    This simulation strategy can handle non-unitary dynamics, but is more costly.

    This function will call `Bloq.my_tensors` on each subbloq in the composite bloq to add
    tensors to a quimb tensor network. This uses ths system+environment strategy for modeling
    open system dynamics. In contrast to `cbloq_to_quimb`, each bloq will have
    its tensors added twice: once to the part of the network representing the "forward"
    wavefunction, and its conjugate added to the part of the network representing the "backward"
    part of the wavefunction. If the bloq returns a sentinel value of the `DiscardInd` class,
    that particular index is *traced out*: the forward and backward copies of the index are joined.
    This corresponds to removing the qubit from the computation and integrating over its possible
    values. Arbitrary non-unitary dynamics can be modeled by unitary interaction of the 'system'
    with an 'environment' that is traced out.

    If a bloq returns a value of type `DiscardInd` in its tensors, this function must be
    used. The ordinary `cbloq_to_quimb` will raise an error.

    Args:
        cbloq: The composite bloq.
        friendly_indices: If set to True, the outer indices of the tensor network will be renamed
            from their Qualtran-computer-readable form to human-friendly strings. This may be
            useful if you plan on manually manipulating the resulting tensor network but will
            preclude any further processing by Qualtran functions. The indices are named
            {soq.reg.name}{soq.idx}_{j}{side}{direction}, where j is the individual bit index,
            side is 'l' or 'r' for left or right (resp.), and direction is 'f' or 'b' for the
            forward or backward (adjoint) wavefunctions.
    """
    tn = qtn.TensorNetwork([])

    logging.info(
        "Constructing a super tensor network for composite bloq of size %d",
        len(cbloq.bloq_instances),
    )

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxns_to_cxn_dict(bloq.signature.lefts(), pred_cxns, get_me=lambda cxn: cxn.right)
        out_d = _cxns_to_cxn_dict(bloq.signature.rights(), succ_cxns, get_me=lambda cxn: cxn.left)

        for tensor in bloq.my_tensors(inc_d, out_d):
            if isinstance(tensor, DiscardInd):
                dind = tensor.ind_tuple
                tn.reindex({(*dind, True): dind, (*dind, False): dind}, inplace=True)
            else:
                forward_tensor = make_forward_tensor(tensor)
                backward_tensor = make_backward_tensor(tensor)
                tn.add(forward_tensor)
                tn.add(backward_tensor)

    # Special case: Add indices corresponding to unused wires
    for cxn in cbloq.connections:
        if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:
            # Connections that directly tie LeftDangle to RightDangle
            for id_tensor in _get_placeholder_tensors(cxn):
                forward_tensor = make_forward_tensor(id_tensor)
                backward_tensor = make_backward_tensor(id_tensor)
                tn.add(forward_tensor)
                tn.add(backward_tensor)

    return tn.reindex(_get_outer_superindices(tn, friendly_indices=friendly_indices))


_SuperOuterIndT = tuple[str, tuple[int, ...], int, str]


def _get_outer_superindices(
    tn: 'qtn.TensorNetwork', friendly_indices: bool = False
) -> dict[_IndT, Union[str, _SuperOuterIndT]]:
    """Provide a mapping for a super-tensor network's outer indices.

    Internal indices effectively use `qualtran.Connection` objects as their indices. The
    outer indices correspond to connections to `DanglingT` items, and you end up having to
    do logic to disambiguate left dangling indices from right dangling indices. This function
    facilitates re-indexing the tensor network's outer indices to better match a bloq's signature.

    In particular, we map each outer index to a tuple (reg.name, soq.idx, j, group) where
    group is 'lf', 'lb', 'rf', or 'rb' corresponding to (left or right) x (forward or backward)
    indices.

    If `friendly_indices` is set to True, the tuple of items is converted to a string.

    This function is called at the end of `cbloq_to_superquimb` as part of a `tn.reindex(...)
    operation.
    """
    # Each index is a (cxn: Connection, j: int, forward: bool) tuple.
    cxn: Connection
    j: int
    forward: bool

    ind_name_map: dict[_IndT, Union[str, _SuperOuterIndT]] = {}
    for ind in tn.outer_inds():
        cxn, j, forward = ind
        if cxn.left.binst is LeftDangle:
            soq = cxn.left
            group = 'lf' if forward else 'lb'
        elif cxn.right.binst is RightDangle:
            soq = cxn.right
            group = 'rf' if forward else 'rb'
        else:
            raise ValueError(
                f"Outer indices of a tensor network should be "
                f"connections to LeftDangle or RightDangle, not {cxn}"
            )

        if friendly_indices:
            idx_str = f'{soq.idx}' if soq.idx else ''
            ind_name_map[ind] = f'{soq.reg.name}{idx_str}_{j}{group}'
        else:
            ind_name_map[ind] = (soq.reg.name, soq.idx, j, group)

    return ind_name_map


def _add_classical_kets(bb: BloqBuilder, registers: Iterable[Register]) -> dict[str, 'SoquetT']:
    """Use `bb` to add `IntState(0)` for all the `vals`."""

    from qualtran.bloqs.basic_gates import IntState

    soqs: dict[str, 'SoquetT'] = {}
    for reg in registers:
        if reg.shape:
            reg_vals = np.zeros(reg.shape, dtype=int)
            soq = np.empty(reg.shape, dtype=object)
            for idx in reg.all_idxs():
                soq[idx] = bb.add(IntState(val=cast(int, reg_vals[idx]), bitsize=reg.bitsize))
        else:
            soq = bb.add(IntState(val=0, bitsize=reg.bitsize))

        soqs[reg.name] = soq
    return soqs


def initialize_from_zero(bloq: Bloq):
    """Take `bloq` and compose it with initial zero states for each left register.

    This can be contracted to a state vector for a given unitary.
    """
    bb = BloqBuilder()

    # Add the initial 'kets' according to the provided values.
    in_soqs = _add_classical_kets(bb, bloq.signature.lefts())

    # Add the bloq itself
    out_soqs = bb.add_d(bloq, **in_soqs)
    return bb.finalize(**out_soqs)
