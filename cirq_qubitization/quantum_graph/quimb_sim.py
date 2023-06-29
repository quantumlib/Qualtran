import itertools
from typing import Dict, List, Optional, Tuple

import quimb.tensor as qtn
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _cxn_to_soq_dict,
    _flatten_soquet_collection,
    _get_flat_dangling_soqs,
    CompositeBloq,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    Soquet,
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
        the convention that each binst will respect its predecessors outgoing index names but
        is in charge of its own outgoing index names.

        This convention breaks down at the end of our graph because we wish our external quimb
        indices match the composite bloq's dangling soquets. Therefore: when the successor
        is `RightDangle` the binst will respect the dangling soquets for naming its outgoing
        indices.
        """
        if isinstance(cxn.right.binst, DanglingT):
            return cxn.right
        return cxn.left

    for binst, incoming, outgoing in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxn_to_soq_dict(
            bloq.registers.lefts(),
            incoming,
            get_me=lambda cxn: cxn.right,
            get_assign=lambda cxn: cxn.left,
        )
        out_d = _cxn_to_soq_dict(
            bloq.registers.rights(),
            outgoing,
            get_me=lambda cxn: cxn.left,
            get_assign=_assign_outgoing,
        )

        bloq.add_my_tensors(tn, binst, incoming=inc_d, outgoing=out_d)
        if pos is not None:
            fix[tuple([binst])] = pos[binst]

    return tn, fix


def get_right_and_left_inds(registers: FancyRegisters) -> List[List[Soquet]]:
    """Return right and left indices.

    In general, this will be returned as a list of length-2 corresponding
    to the right and left indices, respectively. If there *are* no right
    or left indices, that entry will be omitted from the returned list.

    Right indices come first to match the quantum computing / matrix multiplication
    convention where U_tot = U_n ... U_2 U_1.
    """
    inds = []
    rsoqs = _get_flat_dangling_soqs(registers, right=True)
    if rsoqs:
        inds.append(rsoqs)
    lsoqs = _get_flat_dangling_soqs(registers, right=False)
    if lsoqs:
        inds.append(lsoqs)
    return inds


def _cbloq_to_dense(cbloq: CompositeBloq) -> NDArray:
    """Return a contracted, dense ndarray representing the composite bloq.

    The public version of this function is available as the `CompositeBloq.tensor_contract()`
    method.

    This constructs a tensor network and then contracts it according to the cbloq's registers,
    i.e. the dangling indices. The returned array will be 0-, 1- or 2- dimensional. If it is
    a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
    of (right, left) indices.

    For more fine grained control over the final shape of the tensor, use
    `cbloq_to_quimb` and `TensorNetwork.to_dense` directly.
    """
    tn, _ = cbloq_to_quimb(cbloq)
    inds = get_right_and_left_inds(cbloq.registers)

    if inds:
        return tn.to_dense(*inds)

    return tn.contract()


def _cbloq_as_contracted_tensor_data_and_inds(
    cbloq: CompositeBloq,
    registers: FancyRegisters,
    incoming: Dict[str, SoquetT],
    outgoing: Dict[str, SoquetT],
) -> Tuple[NDArray, List[Soquet]]:
    """`add_my_tensors` helper for contracting `cbloq` and adding it as a dense tensor.

    First, we turn the composite bloq into a TensorNetwork with `cbloq_to_quimb`. Then
    we contract it to a dense ndarray. This function returns the dense array as well as
    the indices munged from `incoming` and `outgoing` to match the structure of the ndarray.
    """

    # Turn into a dense ndarray, but instead of folding into a 1- or 2-
    # dimensional state/effect or unitary; we keep all the indices as
    # distinct dimensions.
    rsoqs = _get_flat_dangling_soqs(registers, right=True)
    lsoqs = _get_flat_dangling_soqs(registers, right=False)
    inds_for_contract = rsoqs + lsoqs
    assert len(inds_for_contract) > 0
    tn, _ = cbloq_to_quimb(cbloq)
    data = tn.to_dense(*([x] for x in inds_for_contract))
    assert data.ndim == len(inds_for_contract)

    # Now we just need to make sure the Soquets provided to us are in the correct
    # order: namely the same order as how we got the indices to contract the composite bloq.
    osoqs = (outgoing[reg.name] for reg in registers.rights())
    isoqs = (incoming[reg.name] for reg in registers.lefts())
    inds_for_adding = _flatten_soquet_collection(itertools.chain(osoqs, isoqs))
    assert len(inds_for_adding) == len(inds_for_contract)

    # Data and `inds_for_adding` can be used as `Tensor(data, inds)` arguments.
    return data, inds_for_adding
