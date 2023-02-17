from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, get_soquets, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    Soquet,
)


def _cxn_to_soq_dict(
    regs: Iterable[FancyRegister],
    cxns: Iterable[Connection],
    get_me: Callable[[Connection], Soquet],
    get_assign: Callable[[Connection], Soquet],
) -> Dict[str, SoquetT]:
    """Helper function to get a dictionary of incoming or outgoing soquets.

    This is used in `cbloq_to_quimb`.

    Args:
        regs: Left or right registers (used as a reference to initialize multidimensional
            registers correctly).
        cxns: Predecessor or successor connections from which we get the soquets of interest.
        get_me: A function that says which soquet is used to derive keys for the returned
            dictionary. Generally: if `cxns` is predecessor connections, this will return the
            `right` element of the connection and opposite of successor connections.
        get_assign: A function that says which soquet is used to dervice the values for the
            returned dictionary. Generally, this is the opposite side vs. `get_me`, but we
            do something fancier in `cbloq_to_quimb`.
    """
    soqdict: Dict[str, SoquetT] = {}

    # Initialize multi-dimensional dictionary values.
    for reg in regs:
        if reg.wireshape:
            soqdict[reg.name] = np.empty(reg.wireshape, dtype=object)

    # In the abstract: set `soqdict[me] = assign`. Specifically: use the register name as
    # keys and handle multi-dimensional registers.
    for cxn in cxns:
        me = get_me(cxn)
        assign = get_assign(cxn)

        if me.reg.wireshape:
            soqdict[me.reg.name][me.idx] = assign
        else:
            soqdict[me.reg.name] = assign

    return soqdict


def cbloq_to_quimb(
    cbloq: CompositeBloq, pos: Optional[Dict[BloqInstance, Tuple[float, float]]] = None
) -> Tuple[qtn.TensorNetwork, Dict]:
    """Convert a composite bloq into a Quimb tensor network.

    External indices are the dangling soquets of the compute graph.

    Args:
        cbloq: The composite bloq.
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


def cbloq_to_dense(cbloq: CompositeBloq) -> NDArray:
    """Return a contracted, dense ndarray representing the composite bloq.

    This constructs a tensor network and then contracts it according to the dangling
    indices. For more fine grained control over the final shape of the tensor, use
    `cbloq_to_quimb` and `TensorNetwork.to_dense` directly.
    """

    tn, _ = cbloq_to_quimb(cbloq)
    inds = []
    lsoqs = get_soquets(cbloq.registers, right=False)
    if lsoqs:
        inds.append(lsoqs.values())
    rsoqs = get_soquets(cbloq.registers, right=True)
    if rsoqs:
        inds.append(rsoqs.values())

    if inds:
        return tn.to_dense(*inds)

    return tn.contract()


def bloq_to_dense(bloq: Bloq) -> NDArray:
    """Return a dense ndarray representing this bloq."""
    tn = qtn.TensorNetwork([])
    inds = []
    lsoqs = get_soquets(bloq.registers, right=False)
    if lsoqs:
        inds.append(lsoqs.values())
    rsoqs = get_soquets(bloq.registers, right=True)
    if rsoqs:
        inds.append(rsoqs.values())
    bloq.add_my_tensors(tn, None, incoming=lsoqs, outgoing=rsoqs)
    matrix = tn.to_dense(*inds)
    return matrix
