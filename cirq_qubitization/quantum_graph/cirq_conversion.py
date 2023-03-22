from collections import defaultdict
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import attrs
import cirq
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import _binst_to_cxns
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


def _for_in_vals(
    reg: FancyRegister, binst: BloqInstance, soq_assign: Dict[Soquet, NDArray['cirq.Qid']]
) -> NDArray['cirq.Qid']:
    full_shape = reg.wireshape + (reg.bitsize,)
    arg = np.empty(full_shape, dtype=object)

    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _process_binst(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, NDArray[cirq.Qid]],
) -> Optional[cirq.Operation]:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soq_assign: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]
        del soq_assign[cxn.left]

    def _in_vals(reg: FancyRegister):
        return _for_in_vals(reg, binst, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}

    op = bloq.on_registers(**in_vals)
    out_vals = in_vals.copy()

    for reg in bloq.registers.rights():
        arr = np.asarray(out_vals[reg.name])
        for idx in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=idx)
            soq_assign[soq] = arr[idx]

    return op


def _cbloq_to_cirq_circuit(
    registers: FancyRegisters, quregs: Dict[str, NDArray[cirq.Qid]], binst_graph: nx.DiGraph
) -> cirq.Circuit:
    """Transform CompositeBloq components into a `cirq.Circuit`.

    Args:
        quregs: Assignment from each register to a sequence of `cirq.Qid` for the conversion
            to a `cirq.Circuit`.
        binst_graph: A graph connecting bloq instances with edge attributes containing the
            full list of `Connection`s, as returned by `CompositeBloq._get_binst_graph()`.
            This function does not mutate `binst_graph`.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    soq_assign: Dict[Soquet, NDArray[cirq.Qid]] = {}

    # LeftDangle assignment
    for reg in registers.lefts():
        qarr = np.asarray(quregs[reg.name]).astype(object, copy=False)

        if qarr.shape != reg.wireshape + (reg.bitsize,):
            raise ValueError(f"Cirq qubit array for {reg} is the wrong shape: {qarr.shape}")

        for idx in reg.wire_idxs():
            soq = Soquet(LeftDangle, reg, idx=idx)
            soq_assign[soq] = qarr[idx]

    moments: List[cirq.Moment] = []
    for binsts in nx.topological_generations(binst_graph):
        moment: List[cirq.Operation] = []

        for binst in binsts:
            if isinstance(binst, DanglingT):
                continue

            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            op = _process_binst(binst, pred_cxns, soq_assign)
            if op:
                moment.append(op)
        if moment:
            moments.append(cirq.Moment(moment))

    return cirq.Circuit(moments)
