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

SoquetT = Union[Soquet, NDArray[Soquet]]


def _process_binst(
    binst: BloqInstance, soqmap: Dict[Soquet, Sequence[cirq.Qid]], binst_graph: nx.DiGraph
) -> Optional[cirq.Operation]:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soqmap: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    if isinstance(binst, DanglingT):
        return None

    pred_cxns, _ = _binst_to_cxns(binst, binst_graph)

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soqmap[cxn.right] = soqmap[cxn.left]
        del soqmap[cxn.left]

    bloq = binst.bloq

    # Pull out the qubits from soqmap into qumap which has string keys.
    # This implicitly joins things with the same name.
    quregs: Dict[str, List[cirq.Qid]] = defaultdict(list)
    for reg in bloq.registers.lefts():
        for li in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=li)
            quregs[reg.name].extend(soqmap[soq])
            del soqmap[soq]

    op = bloq.on_registers(**quregs)

    # We pluck things back out from their collapsed by-name qumap into soqmap
    # This does implicit splitting.
    for reg in bloq.registers.rights():
        qarr = np.asarray(quregs[reg.name])
        for ri in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=ri)
            qs = qarr[ri]
            if isinstance(qs, np.ndarray):
                qs = qs.tolist()
            else:
                qs = [qs]
            soqmap[soq] = qs

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[FancyRegister, NDArray[cirq.Qid]], binst_graph: nx.DiGraph
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
    # A mapping of soquet to qubits that we update as operations are appended to the circuit.
    soqmap = {}
    for reg in quregs.keys():
        qarr = np.asarray(quregs[reg])
        for ii in reg.wire_idxs():
            soqmap[Soquet(LeftDangle, reg, idx=ii)] = qarr[ii]

    moments: List[cirq.Moment] = []
    for i, binsts in enumerate(nx.topological_generations(binst_graph)):
        mom: List[cirq.Operation] = []
        for binst in binsts:
            op = _process_binst(binst, soqmap, binst_graph)
            if op:
                mom.append(op)
        if mom:
            moments.append(cirq.Moment(mom))

    return cirq.Circuit(moments)
