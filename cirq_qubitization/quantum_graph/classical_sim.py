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


def _process_classical_binst(binst: BloqInstance, pred_cxns, datamap: Dict[Soquet, NDArray]):
    print(f"Applying {binst}")

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        datamap[cxn.right] = datamap[cxn.left]

    bloq = binst.bloq

    # Pull out the qubits from soqmap into qumap which has string keys.
    # This implicitly joins things with the same name.
    indata = {}
    for reg in bloq.registers.lefts():
        full_shape = reg.wireshape + (reg.bitsize,)
        arg = np.empty(full_shape)

        for idx in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=idx)
            arg[idx, :] = datamap[soq]

        indata[reg.name] = arg

    for k, v in indata.items():
        print('  ', k, v)
    outdata = bloq.apply_classical(**indata)
    print('--')
    for v in outdata:
        print('  ', v)

    # We pluck things back out from their collapsed by-name qumap into soqmap
    # This does implicit splitting.
    for reg, arr in zip(bloq.registers.rights(), outdata):
        arr = np.asarray(arr)
        for ri in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=ri)
            datamap[soq] = arr[ri]


def _apply_classical_cbloq(
    registers: FancyRegisters, data: Dict[str, NDArray], binst_graph: nx.DiGraph
):
    datamap: Dict[Soquet, Any] = {}
    for reg in registers.lefts():
        arr = np.asarray(data[reg.name])
        assert arr.shape == reg.wireshape + (reg.bitsize,), arr.shape
        for ii in reg.wire_idxs():
            datamap[Soquet(LeftDangle, reg, idx=ii)] = arr[ii]

    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        _process_classical_binst(binst, pred_cxns, datamap)

    final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
    for cxn in final_preds:
        datamap[cxn.right] = datamap[cxn.left]

    final_data = {}
    for reg in registers.rights():
        full_shape = reg.wireshape + (reg.bitsize,)
        arg = np.empty(full_shape)

        for idx in reg.wire_idxs():
            soq = Soquet(RightDangle, reg, idx=idx)
            arg[idx] = datamap[soq]

        final_data[reg.name] = arg

    return final_data, datamap
