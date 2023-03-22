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

T = TypeVar('T')


def _for_in_vals(
    reg: FancyRegister, binst: BloqInstance, soq_assign: Dict[Soquet, NDArray[np.uint8]]
) -> NDArray[np.uint8]:
    full_shape = reg.wireshape + (reg.bitsize,)
    arg = np.empty(full_shape, dtype=np.int8)

    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _process_classical_binst(
    binst: BloqInstance, pred_cxns: Iterable[Connection], soq_assign: Dict[Soquet, NDArray[int]]
):
    print(f"Applying {binst}")

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate input with expected API
    def _in_vals(reg: FancyRegister):
        return _for_in_vals(reg, binst, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}

    # Apply function
    out_vals = bloq.apply_classical(**in_vals)

    # Use output
    for out_i, reg in bloq.registers.rights():
        arr = np.asarray(out_vals[out_i]).astype(np.uint8, casting='safe', copy=False)
        for idx in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=idx)
            soq_assign[soq] = arr[idx]


def _cbloq_apply_classical(
    registers: FancyRegisters, vals: Dict[str, NDArray], binst_graph: nx.DiGraph
):
    soq_assign: Dict[Soquet, NDArray[int]] = {}

    # LeftDangle assignment
    for reg in registers.lefts():
        arr = np.asarray(vals[reg.name]).astype(np.uint8, casting='safe', copy=False)
        if arr.shape != reg.wireshape + (reg.bitsize,):
            raise ValueError(f"Classical values for {reg} are the wrong shape: {arr.shape}")

        for idx in reg.wire_idxs():
            soq = Soquet(LeftDangle, reg, idx=idx)
            soq_assign[soq] = arr[idx]

    # Bloq-by-bloq application
    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        _process_classical_binst(binst, pred_cxns, soq_assign)

    # Track bloq-to-dangle name changes
    final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
    for cxn in final_preds:
        soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_vals(reg: FancyRegister):
        return _for_in_vals(reg, RightDangle, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in registers.rights()}
    return final_vals, soq_assign
