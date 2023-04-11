from typing import *

import networkx as nx
import numpy as np

from cirq_qubitization.quantum_graph.composite_bloq import _binst_to_cxns, CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


class Allocator:
    def __init__(self):
        self._free = []
        self.i = 0

    def alloc(self):
        if self._free:
            return self._free.pop(0)
        i = self.i
        self.i += 1
        return i

    def free(self, i):
        self._free.append(i)


def _get_in_vals(binst: BloqInstance, reg: FancyRegister, soq_assign: Dict[Soquet, Any]):
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    # TODO: use for left dangle?

    if not reg.wireshape:
        return soq_assign[Soquet(binst, reg)]

    if reg.bitsize > 64:
        raise NotImplementedError("Come back later")

    arg = np.empty(reg.wireshape, dtype=np.uint64)
    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _binst_do_score(binst: BloqInstance, pred_cxns: Iterable[Connection], assign, alloc):
    """Call `apply_classical` on a given binst."""

    # Track inter-Bloq name changes
    layer = []
    for cxn in pred_cxns:
        assign[cxn.right] = assign[cxn.left]
        layer.append(assign[cxn.right])

    # def _in_vals(reg: FancyRegister):
    #     # close over binst and `soq_assign`
    #     return _get_in_vals(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    # in_vals = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}

    # Apply function
    # out_vals = bloq.apply_classical(**in_vals)

    # Use output
    for reg in bloq.registers.rights():
        if reg.wireshape:
            for idx in reg.wire_idxs():
                soq = Soquet(binst, reg, idx=idx)

                if soq in assign:
                    pass
                else:
                    assign[soq] = alloc.alloc()

                layer.append(assign[soq])
        else:
            soq = Soquet(binst, reg)
            if soq in assign:
                pass
            else:
                assign[soq] = alloc.alloc()
            layer.append(assign[soq])

    for cxn in pred_cxns:
        if cxn.right.reg.side is Side.LEFT:
            alloc.free(assign[cxn.right])

    return layer


def cbloq_musical_score(registers: FancyRegisters, binst_graph: nx.DiGraph):
    """Propogate `apply_classical` calls through a composite bloq's contents.

    Args:
        registers: The cbloq's registers for validating inputs
        vals: Mapping from register name to bit values
        binst_graph: The cbloq's binst graph.
    """

    # Keep track of each soquet's bit array.
    layers = []
    assign = {}
    alloc = Allocator()

    # LeftDangle assignment
    for reg in registers.lefts():
        if reg.wireshape:
            for idx in reg.wire_idxs():
                soq = Soquet(LeftDangle, reg, idx=idx)
                assign[soq] = alloc.alloc()
        else:
            soq = Soquet(LeftDangle, reg)
            assign[soq] = alloc.alloc()

    # Bloq-by-bloq application
    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        layer = _binst_do_score(binst, pred_cxns, assign, alloc)
        layers.append(layer)

    # Track bloq-to-dangle name changes
    final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
    for cxn in final_preds:
        assign[cxn.right] = assign[cxn.left]
    return layers, assign


def draw(bloq: CompositeBloq):
    from matplotlib import pyplot as plt

    layers, assign = cbloq_musical_score(bloq.registers, bloq._binst_graph)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set_xlim((-2, len(bloq.bloq_instances) + 1))
    ax.set_ylim((min(assign.values()) - 1, max(assign.values()) + 1))

    for soq in bloq.all_soquets:
        y = assign[soq]
        if soq.binst is LeftDangle:
            x = -1
        elif soq.binst is RightDangle:
            x = len(bloq.bloq_instances)
        else:
            x = soq.binst.i

        if soq.binst is LeftDangle or soq.binst is RightDangle:
            bbox = dict(fc='white', ec='none')
        elif soq.reg.side is Side.LEFT:
            bbox = dict(fc='white', boxstyle='RArrow')
        elif soq.reg.side is Side.RIGHT:
            bbox = dict(fc='white', boxstyle='LArrow')
        else:
            bbox = dict(fc='white')
        ax.text(
            x,
            y,
            f'{soq.reg.name}',
            transform=ax.transData,
            fontsize=10,
            ha='center',
            va='center',
            bbox=bbox,
        )

    return fig, ax
