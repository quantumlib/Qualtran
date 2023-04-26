import json
from typing import Dict, Iterable, Tuple, Union

import attrs
import networkx as nx
import numpy as np
from attrs import frozen
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from cirq_qubitization.bloq_algos.and_bloq import And
from cirq_qubitization.bloq_algos.basic_gates import CNOT
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
from cirq_qubitization.quantum_graph.util_bloqs import Join, Split


@frozen
class QLine:
    y: int
    seq_x: int
    topo_gen: int


_IN_USE = set()


def _new_y():
    global _IN_USE
    i = 0
    while True:
        if i not in _IN_USE:
            _IN_USE.add(i)
            return i
        i += 1


def _new(reg: FancyRegister, seq_x: int, topo_gen: int) -> Union[QLine, NDArray[QLine]]:
    global _IN_USE
    if not reg.wireshape:
        return QLine(y=_new_y(), seq_x=seq_x, topo_gen=topo_gen)

    arg = np.zeros(reg.wireshape, dtype=object)
    for idx in reg.wire_idxs():
        arg[idx] = QLine(y=_new_y(), seq_x=seq_x, topo_gen=topo_gen)
    return arg


def _free(reg: FancyRegister, arr: Union[QLine, NDArray[QLine]]):
    global _IN_USE
    if not reg.wireshape:
        assert isinstance(arr, QLine), arr
        _IN_USE.remove(arr.y)
        return

    for qline in arr.reshape(-1):
        _IN_USE.remove(qline.y)


def _get_in_vals(
    binst: BloqInstance, reg: FancyRegister, soq_assign: Dict[Soquet, QLine]
) -> Union[QLine, NDArray[QLine]]:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.wireshape:
        return soq_assign[Soquet(binst, reg)]

    arg = np.empty(reg.wireshape, dtype=object)
    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_vals(
    regs: Iterable[FancyRegister],
    binst: BloqInstance,
    vals: Dict[str, QLine],
    soq_assign: Dict[Soquet, QLine],
    seq_x: int,
    topo_gen: int,
):
    """Update `soq_assign` using `vals`.

    This helper function is responsible for error checking. We use `regs` to make sure all the
    keys are present in the vals dictionary. We check the classical value shapes, types, and
    ranges.
    """
    for reg in regs:
        try:
            arr = vals[reg.name]
        except KeyError:
            arr = _new(reg=reg, seq_x=seq_x, topo_gen=topo_gen)

        if reg.wireshape:
            arr = np.asarray(arr)
            if arr.shape != reg.wireshape:
                raise ValueError(
                    f"Incorrect shape {arr.shape} received for {binst}.{reg.name}. "
                    f"Want {reg.wireshape}."
                )

            for idx in reg.wire_idxs():
                soq = Soquet(binst, reg, idx=idx)
                soq_assign[soq] = attrs.evolve(arr[idx], seq_x=seq_x, topo_gen=topo_gen)
        else:
            soq = Soquet(binst, reg)
            soq_assign[soq] = attrs.evolve(arr, seq_x=seq_x, topo_gen=topo_gen)


def _binst_assign_line(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, QLine],
    seq_x: int,
    topo_gen: int,
):
    """Call `on_classical_vals` on a given binst.

    Args:
        binst: The bloq instance whose bloq we will call `on_classical_vals`.
        pred_cxns: Predecessor connections for the bloq instance.
        soq_assign: Current assignment of soquets to classical values.
    """

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = attrs.evolve(soq_assign[cxn.left], seq_x=seq_x, topo_gen=topo_gen)

    def _in_vals(reg: FancyRegister):
        # close over binst and `soq_assign`
        return _get_in_vals(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}
    partial_out_vals = {
        reg.name: in_vals[reg.name] for reg in bloq.registers if reg.side is Side.THRU
    }


    _update_assign_from_vals(
        bloq.registers.rights(), binst, partial_out_vals, soq_assign, seq_x=seq_x, topo_gen=topo_gen
    )

    # free stuff
    for reg in bloq.registers:
        if not reg.side is Side.LEFT:
            continue
        _free(reg, in_vals[reg.name])


def _cbloq_musical_score(
    registers: FancyRegisters, vals: Dict[str, QLine], binst_graph: nx.DiGraph
) -> Tuple[Dict[str, QLine], Dict[Soquet, QLine]]:
    """Propagate `on_classical_vals` calls through a composite bloq's contents.

    While we're handling the plumbing, we also do error checking on the arguments; see
    `_update_assign_from_vals`.

    Args:
        registers: The cbloq's registers for validating inputs
        vals: Mapping from register name to classical values
        binst_graph: The cbloq's binst graph.

    Returns:
        final_vals: A mapping from register name to output classical values
        soq_assign: An assignment from each soquet to its classical value. Soquets
            corresponding to thru registers will be mapped to the *output* classical
            value.
    """
    global _IN_USE
    _IN_USE = set()
    # Keep track of each soquet's bit array. Initialize with LeftDangle
    soq_assign: Dict[Soquet, QLine] = {}
    _update_assign_from_vals(registers.lefts(), LeftDangle, vals, soq_assign, seq_x=-1, topo_gen=0)

    # Bloq-by-bloq application
    seq_x = 0
    for topo_gen, binsts in enumerate(nx.topological_generations(binst_graph)):
        for binst in binsts:
            if isinstance(binst, DanglingT):
                continue
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            _binst_assign_line(binst, pred_cxns, soq_assign, seq_x=seq_x, topo_gen=topo_gen)
            seq_x += 1

    # Track bloq-to-dangle name changes
    if len(list(registers.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = attrs.evolve(soq_assign[cxn.left], seq_x=seq_x, topo_gen=topo_gen)

    # Formulate output with expected API
    def _f_vals(reg: FancyRegister):
        return _get_in_vals(RightDangle, reg, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in registers.rights()}
    return final_vals, soq_assign


@frozen
class Symb:
    pass


@frozen
class TextBox(Symb):
    text: str

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=10,
            ha='center',
            va='center',
            bbox={'boxstyle': 'round', 'fc': 'white'},
        )


@frozen
class Text(Symb):
    text: str

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=10,
            ha='center',
            va='center',
            bbox={'lw': 0, 'fc': 'white'},
        )

@frozen
class RarrowTextBox(Symb):
    text:str

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=10,
            ha='center',
            va='center',
            bbox={'boxstyle': 'rarrow', 'fc': 'white'},
        )

@frozen
class LarrowTextBox(Symb):
    text:str

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=10,
            ha='center',
            va='center',
            bbox={'boxstyle': 'larrow', 'fc': 'white'},
        )


@frozen
class Circle(Symb):
    filled: bool = True

    def draw(self, ax, x, y):
        fc = 'k' if self.filled else 'w'
        c = plt.Circle((x, -y), radius=0.25, fc=fc, ec='k')
        ax.add_patch(c)


@frozen
class ModPlus(Symb):
    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            "⊕",
            transform=ax.transData,
            fontsize=20,
            ha='center',
            va='center',
            bbox={'fc': 'none', 'lw': 0},
        )


def _draw_soq(soq: Soquet, soq_assign, ax):
    y = soq_assign[soq].y
    if isinstance(soq.binst, DanglingT):
        return Text(soq.pretty())

    if isinstance(soq.binst.bloq, And) and soq.reg.name == 'ctrl':
        c_idx, = soq.idx
        filled = bool(soq.binst.bloq.cv1 if c_idx==0 else soq.binst.bloq.cv2)
        return Circle(filled)

    if isinstance(soq.binst.bloq, CNOT):
        if soq.reg.name == 'ctrl':
            return Circle()
        elif soq.reg.name == 'target':
            return ModPlus()
        else:
            raise AssertionError()

    text = soq.pretty()
    if isinstance(soq.binst.bloq, (Split,Join)) and soq.reg.wireshape:
        text = f'[{", ".join(str(i) for i in soq.idx)}]'
    if isinstance(soq.binst.bloq, And) and soq.reg.name == 'target':
        text = '∧'


    if soq.reg.side is Side.THRU:
        return TextBox(text)
    elif soq.reg.side is Side.LEFT:
        return RarrowTextBox(text)
    elif soq.reg.side is Side.RIGHT:
        return LarrowTextBox(text)


def draw(cb: CompositeBloq, soq_assign: Dict[Soquet, QLine]):
    max_i = max(binst.i for binst in cb.bloq_instances)
    max_y = max(v.y for v in soq_assign.values())

    fig, ax = plt.subplots(figsize=(max(5.0, 0.4 * max_i), 5))
    soqs = []
    vlines = []

    for binst in nx.topological_sort(cb._binst_graph):
        preds, succs = _binst_to_cxns(binst, binst_graph=cb._binst_graph)

        binst_top_y = 0
        binst_bot_y = max_y
        binst_x = None

        for pred in preds:
            me = pred.right
            symb = _draw_soq(me, soq_assign, ax)
            qline = soq_assign[me]

            if me.reg.side is Side.THRU:
                x = qline.seq_x
            elif me.reg.side is Side.LEFT:
                x = qline.seq_x #- 1/3
            else:
                raise AssertionError(me.reg)

            symb.draw(ax, x, qline.y)
            soqs.append({
                'symb': symb.__class__.__name__,
                'attrs': attrs.asdict(symb),
                'x': x,
                'y': qline.y
            })

            if qline.y < binst_bot_y:
                binst_bot_y = qline.y
            if qline.y > binst_top_y:
                binst_top_y = qline.y

            if binst_x is not None:
                assert binst_x == qline.seq_x
            else:
                binst_x = qline.seq_x

        for succ in succs:
            me = succ.left
            symb = _draw_soq(me, soq_assign, ax)
            qline = soq_assign[me]

            if isinstance(me.binst, DanglingT):
                # still need to draw dangles
                x = qline.seq_x
            elif me.reg.side is Side.THRU:
                # ALREADY DREW IT
                continue
            elif me.reg.side is Side.RIGHT:
                x = qline.seq_x# + 1/3
            else:
                raise AssertionError(me.reg)

            symb.draw(ax, x, qline.y)
            soqs.append({
                'symb': symb.__class__.__name__,
                'attrs': attrs.asdict(symb),
                'x': x,
                'y': qline.y
            })

            if qline.y < binst_bot_y:
                binst_bot_y = qline.y
            if qline.y > binst_top_y:
                binst_top_y = qline.y

            if binst_x is not None:
                assert binst_x == qline.seq_x
            else:
                binst_x = qline.seq_x

        if not isinstance(binst, DanglingT):
            ax.vlines(binst_x, -binst_top_y, -binst_bot_y, color='k', zorder=-1)
            vlines.append({
                'x': binst_x,
                'top_y': binst_top_y,
                'bot_y': binst_bot_y,
            })

    ax.set_xlim((-2, max_i + 1))
    ax.set_ylim((-max_y - 1, 0))
    ax.axis('equal')
    fig.tight_layout()
    with open('unary.json', 'w') as f:
        json.dump({'soqs': soqs, 'vlines': vlines}, f)
