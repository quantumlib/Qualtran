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

"""Tools for laying out composite bloq graphs onto a "musical score".

A musical score is one where time proceeds from left to right and each horizontal line
represents a qubit or register of qubits.
"""

import abc
import heapq
import json
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Set, Tuple, Union

import attrs
import networkx as nx
import numpy as np
from attrs import frozen, mutable
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    Register,
    RightDangle,
    Side,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns


@frozen
class RegPosition:
    """Coordinates for a register when visualizing on a musical score.

    Throughout, we consider two different "x" (i.e. time) coordinates: a sequential one
    where each bloq is in its own column and a topological one where bloqs that are
    topologically independent can share a time slice.

    Attributes:
        y: The y (vertical) position as an integer.
        seq_x: The x (horizontal) position where each bloq is enumerated in sequence.
        topo_gen: The index of the topological generation to which the bloq belongs.
    """

    y: int
    seq_x: int
    topo_gen: int

    def json_dict(self):
        return attrs.asdict(self)


@frozen(order=True)
class HLine:
    """Dataclass representing a horizontal line segment at a given vertical position `x`.

    It runs from (sequential) x positions `seq_x_start` to `seq_x_end`, inclusive. If `seq_x_end`
    is `None`, that indicates we've started a line (by allocating a new qubit perhaps) but
    we don't know where it ends yet.
    """

    y: int
    seq_x_start: int
    seq_x_end: Optional[int] = None

    def json_dict(self):
        return attrs.asdict(self)


class LineManager:
    """Methods to manage allocation and de-allocation of lines representing a register of qubits."""

    def __init__(self, max_n_lines: int = 100):
        self.available = list(range(max_n_lines))
        heapq.heapify(self.available)
        self.hlines: Set[HLine] = set()
        self._reserved: List[Tuple[List[int], Callable]] = []

    def new_y(self, binst: BloqInstance, reg: Register, idx=None):
        """Allocate a new y position (i.e. a new qubit or register)."""
        return heapq.heappop(self.available)

    def reserve_n(self, n: int, until):
        """Reserve `n` lines until further notice.

        To have fine-grained control over the vertical layout of HLines, consider
        overriding `maybe_reserve` which can call this method to reserve lines
        depending on the musical score context.
        """
        nums = []
        for _ in range(n):
            nums.append(heapq.heappop(self.available))
        self._reserved.append((nums, until))

    def unreserve(self, binst: BloqInstance, reg: Register):
        """Go through our reservations and rescind them depending on the `until` predicate."""
        kept = []
        for ys, until in self._reserved:
            if until(binst, reg):
                for y in ys:
                    heapq.heappush(self.available, y)
            else:
                kept.append((ys, until))
        self._reserved = kept

    def maybe_reserve(
        self, binst: Union[DanglingT, BloqInstance], reg: Register, idx: Tuple[int, ...]
    ):
        """Override this method to provide custom control over line allocation.

        After a new y position is allocated and after a y position is freed, this method
        is called  with the current `binst, reg, idx`. You can inspect these elements to
        determine whether you want to continue allocating lines first-come-first-serve by
        returning without doing anything;
        or you can call `self.reserve_n(n, until)` to keep the next `n` lines unavailable
        until the `until` callback predicate evaluates to True.

        Whenever a new register is encountered, we first go through existing reservations
        and call the `until` predicate on `binst, reg`.
        """

    def new(
        self, binst: BloqInstance, reg: Register, seq_x: int, topo_gen: int
    ) -> Union[RegPosition, NDArray[RegPosition]]:  # type: ignore[type-var]
        """Allocate a position or positions for `reg`.

        `binst` and `reg` can optionally modify the allocation strategy.
        `seq_x` and `topo_gen` are passed through.
        """
        self.unreserve(binst, reg)
        if not reg.shape:
            y = self.new_y(binst, reg)
            self.hlines.add(HLine(y=y, seq_x_start=seq_x))
            self.maybe_reserve(binst, reg, idx=tuple())
            return RegPosition(y=y, seq_x=seq_x, topo_gen=topo_gen)

        arg = np.zeros(reg.shape, dtype=object)
        for idx in reg.all_idxs():
            y = self.new_y(binst, reg, idx)
            self.hlines.add(HLine(y=y, seq_x_start=seq_x))
            arg[idx] = RegPosition(y=y, seq_x=seq_x, topo_gen=topo_gen)
            self.maybe_reserve(binst, reg, idx)
        return arg

    def finish_hline(self, y: int, seq_x_end: int):
        """Update `self.hlines` once we know where an HLine ends."""
        (partial_h_line,) = (h for h in self.hlines if h.y == y and h.seq_x_end is None)
        self.hlines.remove(partial_h_line)
        self.hlines.add(attrs.evolve(partial_h_line, seq_x_end=seq_x_end))

    def free(
        self,
        binst: Union[DanglingT, BloqInstance],
        reg: Register,
        arr: Union[RegPosition, NDArray[RegPosition]],
    ):
        """De-allocate a position or positions for `reg`.

        This will free the position for future allocation. This will find the in-progress
        HLine associate with `reg` and update it to indicate the end point.
        """
        if not reg.shape:
            qline = arr
            assert isinstance(qline, RegPosition), qline
            heapq.heappush(self.available, qline.y)
            self.finish_hline(qline.y, seq_x_end=qline.seq_x)
            self.maybe_reserve(binst, reg, idx=tuple())
            return
        assert isinstance(arr, np.ndarray)
        for idx in reg.all_idxs():
            qline = arr[idx]
            assert isinstance(qline, RegPosition)
            heapq.heappush(self.available, qline.y)
            self.finish_hline(qline.y, seq_x_end=qline.seq_x)
            self.maybe_reserve(binst, reg, idx)


def _get_in_vals(
    binst: Union[DanglingT, BloqInstance], reg: Register, soq_assign: Dict[Soquet, RegPosition]
) -> Union[RegPosition, NDArray[RegPosition]]:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.shape:
        return soq_assign[Soquet(binst, reg)]

    arg = np.empty(reg.shape, dtype=object)
    for idx in reg.all_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_vals(
    regs: Iterable[Register],
    binst: Union[DanglingT, BloqInstance],
    vals: Dict[str, RegPosition],
    soq_assign: Dict[Soquet, RegPosition],
    seq_x: int,
    topo_gen: int,
    manager: LineManager,
):
    """Update `soq_assign` using `vals`.

    If a given register is not in the `vals` dictionary, we will allocate a new position for it.
    This helper function does some shape-compatibility checking.
    """
    for reg in regs:
        try:
            arr: Union[RegPosition, NDArray[RegPosition]] = vals[reg.name]
        except KeyError:
            arr = manager.new(
                binst=cast(BloqInstance, binst), reg=reg, seq_x=seq_x, topo_gen=topo_gen
            )

        if reg.shape:
            arr = np.asarray(arr)
            if arr.shape != reg.shape:
                raise ValueError(
                    f"Incorrect shape {arr.shape} received for {binst}.{reg.name}. "
                    f"Want {reg.shape}."
                )

            for idx in reg.all_idxs():
                soq = Soquet(binst, reg, idx=idx)
                soq_assign[soq] = attrs.evolve(arr[idx], seq_x=seq_x, topo_gen=topo_gen)
        else:
            soq = Soquet(binst, reg)
            assert isinstance(arr, RegPosition)
            soq_assign[soq] = attrs.evolve(arr, seq_x=seq_x, topo_gen=topo_gen)


def _binst_assign_line(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, RegPosition],
    seq_x: int,
    topo_gen: int,
    manager: LineManager,
):
    """Assign positions for a binst.

    Args:
        binst: The bloq instance whose bloq we will call `on_classical_vals`.
        pred_cxns: Predecessor connections for the bloq instance.
        soq_assign: Current assignment of soquets to classical values.
        seq_x: The sequential x index of the binst.
        topo_gen: The topological generation of the binst.
        manager: The LineManager.
    """

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = attrs.evolve(soq_assign[cxn.left], seq_x=seq_x, topo_gen=topo_gen)

    def _in_vals(reg: Register):
        # close over binst and `soq_assign`
        return _get_in_vals(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}
    partial_out_vals = {
        reg.name: in_vals[reg.name] for reg in bloq.signature if reg.side is Side.THRU
    }

    # The following will use `partial_out_vals` to re-use existing THRU lines; otherwise
    # the following will allocate new lines.
    _update_assign_from_vals(
        bloq.signature.rights(),
        binst,
        partial_out_vals,
        soq_assign,
        seq_x=seq_x,
        topo_gen=topo_gen,
        manager=manager,
    )

    # Free any purely-left registers.
    for reg in bloq.signature:
        if reg.side is Side.LEFT:
            manager.free(binst, reg, in_vals[reg.name])


def _cbloq_musical_score(
    signature: Signature, binst_graph: nx.DiGraph, manager: Optional[LineManager] = None
) -> Tuple[Dict[str, RegPosition], Dict[Soquet, RegPosition], LineManager]:
    """Assign musical score positions through a composite bloq's contents.

    Args:
        signature: The cbloq's signature.
        binst_graph: The cbloq's binst graph.

    Returns:
        final_vals: A mapping from register name to output positions
        soq_assign: An assignment from each soquet to its position
        manager: The line manager (now containing the final `hlines` collection).
    """
    if manager is None:
        manager = LineManager()

    # Keep track of each soquet's position. Initialize by implicitly allocating new positions.
    # We introduce the convention that `LeftDangle`s are a seq_x=-1 and topo_gen=0
    soq_assign: Dict[Soquet, RegPosition] = {}
    topo_gen = 0
    _update_assign_from_vals(
        signature.lefts(), LeftDangle, {}, soq_assign, seq_x=-1, topo_gen=topo_gen, manager=manager
    )

    # Bloq-by-bloq application
    seq_x = 0
    for topo_gen, binsts in enumerate(nx.topological_generations(binst_graph)):
        for binst in binsts:
            if isinstance(binst, DanglingT):
                continue
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            _binst_assign_line(
                binst, pred_cxns, soq_assign, seq_x=seq_x, topo_gen=topo_gen, manager=manager
            )
            seq_x += 1

    # Track bloq-to-dangle name changes
    if len(list(signature.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = attrs.evolve(
                soq_assign[cxn.left], seq_x=seq_x, topo_gen=topo_gen
            )

    # Formulate output with expected API
    def _f_vals(reg: Register):
        return _get_in_vals(RightDangle, reg, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in signature.rights()}
    for reg in signature.rights():
        manager.free(RightDangle, reg, final_vals[reg.name])
    return final_vals, soq_assign, manager


@frozen
class WireSymbol(metaclass=abc.ABCMeta):
    """Base class for a symbol.

    A symbol is a particular visual representation of a bloq's register.
    """

    @abc.abstractmethod
    def draw(self, ax, x, y) -> None:
        """Draw this symbol using matplotlib."""

    def adjoint(self) -> 'WireSymbol':
        """Return a symbol that is the adjoint of this."""
        return self

    def json_dict(self) -> Dict[str, Any]:
        return {'symb_cls': self.__class__.__name__, 'symb_attributes': attrs.asdict(self)}


def _text_adjoint(text: str) -> str:
    """Add / Remove a dagger from the end of the text."""
    return text.strip('†') if text.endswith('†') else text + '†'


@frozen
class TextBox(WireSymbol):
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

    def adjoint(self) -> 'TextBox':
        return TextBox(_text_adjoint(self.text))


@frozen
class Text(WireSymbol):
    text: str
    fontsize: int = 10

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=self.fontsize,
            ha='center',
            va='center',
            bbox={'lw': 0, 'fc': 'white'},
        )

    def adjoint(self) -> 'Text':
        return Text(_text_adjoint(self.text), self.fontsize)


@frozen
class RarrowTextBox(WireSymbol):
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
            bbox={'boxstyle': 'rarrow', 'fc': 'white'},
        )

    def adjoint(self):
        return LarrowTextBox(text=self.text)


@frozen
class LarrowTextBox(WireSymbol):
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
            bbox={'boxstyle': 'larrow', 'fc': 'white'},
        )

    def adjoint(self) -> 'WireSymbol':
        return RarrowTextBox(text=self.text)


@frozen
class Circle(WireSymbol):
    filled: bool = True

    def draw(self, ax, x, y):
        fc = 'k' if self.filled else 'w'
        c = plt.Circle((x, -y), radius=0.1, fc=fc, ec='k')
        ax.add_patch(c)


@frozen
class ModPlus(WireSymbol):
    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            "⊕",
            transform=ax.transData,
            fontsize=18,
            ha='center',
            va='center',
            bbox={'fc': 'none', 'lw': 0},
        )


def directional_text_box(text: str, side: Side) -> WireSymbol:
    if side is Side.THRU:
        return TextBox(text)
    elif side is Side.LEFT:
        return RarrowTextBox(text)
    elif side is Side.RIGHT:
        return LarrowTextBox(text)
    raise ValueError(f"Unknown side: {side}")


def _soq_to_symb(soq: Soquet) -> WireSymbol:
    """Return a pleasing symbol for the given soquet."""

    # Use text (with no box) for dangling register identifiers.
    if isinstance(soq.binst, DanglingT):
        return Text(soq.pretty() + f'/{soq.reg.dtype}', fontsize=8)

    # Otherwise, use `Bloq.wire_symbol`.
    return soq.binst.bloq.wire_symbol(soq.reg, soq.idx)


@mutable
class SoqData:
    """Data needed to draw a soquet.

    The symbol `symb` and position `RegPosition`. This also includes a string
    `ident` which can be used as a d3.js "key" to associate objects when transitioning
    between two musical scores.
    """

    symb: WireSymbol
    rpos: RegPosition
    ident: str

    def json_dict(self):
        d = self.symb.json_dict()
        d |= self.rpos.json_dict()
        d['ident'] = self.ident
        return d


@frozen
class VLine:
    """Data for drawing vertical lines."""

    x: int
    top_y: int
    bottom_y: int
    label: Text

    def json_dict(self):
        return attrs.asdict(self)


@mutable
class MusicalScoreData:
    """All the data required to draw a musical score.

    This can be passed to `draw_musical_score` which will use matplotlib
    to draw the entities or dumped to json with `dump_musical_score` and then
    loaded by the d3.js visualization code.
    """

    max_x: int
    max_y: int
    soqs: List[SoqData]
    hlines: List[HLine]
    vlines: List[VLine]

    def json_dict(self):
        return attrs.asdict(self, recurse=False)


def _make_ident(binst: BloqInstance, me: Soquet):
    """Make a unique string identifier key for a soquet."""
    soqi = f'{me.reg.name},{me.reg.side},{me.idx}'
    if isinstance(binst, DanglingT):
        sidestr = 'l' if binst is LeftDangle else 'r'
        return f'dang,{me.reg.name},{sidestr},{me.idx}'

    return f'{binst.i},{soqi}'


def get_musical_score_data(bloq: Bloq, manager: Optional[LineManager] = None) -> MusicalScoreData:
    """Get the musical score data for a (composite) bloq.

    This will first walk through the compute graph to assign each soquet
    to a register position. Then we iterate again to finalize drawing-relevant
    properties like symbols and the various horizontal and vertical lines.

    Args:
        bloq: The bloq or composite bloq to get drawing data for
        manager: Optionally provide an override of `LineManager` if you want
            to control the allocation of horizontal (register) lines.
    """
    cbloq = bloq.as_composite_bloq()

    _, soq_assign, manager = _cbloq_musical_score(
        cbloq.signature, binst_graph=cbloq._binst_graph, manager=manager
    )
    msd = MusicalScoreData(
        max_x=max(v.seq_x for v in soq_assign.values()),
        max_y=max(v.y for v in soq_assign.values()),
        soqs=[],
        vlines=[],
        hlines=sorted(manager.hlines),
    )

    for hline in manager.hlines:
        if hline.seq_x_end is None:
            raise ValueError(f"A horizontal line has no end: {hline}")

    for binst in nx.topological_sort(cbloq._binst_graph):
        preds, succs = _binst_to_cxns(binst, binst_graph=cbloq._binst_graph)

        # Keep track of the extent of our vlines
        binst_top_y = 0
        binst_bot_y = msd.max_y
        binst_x = None

        for pred in preds:
            me = pred.right
            symb = _soq_to_symb(me)
            rpos = soq_assign[me]
            ident = _make_ident(binst, me)

            msd.soqs.append(SoqData(symb=symb, rpos=rpos, ident=ident))

            if rpos.y < binst_bot_y:
                binst_bot_y = rpos.y
            if rpos.y > binst_top_y:
                binst_top_y = rpos.y

            if binst_x is not None:
                assert binst_x == rpos.seq_x
            else:
                binst_x = rpos.seq_x

        for succ in succs:
            me = succ.left
            symb = _soq_to_symb(me)
            rpos = soq_assign[me]

            if me.reg.side is Side.THRU and binst is not LeftDangle:
                # Already drew is as part of the preds
                continue

            ident = _make_ident(binst, me)
            msd.soqs.append(SoqData(symb=symb, rpos=rpos, ident=ident))

            if rpos.y < binst_bot_y:
                binst_bot_y = rpos.y
            if rpos.y > binst_top_y:
                binst_top_y = rpos.y

            if binst_x is not None:
                assert binst_x == rpos.seq_x
            else:
                binst_x = rpos.seq_x

        if not isinstance(binst, DanglingT):
            if binst_x is None:
                # No predecessors or successors
                continue
            msd.vlines.append(
                VLine(
                    x=binst_x,
                    top_y=binst_top_y,
                    bottom_y=binst_bot_y,
                    label=binst.bloq.wire_symbol(reg=None),
                )
            )

    return msd


def draw_musical_score(
    msd: MusicalScoreData,
    unit_to_inches: float = 0.8,
    max_width: float = 10.0,
    max_height: float = 8.0,
):
    # First, set up data coordinate limits and figure size.
    # X coordinates go from -1 to max_x
    #    with 1 unit of padding it goes from -2 to max_x+1
    xlim = (-2, msd.max_x + 1)
    x_extent = msd.max_x + 3.0
    # Y coordinates of non-labels goes from 0 to -max_y;
    #     with the bloq label above it goes from 0.5 to -max_y
    #     with 0.5 units of padding it goes from 1 to -(max_y+0.5)
    ylim = (-msd.max_y - 0.5, 1)
    y_extent = msd.max_y + 1.5

    # The width and height are proportional.
    width = unit_to_inches * x_extent
    height = unit_to_inches * y_extent

    # But we cap width and height (but keep it proportional).
    if width > height and width > max_width:
        scale = max_width / width
        width *= scale
        height *= scale
    elif height > max_height:
        scale = max_height / height
        height *= scale
        width *= scale

    fig, ax = plt.subplots(figsize=(width, height))

    for hline in msd.hlines:
        assert hline.seq_x_end is not None, hline
        ax.hlines(-hline.y, hline.seq_x_start, hline.seq_x_end, color='k', zorder=-1)

    for vline in msd.vlines:
        ax.vlines(vline.x, -vline.top_y, -vline.bottom_y, color='k', zorder=-1)
        if vline.label.text:
            vline.label.draw(ax, vline.x, vline.bottom_y - 0.5)

    for soq in msd.soqs:
        symb = soq.symb
        symb.draw(ax, soq.rpos.seq_x, soq.rpos.y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    fig.tight_layout()
    return fig, ax


class MusicalScoreEncoder(json.JSONEncoder):
    """An encoder that handles `MusicalScoreData` classes and those of its contents."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (SoqData, HLine, VLine, MusicalScoreData, WireSymbol, RegPosition)):
            return o.json_dict()

        return super().default(o)


def dump_musical_score(msd: MusicalScoreData, name: str):
    with open(f'{name}.json', 'w') as f:
        json.dump(msd, f, indent=2, cls=MusicalScoreEncoder)
