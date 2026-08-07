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
import functools
import heapq
import json
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Self,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    Union,
)

import attrs
import matplotlib.patches
import matplotlib.path as mplpath
import matplotlib.transforms
import networkx as nx
import numpy as np
from attrs import frozen, mutable
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqInstance,
    CDType,
    CompositeBloq,
    DanglingT,
    LeftDangle,
    QCDType,
    QDType,
    Register,
    RightDangle,
    Side,
    Signature,
)
from qualtran._infra.binst_graph_iterators import greedy_topological_sort
from qualtran._infra.composite_bloq import _binst_to_cxns
from qualtran._infra.quantum_graph import _Soquet

HLineID: TypeAlias = str


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


class HLineFlavor(Enum):
    """Horizonal lines can represent quantum or classical data."""

    QUANTUM = 1
    CLASSICAL = 2

    @classmethod
    def from_qcdtype(cls, qcdtype: QCDType) -> 'HLineFlavor':
        if isinstance(qcdtype, QDType):
            return cls.QUANTUM
        if isinstance(qcdtype, CDType):
            return cls.CLASSICAL

        # Fallback
        return cls.QUANTUM


@frozen(order=True)
class HLine:
    """Dataclass representing a horizontal line segment at a given vertical position `x`.

    It runs from (sequential) x positions `seq_x_start` to `seq_x_end`, inclusive. If `seq_x_end`
    is `None`, that indicates we've started a line (by allocating a new qubit perhaps) but
    we don't know where it ends yet.

    The horizontal line can be of a particular `flavor`, e.g. a quantum wire or a classical wire.
    """

    y: int
    seq_x_start: int
    seq_x_end: Optional[int] = None
    flavor: HLineFlavor = HLineFlavor.QUANTUM

    def json_dict(self) -> Dict[str, Any]:
        d = attrs.asdict(self)
        d['flavor'] = str(d['flavor'])
        return d


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
        flavor = HLineFlavor.from_qcdtype(reg.dtype)
        if not reg.shape:
            y = self.new_y(binst, reg)
            self.hlines.add(HLine(y=y, seq_x_start=seq_x, flavor=flavor))
            self.maybe_reserve(binst, reg, idx=tuple())
            return RegPosition(y=y, seq_x=seq_x, topo_gen=topo_gen)

        arg = np.zeros(reg.shape, dtype=object)
        for idx in reg.all_idxs():
            y = self.new_y(binst, reg, idx)
            self.hlines.add(HLine(y=y, seq_x_start=seq_x, flavor=flavor))
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
        arr: Union[RegPosition, NDArray[RegPosition]],  # type: ignore[type-var]
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
    binst: Union[DanglingT, BloqInstance], reg: Register, soq_assign: Dict[_Soquet, Any]
) -> Any:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.shape:
        return soq_assign[_Soquet(binst, reg)]

    arg = np.empty(reg.shape, dtype=object)
    for idx in reg.all_idxs():
        soq = _Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


class _MusicalScoreLayoutBuilder:
    """For a given CompositeBloq compute graph, assign abstract geometric properties.

    If you are not interested in customizing the layout, consider using the free-function
    `get_musical_score_data()` which is a wrapper around the `do_layout` method of this class.

    Otherwise, you can follow the layout steps shown in `do_layout`.

     0. Construct a layout builder with the `MusicalScoreLayoutBuilder.from_cbloq(cbloq)` factory
        class method.
     1. Call the `.do_horizontal_layout()` method to walk the compute graph and assign extents and
        identifiers to each horizontal line; and assign soquets to horizontal lines.
     2. Either: call the `.do_vertical_layout()` method or provide your own vertical layout
        to order the horizontal lines.
     3. Call `.finalize_aboslute_layout()` with the ordered horizontal line identifiers from
        the previous step.

    This process will get you a MusicalScoreData object, which can be plotted with
    `draw_musical_score`.

    The vertical layout step has the most impact on the visual display of the circuit diagram.
    The automated method takes an optional `LineManager`, which can be subclassed to provide
    custom vertical layout behavior. In practice, it is often difficult to get the desired
    result with the LineManager API. An alternative is to manually order the horizontal line
    identifiers generated during `.do_horizontal_layout()`.
     - The `print_hline_extents()` method will print the identifiers and their start and end
       coordinates. You can try to use this information to manually order the identifiers.
     - From a Jupyter notebook, the `draw_hline_labels` method will use Graphviz to layout
       a fluid graphlike view of the circuit where each edge is labeled with its horizontal
       line identifier from this class. This can give you a visual idea of which hline_id
       is which part of your circuit.
    """

    def __init__(
        self, ssa_names: Dict[_Soquet, str], signature: Signature, binst_graph: nx.DiGraph
    ):

        # The compute graph
        self._ssa_names = ssa_names
        self._signature = signature
        self._binst_graph = binst_graph
        self._binst_iter = greedy_topological_sort(self._binst_graph)

        # Goal 1: map each soquet to an abstract horizontal line (by string identifier)
        self._hline_ids: Set[HLineID] = set()
        self.hline_id_map: Dict[_Soquet, HLineID] = {}
        self.x_coord_map: Dict[_Soquet, int] = {}

        # Goal 2: keep track of where horizontal lines begin and end.
        self.hline_start_coords: Dict[HLineID, int] = {}
        self.hline_end_coords: Dict[HLineID, int] = {}
        self.hline_flavors: Dict[HLineID, HLineFlavor] = {}
        # And for backwards compatibility with LineManager...
        self.hline_start_soqs: Dict[HLineID, _Soquet] = {}
        self.hline_end_soqs: Dict[HLineID, _Soquet] = {}

        # State of the iteration through the compute graph
        self._binst: BloqInstance = LeftDangle
        self._previous_binst: Optional[BloqInstance] = None
        self._x_coord: int = -1

        # Initialize mappings
        self._update(LeftDangle, signature.lefts(), {})
        self._x_coord: int = 0

    @classmethod
    def from_cbloq(cls, cbloq: 'CompositeBloq') -> '_MusicalScoreLayoutBuilder':
        """Initiate a musical score layout builder from a CompositeBloq.

        Args:
            cbloq: The composite bloq

        Returns:
            A new layout builder.
        """
        return cls(ssa_names={}, signature=cbloq.signature, binst_graph=cbloq._binst_graph)

    def _assign_soq_attributes(self, soq: _Soquet, *, hline_id: HLineID, x_coord: int) -> None:
        self.hline_id_map[soq] = hline_id
        self.x_coord_map[soq] = x_coord

    def _get_unique_hline_id(self, prefix: str) -> str:
        i = 0
        attempt = prefix
        while True:
            if attempt not in self._hline_ids:
                self._hline_ids.add(attempt)
                return attempt
            i += 1
            attempt = f'{prefix}{i}'

    def _alloc(self, binst: BloqInstance, reg: Register) -> Union[HLineID, NDArray[HLineID]]:
        """Helper for recording the start of a new horizontal line from a given register."""
        flavor = HLineFlavor.from_qcdtype(reg.dtype)
        if reg.shape:
            arr = np.zeros(reg.shape, dtype=object)
            for idx in reg.all_idxs():
                soq = _Soquet(binst, reg, idx)
                if soq in self._ssa_names:
                    prefix = self._ssa_names[soq]
                else:
                    prefix = reg.name
                hline_id = self._get_unique_hline_id(prefix=prefix)
                arr[idx] = hline_id
                self.hline_start_coords[hline_id] = self._x_coord
                self.hline_flavors[hline_id] = flavor
                self.hline_start_soqs[hline_id] = soq
            return arr

        soq = _Soquet(binst, reg)
        if soq in self._ssa_names:
            prefix = self._ssa_names[soq]
        else:
            prefix = reg.name
        hline_id = self._get_unique_hline_id(prefix=prefix)
        self.hline_start_coords[hline_id] = self._x_coord
        self.hline_flavors[hline_id] = flavor
        self.hline_start_soqs[hline_id] = _Soquet(binst, reg)
        return hline_id

    def _free(
        self, binst: BloqInstance, reg: Register, arr: Union[HLineID, NDArray[HLineID]]
    ) -> None:
        """Helper for recording the end of a horizontal line."""
        if reg.shape:
            for idx in reg.all_idxs():
                hline_id = arr[idx]
                self.hline_end_coords[hline_id] = self._x_coord
                self.hline_end_soqs[hline_id] = _Soquet(binst, reg, idx)
            return

        hline_id = arr
        self.hline_end_coords[hline_id] = self._x_coord
        self.hline_end_soqs[hline_id] = _Soquet(binst, reg)

    def _update(
        self,
        binst: Union[BloqInstance, DanglingT],
        regs: Iterable[Register],
        vals: Dict[str, HLineID],
    ):
        """Update our mappings using `vals`.

        If a given register is not in the `vals` dictionary, we will allocate a new hline for it.
        This helper function does some shape-compatibility checking.
        """
        for reg in regs:
            debug_str = f'{binst}.{reg.name}'
            try:
                arr: Union[HLineID, NDArray[HLineID]] = vals[reg.name]
            except KeyError:
                arr = self._alloc(binst, reg)

            if reg.shape:
                arr = np.asarray(arr)
                if arr.shape != reg.shape:
                    raise ValueError(
                        f"Incorrect shape {arr.shape} received for {debug_str}. "
                        f"Want {reg.shape}."
                    )

                for idx in reg.all_idxs():
                    self._assign_soq_attributes(
                        _Soquet(binst, reg, idx=idx), hline_id=arr[idx], x_coord=self._x_coord
                    )
            else:
                self._assign_soq_attributes(
                    _Soquet(binst, reg), hline_id=arr, x_coord=self._x_coord
                )

    def step(self) -> Self:
        """Assign positions for the next binst.

        This advances our internal iteration by one step. Consider using `do_horizontal_layout()`
        to do all steps.
        """
        self._previous_binst = self._binst
        self._binst = next(self._binst_iter)
        if isinstance(self._binst, DanglingT):
            return self

        pred_cxns, succ_cxns = _binst_to_cxns(self._binst, binst_graph=self._binst_graph)

        # Track inter-Bloq name changes
        for cxn in pred_cxns:
            self._assign_soq_attributes(
                cxn.right, hline_id=self.hline_id_map[cxn.left], x_coord=self._x_coord
            )

        def _in_vals(reg: Register):
            # close over binst and `soq_assign`
            return _get_in_vals(self._binst, reg, soq_assign=self.hline_id_map)

        bloq = self._binst.bloq
        in_vals = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}
        partial_out_vals = {
            reg.name: in_vals[reg.name] for reg in bloq.signature if reg.side is Side.THRU
        }

        # The following will use `partial_out_vals` to re-use existing THRU lines; otherwise
        # the following will allocate new lines.
        self._update(self._binst, bloq.signature.rights(), partial_out_vals)

        # Free any purely-left registers.
        for reg in bloq.signature:
            if reg.side is Side.LEFT:
                self._free(self._binst, reg, in_vals[reg.name])

        self._x_coord += 1
        return self

    def finalize(self) -> Self:
        """Assign positions for RightDangle after stepping through all binsts.

        This should be called after `.step()` has been exhausted.
        """
        # Track bloq-to-dangle name changes
        if len(list(self._signature.rights())) > 0:
            final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=self._binst_graph)
            for cxn in final_preds:
                self._assign_soq_attributes(
                    cxn.right, hline_id=self.hline_id_map[cxn.left], x_coord=self._x_coord
                )

        # Final mappings
        def _f_vals(reg: Register):
            return _get_in_vals(RightDangle, reg, soq_assign=self.hline_id_map)

        final_vals = {reg.name: _f_vals(reg) for reg in self._signature.rights()}
        for reg in self._signature.rights():
            self._free(RightDangle, reg, final_vals[reg.name])

        return self

    def do_horizontal_layout(self) -> Self:
        """Do the horizontal layout of the circuit. This is the first step.

        This method
         - Assigns each soquet to an abstract horizontal line by string identifier
         - Positions each soquet along the x-axis.

        This is the first step in the overall `.do_layout()` method. The next step is to
        lay out the horizontal lines vertically (i.e. in a definite order).
        """
        try:
            while True:
                self.step()
        except StopIteration:
            return self.finalize()

    def print_hline_extents(self) -> None:
        for hline_id in self.hline_start_coords.keys():
            if hline_id in self.hline_end_coords:
                print(
                    f'{hline_id:10s} {self.hline_start_coords[hline_id]:4d} -> {self.hline_end_coords[hline_id]:4d}'
                )
            else:
                print(f'{hline_id:10s} {self.hline_start_coords[hline_id]:4d} -> ????')

    def draw_hline_labels(self, bloq):
        from qualtran.drawing.graphviz import EdgeLabeledGraphDrawer

        return EdgeLabeledGraphDrawer(bloq, self.hline_id_map).get_svg()

    def do_vertical_layout(
        self, manager: Optional[LineManager] = None
    ) -> List[Tuple[HLineID, ...]]:
        """Do the vertical layout of the circuit. This is the second step.

        This method greedily orders the horizontal lines. When a new horizontal line starts,
        it is added next. After a horizontal line ends, its y-coordinate is available for
        any following lines to take.

        You can manually order the hlines and skip this method.

        This returns a list of tuples of horizontal line identifiers. The list index is the
        y-coordinate and the tuple enumerates each of the horizontal lines at that y-coordinate.
        The return value should be passed into the next step, `finalize_absolute_layout()`.
        """
        y_mapping: Dict[HLineID, int] = {}
        if manager is None:
            manager = LineManager()

        for x in sorted(set(self.x_coord_map.values())):
            for hline_id, start_x in self.hline_start_coords.items():
                if start_x != x:
                    continue

                # line is starting now, use `manager`.
                soq = self.hline_start_soqs[hline_id]
                y = manager.new_y(soq.binst, soq.reg, soq.idx)
                y_mapping[hline_id] = y
                manager.maybe_reserve(soq.binst, soq.reg, soq.idx)

            for hline_id, end_x in self.hline_end_coords.items():
                if end_x != x:
                    continue

                # line is ending now, use `manager`.
                soq = self.hline_end_soqs[hline_id]
                heapq.heappush(manager.available, y_mapping[hline_id])
                manager.maybe_reserve(soq.binst, soq.reg, soq.idx)

        grouped_hline_ids = defaultdict(list)
        for hline_id, y in y_mapping.items():
            grouped_hline_ids[y].append(hline_id)
        ordered_hline_ids = [
            tuple(grouped_hline_ids.get(i, ())) for i in range(0, max(grouped_hline_ids.keys(), default=0) + 1)
        ]
        return ordered_hline_ids

    def finalize_aboslute_layout(
        self,
        ordered_hline_ids: Sequence[Union[HLineID, Tuple[HLineID, ...]]],
        labeller: Optional['BinstLabeller'] = None,
    ) -> 'MusicalScoreData':
        """Finalize the layout. This is the final step.

        Args:
             ordered_hline_ids: The horizontal line identifiers in the order they will
                appear vertically. This can be automatically generated with the
                `do_vertical_layout()` method or provided manually. Each entry in the list
                specifies the hline or hlines that are positioned at that y-coordinate.

        Returns:
            msd: The final `MusicalScoreData` object that can be drawn with various backends,
                including `draw_musical_score(msd)`.

        """
        hline_id_to_y: Dict[HLineID, int] = {}
        for i, hlids in enumerate(ordered_hline_ids):
            if isinstance(hlids, str):
                hlid = hlids
                hline_id_to_y[hlid] = i
            elif isinstance(hlids, tuple):
                for hlid in hlids:
                    hline_id_to_y[hlid] = i
            else:
                raise ValueError(
                    f"Unknown hline_id {hlids}. Must be a string id or a tuple of string ids."
                )

        hlines = [
            HLine(
                y=hline_id_to_y[hline_id],
                seq_x_start=self.hline_start_coords[hline_id],
                seq_x_end=self.hline_end_coords[hline_id],
                flavor=self.hline_flavors[hline_id],
            )
            for hline_id in self.hline_start_coords
        ]

        soq_assign = {
            soq: RegPosition(
                y=hline_id_to_y[self.hline_id_map[soq]], seq_x=self.x_coord_map[soq], topo_gen=None
            )
            for soq in self.hline_id_map.keys()
        }

        if labeller is None:
            labeller = BinstLabeller()

        msd = _finalize_line_extents(self._binst_graph, soq_assign, hlines, labeller=labeller)
        return msd

    def do_layout(
        self, manager: Optional[LineManager] = None, labeller: Optional['BinstLabeller'] = None
    ) -> 'MusicalScoreData':
        """Build a full layout for a circuit.

        This does each of the three steps. If you want to provide your own hline ordering,
        please call the methods for each of the steps yourself.

        Consider using the `get_musical_score_data` top-level function if you don't need
        layout customization.
        """
        self.do_horizontal_layout()
        ordered_hline_ids = self.do_vertical_layout(manager=manager)
        return self.finalize_aboslute_layout(ordered_hline_ids=ordered_hline_ids, labeller=labeller)


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


# Define a custom BoxStyle class
class PentagonBoxStyle:
    def __init__(self, *, pad=0.3, lr='l', **kwargs):
        self.pad = pad
        self.lr = lr
        if kwargs:
            print(f"Pentagon box style called with {kwargs}")
        pass

    def __call__(self, x0, y0, width, height, mutation_size):
        # Calculate padding
        pad = mutation_size * self.pad

        # Define vertices for a "house" pentagon
        # (x0, y0) is the bottom-left of the text bounding box
        left = x0 - pad
        right = x0 + width + pad
        bottom = y0 - pad
        top = y0 + height + pad

        # Calculate the "peak" of the pentagon
        mid_x = (left + right) / 2
        mid_y = (top + bottom) / 2

        if self.lr not in ['l', 'r']:
            raise ValueError(f"Unknown lr {self.lr}")

        # Define the path codes
        path_data = [(mplpath.Path.MOVETO, (left, bottom)), (mplpath.Path.LINETO, (right, bottom))]
        if self.lr == 'l':
            path_data += [(mplpath.Path.LINETO, (right, top))]
        else:
            peak_x = right + (height / 4)
            path_data += [
                (mplpath.Path.LINETO, (peak_x, mid_y)),
                (mplpath.Path.LINETO, (right, top)),
            ]

        path_data += [(mplpath.Path.LINETO, (left, top))]

        if self.lr == 'l':
            peak_x = left - (height / 4)
            path_data += [
                (mplpath.Path.LINETO, (peak_x, mid_y)),
                (mplpath.Path.CLOSEPOLY, (left, bottom)),
            ]
        else:
            path_data += [(mplpath.Path.CLOSEPOLY, (left, bottom))]

        # transform = matplotlib.transforms.Affine2D().rotate_deg_around(mid_x, mid_y, 90)

        codes, verts = zip(*path_data)
        return matplotlib.path.Path(verts, codes)


matplotlib.patches.BoxStyle._style_list["lpentagon"] = functools.partial(PentagonBoxStyle, lr='l')
matplotlib.patches.BoxStyle._style_list["rpentagon"] = functools.partial(PentagonBoxStyle, lr='r')


@frozen
class TextBox(WireSymbol):
    text: str

    def draw(self, ax, x, y):
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=11,
            ha='center',
            va='center',
            bbox={'boxstyle': 'square,pad=0.2', 'fc': 'white'},
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
            bbox={'boxstyle': 'rpentagon', 'fc': 'white'},
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
            bbox={'boxstyle': 'lpentagon', 'fc': 'white'},
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
        lw = matplotlib.rcParams.get('lines.linewidth', 1.0)
        radius_px = 8

        offset = matplotlib.transforms.ScaledTranslation(x, -y, ax.transData)
        xform = matplotlib.transforms.IdentityTransform() + offset

        # Note: We define them at (0, 0) because the transform handles the position
        circle = matplotlib.patches.Circle(
            (0, 0), radius_px, fill=False, edgecolor='k', lw=lw, transform=xform
        )

        line1 = plt.Line2D([-radius_px, radius_px], [0, 0], color='k', lw=lw, transform=xform)
        line2 = plt.Line2D([0, 0], [-radius_px, radius_px], color='k', lw=lw, transform=xform)

        ax.add_patch(circle)
        ax.add_artist(line1)
        ax.add_artist(line2)


def directional_text_box(text: str, side: Side) -> WireSymbol:
    if side is Side.THRU:
        return TextBox(text)
    elif side is Side.LEFT:
        return RarrowTextBox(text)
    elif side is Side.RIGHT:
        return LarrowTextBox(text)
    raise ValueError(f"Unknown side: {side}")


def _soq_to_symb(soq: _Soquet) -> WireSymbol:
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
    label: str

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


def _make_ident(binst: BloqInstance, me: _Soquet):
    """Make a unique string identifier key for a soquet."""
    soqi = f'{me.reg.name},{me.reg.side},{me.idx}'
    if isinstance(binst, DanglingT):
        sidestr = 'l' if binst is LeftDangle else 'r'
        return f'dang,{me.reg.name},{sidestr},{me.idx}'

    return f'{binst.i},{soqi}'


class BinstLabeller:
    def get_label(self, binst: BloqInstance) -> str:
        text_symb = binst.bloq.wire_symbol(reg=None)
        if not isinstance(text_symb, Text):
            raise ValueError(
                f"{binst.bloq} gave an invalid top-label: {text_symb}. "
                f"Should be an instance of qualtran.drawing.Text."
            )
        return text_symb.text

    def indexed_register_name(self, reg: Register, idx: Tuple[int, ...]) -> str:
        label = reg.name
        if len(idx) > 0:
            return f'{label}[{",".join(str(i) for i in idx)}]'
        return label

    def get_symbol(self, binst: BloqInstance, reg: Register, idx: Tuple[int, ...]) -> WireSymbol:
        if isinstance(binst, DanglingT):
            return Text(self.indexed_register_name(reg, idx) + f'/{reg.dtype}', fontsize=8)

        return binst.bloq.wire_symbol(reg, idx)


def _finalize_line_extents(
    binst_graph: nx.DiGraph,
    soq_assign: Dict[_Soquet, RegPosition],
    hlines: Sequence[HLine],
    labeller: BinstLabeller,
) -> MusicalScoreData:
    """Given a layout of soquets and hlines, determine the geometric details

     - Get the correct symbol for each soquet; de-duplicate THRU soquets.
     - Find the y-extents of the vertical lines connecting all a binst's soquets.

    This returns the complete `MusicalScoreData` object.
    """
    msd = MusicalScoreData(
        max_x=max((v.seq_x for v in soq_assign.values()), default=0),
        max_y=max((v.y for v in soq_assign.values()), default=0),
        soqs=[],
        vlines=[],
        hlines=sorted(hlines),
    )

    for hline in hlines:
        if hline.seq_x_end is None:
            raise ValueError(f"A horizontal line has no end: {hline}")

    for binst in nx.topological_sort(binst_graph):
        preds, succs = _binst_to_cxns(binst, binst_graph=binst_graph)

        # Keep track of the extent of our vlines
        binst_top_y = 0
        binst_bot_y = msd.max_y
        binst_x = None

        for pred in preds:
            me = pred.right
            symb = labeller.get_symbol(binst, me.reg, me.idx)
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
            symb = labeller.get_symbol(binst, me.reg, me.idx)
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
                    label=labeller.get_label(binst),
                )
            )

    return msd


def get_musical_score_data(
    bloq: Bloq, manager: Optional[LineManager] = None, labeller: Optional[BinstLabeller] = None
) -> MusicalScoreData:
    """Get the musical score data for a (composite) bloq.

    This will first walk through the compute graph to assign each soquet
    to a register position. Then we iterate again to finalize drawing-relevant
    properties like symbols and the various horizontal and vertical lines.

    Args:
        bloq: The bloq or composite bloq to get drawing data for
        manager: Optionally provide an override of `LineManager` if you want
            to control the allocation of horizontal (register) lines.
        labeller: Optionally provide an override of `BinstLabeller` if you want to
            control how the "labels" at the top of the gates are generated.
    """
    cbloq = bloq.as_composite_bloq()
    return _MusicalScoreLayoutBuilder.from_cbloq(cbloq).do_layout(
        manager=manager, labeller=labeller
    )


def draw_musical_score(
    msd: MusicalScoreData,
    unit_to_inches: float = 0.8,
    max_width: float = 50.0,
    max_height: float = 50.0,
    ax=None,
):
    """Draw the diagram with matplotlib.

    Args:
        msd: The musical score data from `get_musical_score_data`.
        unit_to_inches: The musical score is laid out in "units". Each unit is multiplied
            by this conversion factor to get inches.
        max_width: The maximum width in inches of the figure. It's generally better to
            set this to a large value and let the display backend properly scale down
            large figures. Otherwise, the proportions of symbols to spacing is skewed.
        max_height: The maximum width in inches of the figure.
        ax: If provided, use this matplotlib axis. This causes `max_width` and `max_height`
            to be ignored. The caller is responsible for ensuring an equal aspect ratio,
            or the symbols can be strange.

    """
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
        we_control_fig = True
    else:
        we_control_fig = False

    for hline in msd.hlines:
        assert hline.seq_x_end is not None, hline
        color = 'b' if hline.flavor is HLineFlavor.CLASSICAL else 'k'
        ax.hlines(-hline.y, hline.seq_x_start, hline.seq_x_end, color=color, zorder=-1)

    for vline in msd.vlines:
        ax.vlines(vline.x, -vline.top_y, -vline.bottom_y, color='k', zorder=-1)
        if vline.label:
            symb = Text(vline.label)
            symb.draw(ax, vline.x, vline.bottom_y - 0.5)

    for soq in msd.soqs:
        symb = soq.symb
        symb.draw(ax, soq.rpos.seq_x, soq.rpos.y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    if we_control_fig:
        fig.tight_layout()
        return fig, ax
    else:
        return ax


class MusicalScoreEncoder(json.JSONEncoder):
    """An encoder that handles `MusicalScoreData` classes and those of its contents."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (SoqData, HLine, VLine, MusicalScoreData, WireSymbol, RegPosition)):
            return o.json_dict()

        return super().default(o)


def dump_musical_score(msd: MusicalScoreData, name: str):
    with open(f'{name}.json', 'w') as f:
        json.dump(msd, f, indent=2, cls=MusicalScoreEncoder)
