import itertools
import textwrap
from typing import Sequence, Union, Tuple, Set, Optional, Any, Dict

import pydot

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import Soquets, FancyRegister, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    Wire,
    LeftDangle,
    BloqInstance,
    Connection,
    DanglingT,
    RightDangle,
)
from cirq_qubitization.quantum_graph.util_bloqs import Split, Join


def _get_idx_label(label: str, idx: Tuple[int, ...] = tuple()):
    if len(idx) > 0:
        return f'{label}[{", ".join(str(i) for i in idx)}]'
    return str(label)


class _IDBuilder:
    def __init__(self):
        self._to_id: Dict[Any, str] = {}
        self._ids: Set[str] = set()
        self._disambiguator = 0
        self._valid = True

    def add(self, item: Any, desired_id: str):
        if item in self._to_id:
            raise ValueError(f"Item {item} was already added to the ID mapping.")

        if not self._valid:
            raise ValueError("This builder is no longer valid.")

        if desired_id not in self._ids:
            unique_id = desired_id
        else:
            unique_id = f'{desired_id}_G{self._disambiguator}'
            self._disambiguator += 1

        self._ids.add(unique_id)
        self._to_id[item] = unique_id

    def build(self) -> Dict[Any, str]:
        self._valid = False
        return self._to_id


def _get_binsts_from_cxns(cxns: Sequence[Connection]) -> Set[BloqInstance]:
    binsts = set(cxn.left.binst for cxn in cxns if not isinstance(cxn.left.binst, DanglingT))
    binsts |= set(wire.right.binst for wire in cxns if not isinstance(wire.right.binst, DanglingT))
    return binsts


def _get_wires_from_conns(cxns: Sequence[Connection]) -> Set[Wire]:
    wires = set(conn.left for conn in cxns)
    wires |= set(conn.right for conn in cxns)
    return wires


class SimplestGraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq
        self._cbloq = cbloq
        self._binsts = _get_binsts_from_cxns(cbloq.cxns)
        self._wires = _get_wires_from_conns(cbloq.cxns)

        # id gen
        i = _IDBuilder()

        for binst in self._binsts:
            i.add(binst, f'{binst.bloq.__class__.__name__}')

        for wire in self._wires:
            i.add(wire, f'{wire.soq.name}')

        self.ids = i.build()

        self.aa = 0  # todo

    def get_dangle_node(self, wire: Wire) -> pydot.Node:
        """Get a Node representing dangling indices."""
        assert isinstance(wire.binst, DanglingT)
        return pydot.Node(self.ids[wire], label=textwrap.fill(repr(wire), 20))

    def add_dangles(self, graph: pydot.Graph, soquets: Soquets, dangle: DanglingT) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        if dangle is LeftDangle:
            soqs = soquets.lefts()
        elif dangle is RightDangle:
            soqs = soquets.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for soq in soqs:
            for si in itertools.product(*(range(sh) for sh in soq.wireshape)):
                subg.add_node(self.get_dangle_node(Wire(dangle, soq, idx=si)))
        graph.add_subgraph(subg)
        return graph

    def get_wire_node(self, wire: Wire) -> pydot.Node:
        return pydot.Node(self.ids[wire], label=textwrap.fill(repr(wire), 20))

    def get_binst_cluster(self, binst: BloqInstance) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst], label=textwrap.fill(repr(binst), 20))

    def get_soq_cluster(self, name: str, soqs: Soquets) -> pydot.Cluster:
        self.aa += 1  # TODO
        return pydot.Cluster(f'soqs{self.aa}', label=name)

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = self.get_binst_cluster(binst)

        for name, soqs in binst.bloq.soquets.groups():
            subsubg = self.get_soq_cluster(name, soqs)

            for soq in soqs:
                for idx in soq.wire_idxs():
                    wire = Wire(binst, soq, idx)
                    subsubg.add_node(self.get_wire_node(wire))

            subg.add_subgraph(subsubg)

        graph.add_subgraph(subg)
        return graph

    def add_cxn(self, graph: pydot.Graph, cxn: Connection) -> pydot.Graph:
        graph.add_edge(
            pydot.Edge(
                self.ids[cxn.left] + ':e',
                self.ids[cxn.right] + ':w',
                label=str(cxn.shape),
                labelfloat=True,
                fontsize=10,
            )
        )

        return graph

    def graphviz(self) -> pydot.Graph:
        graph = pydot.Dot('my_graph', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, self._cbloq.soquets, LeftDangle)

        for binst in self._binsts:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, self._cbloq.soquets, RightDangle)

        for cxn in self._cbloq.cxns:
            graph = self.add_cxn(graph, cxn)

        return graph


class PortGraphDrawer(SimplestGraphDrawer):
    def get_dangle_node(self, wire: Wire) -> pydot.Node:
        return pydot.Node(
            self.ids[wire], label=_get_idx_label(wire.soq.name, wire.idx), shape='plaintext'
        )

    def get_binst_cluster(self, binst: BloqInstance) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst], label=binst.bloq.pretty_name())

    def get_wire_node(self, wire: Wire) -> pydot.Node:
        if wire.soq.side is Side.THRU:
            draw_args = dict(shape='rect', style='rounded')
        elif wire.soq.side is Side.LEFT:
            draw_args = dict(shape='house', orientation=-90)
        elif wire.soq.side is Side.RIGHT:
            draw_args = dict(shape='house', orientation=90)
        else:
            raise ValueError()
        return pydot.Node(
            self.ids[wire], label=_get_idx_label(wire.soq.name, wire.idx), **draw_args
        )
