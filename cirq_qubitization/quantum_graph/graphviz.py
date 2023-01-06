import textwrap
from typing import Any, Dict, Set, Tuple

import pydot

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)




class SimplestGraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq
        self._cbloq = cbloq
        self._binsts = cbloq.bloq_instances
        self._soquets = cbloq.all_soquets

        # id gen
        ibuilder = _IDBuilder()

        for binst in self._cbloq.bloq_instances:
            ibuilder.add(binst, f'{binst.bloq.__class__.__name__}')

            for groupname, groupregs in binst.bloq.registers.groups():
                ibuilder.add((binst, groupname), groupname)

        for soq in self._cbloq.all_soquets:
            ibuilder.add(soq, f'{soq.reg.name}')

        self.ids = ibuilder.build()

    def get_dangle_node(self, soq: Soquet) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(self.ids[soq], label=textwrap.fill(repr(soq), 20))

    def add_dangles(
        self, graph: pydot.Graph, registers: FancyRegisters, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        if dangle is LeftDangle:
            regs = registers.lefts()
        elif dangle is RightDangle:
            regs = registers.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for reg in regs:
            for idx in reg.wire_idxs():
                subg.add_node(self.get_dangle_node(Soquet(dangle, reg, idx=idx)))
        graph.add_subgraph(subg)
        return graph

    def get_soquet_node(self, soq: Soquet) -> pydot.Node:
        return pydot.Node(self.ids[soq], label=textwrap.fill(repr(soq), 20))

    def get_binst_cluster(self, binst: BloqInstance) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst], label=textwrap.fill(repr(binst), 20))

    def get_reggroup_cluster(
        self, binst: BloqInstance, name: str, groupregs: FancyRegisters
    ) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst, name], label=name)

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = self.get_binst_cluster(binst)

        for name, regs in binst.bloq.registers.groups():
            subsubg = self.get_reggroup_cluster(binst, name, regs)

            for reg in regs:
                for idx in reg.wire_idxs():
                    soq = Soquet(binst, reg, idx)
                    subsubg.add_node(self.get_soquet_node(soq))

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
        graph = self.add_dangles(graph, self._cbloq.registers, LeftDangle)

        for binst in self._binsts:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, self._cbloq.registers, RightDangle)

        for cxn in self._cbloq.connections:
            graph = self.add_cxn(graph, cxn)

        return graph


class PortGraphDrawer(SimplestGraphDrawer):
    def get_dangle_node(self, soq: Soquet) -> pydot.Node:
        return pydot.Node(
            self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), shape='plaintext'
        )

    def get_binst_cluster(self, binst: BloqInstance) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst], label=binst.bloq.pretty_name())

    def get_soquet_node(self, soq: Soquet) -> pydot.Node:
        if soq.reg.side is Side.THRU:
            draw_args = dict(shape='rect', style='rounded')
        elif soq.reg.side is Side.LEFT:
            draw_args = dict(shape='house', orientation=-90)
        elif soq.reg.side is Side.RIGHT:
            draw_args = dict(shape='house', orientation=90)
        else:
            raise ValueError()
        return pydot.Node(self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), **draw_args)


class SmallPortGraphDrawer(PortGraphDrawer):
    def get_soquet_node(self, soq: Soquet) -> pydot.Node:
        return pydot.Node(self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), shape='box')

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = self.get_binst_cluster(binst)

        for reg in binst.bloq.registers:
            for idx in reg.wire_idxs():
                soq = Soquet(binst, reg, idx)
                subg.add_node(self.get_soquet_node(soq))

        graph.add_subgraph(subg)
        return graph


class RegisterGraphDrawer(SimplestGraphDrawer):
    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = self.get_binst_cluster(binst)

        for name, regs in binst.bloq.registers.groups():
            subg.add_node(pydot.Node(self.ids[binst, name], label=name, shape='box'))

        graph.add_subgraph(subg)
        return graph

    def add_cxn(self, graph: pydot.Graph, cxn: Connection) -> pydot.Graph:
        def _id(soq: Soquet):
            if isinstance(soq.binst, DanglingT):
                return self.ids[soq]
            return self.ids[soq.binst, soq.reg.name]

        graph.add_edge(
            pydot.Edge(
                _id(cxn.left) + ':e',
                _id(cxn.right) + ':w',
                label=str(cxn.shape),
                labelfloat=True,
                fontsize=10,
            )
        )

        return graph

    def get_dangle_node(self, soq: Soquet) -> pydot.Node:
        return pydot.Node(
            self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), shape='plaintext'
        )

    def get_binst_cluster(self, binst: BloqInstance) -> pydot.Cluster:
        return pydot.Cluster(self.ids[binst], label=binst.bloq.pretty_name())

    def get_soquet_node(self, soq: Soquet) -> pydot.Node:
        if soq.reg.side is Side.THRU:
            draw_args = dict(shape='rect', style='rounded')
        elif soq.reg.side is Side.LEFT:
            draw_args = dict(shape='house', orientation=-90)
        elif soq.reg.side is Side.RIGHT:
            draw_args = dict(shape='house', orientation=90)
        else:
            raise ValueError()
        return pydot.Node(self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), **draw_args)
