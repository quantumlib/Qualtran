import itertools
from typing import Any, Dict, Optional, Set, Tuple, Union

import pydot

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)
from cirq_qubitization.quantum_graph.util_bloqs import Join, Partition, Split, Unpartition


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


def _get_idx_label(label: str, idx: Tuple[int, ...] = tuple()):
    if len(idx) > 0:
        return f'{label}[{", ".join(str(i) for i in idx)}]'
    return str(label)


def _get_idx_label2(soq: Soquet):
    return _get_idx_label(soq.reg.name, soq.idx)


class GraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq
        self._cbloq = cbloq
        self._binsts = cbloq.bloq_instances
        self._soquets = cbloq.all_soquets

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
        return pydot.Node(
            self.ids[soq], label=_get_idx_label(soq.reg.name, soq.idx), shape='plaintext'
        )

    def add_dangles(
        self, graph: pydot.Graph, soquets: FancyRegisters, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        if dangle is LeftDangle:
            regs = soquets.lefts()
        elif dangle is RightDangle:
            regs = soquets.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for reg in regs:
            for idx in reg.wire_idxs():
                subg.add_node(self.get_dangle_node(Soquet(dangle, reg, idx=idx)))
        graph.add_subgraph(subg)
        return graph

    def soq_label(self, soq: Soquet):
        return soq.pretty()

    def get_thru_register(self, thru: Soquet):
        return f'<TR><TD colspan="2" port="{self.ids[thru]}">{self.soq_label(thru)}</TD></TR>'

    def get_register(
        self, left: Soquet, right: Soquet, left_rowspan=1, right_rowspan=1, with_empty_td=True
    ):

        label = '<TR>'
        if left is not None:
            if left_rowspan != 1:
                assert left_rowspan > 1
                left_rowspan = f'rowspan="{left_rowspan}"'
            else:
                left_rowspan = ''

            label += f'<TD {left_rowspan} port="{self.ids[left]}">{self.soq_label(left)}</TD>'
        else:
            if with_empty_td:
                label += '<TD></TD>'
            pass

        if right is not None:
            if right_rowspan != 1:
                assert right_rowspan > 1
                right_rowspan = f'rowspan="{right_rowspan}"'
            else:
                right_rowspan = ''
            label += f'<TD {right_rowspan} port="{self.ids[right]}">{self.soq_label(right)}</TD>'
        else:
            if with_empty_td:
                label += '<TD></TD>'
            pass

        label += '</TR>'
        return label

    def get_binst_table_attributes(self) -> str:
        """Return the desired table attributes for the bloq."""
        return 'BORDER="1" CELLBORDER="1" CELLSPACING="3"'

    def get_binst_header_text(self, binst: BloqInstance) -> str:
        """Get the text used for the 'header' cell of a bloq."""
        return f'{binst.bloq.pretty_name()}'

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        """Add a BloqInstance to the graph."""

        label = '<'  # graphviz: start an HTML section
        label += f'<TABLE {self.get_binst_table_attributes()}>'

        label += f'<tr><td colspan="2">{self.get_binst_header_text(binst)}</td></tr>'

        for groupname, groupregs in binst.bloq.registers.groups():
            lefts = []
            rights = []
            thrus = []
            for reg in groupregs:
                for idx in reg.wire_idxs():
                    soq = Soquet(binst, reg, idx)
                    if reg.side is Side.LEFT:
                        lefts.append(soq)
                    elif reg.side is Side.RIGHT:
                        rights.append(soq)
                    else:
                        assert reg.side is Side.THRU
                        thrus.append(soq)

            if len(thrus) > 0:
                assert len(lefts) == 0
                assert len(rights) == 0
                for t in thrus:
                    label += self.get_thru_register(t)
            else:
                n_surplus_lefts = max(0, len(lefts) - len(rights))
                n_surplus_rights = max(0, len(rights) - len(lefts))
                n_common = min(len(lefts), len(rights))

                if n_common >= 1:
                    for i in range(n_common - 1):
                        label += self.get_register(lefts[i], rights[i])
                    label += self.get_register(
                        lefts[n_common - 1],
                        rights[n_common - 1],
                        left_rowspan=n_surplus_rights + 1,
                        right_rowspan=n_surplus_lefts + 1,
                    )
                    for l, r in itertools.zip_longest(
                        lefts[n_common:], rights[n_common:], fillvalue=None
                    ):
                        label += self.get_register(l, r, with_empty_td=False)
                else:
                    for l, r in itertools.zip_longest(lefts, rights, fillvalue=None):
                        label += self.get_register(l, r)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        kwargs = {}
        if isinstance(binst.bloq, CompositeBloq):
            kwargs['color'] = 'navy'

        graph.add_node(pydot.Node(self.ids[binst], label=label, shape='plain', **kwargs))
        return graph

    def wire_label(self, wire: Connection) -> str:
        return str(wire.shape)

    def add_cxn(self, graph: pydot.Graph, cxn: Connection) -> pydot.Graph:
        if cxn.left.binst is LeftDangle:
            left = f'{self.ids[cxn.left]}:e'
        else:
            left = f'{self.ids[cxn.left.binst]}:{self.ids[cxn.left]}:e'

        if cxn.right.binst is RightDangle:
            right = f'{self.ids[cxn.right]}:w'
        else:
            right = f'{self.ids[cxn.right.binst]}:{self.ids[cxn.right]}:w'

        graph.add_edge(
            pydot.Edge(
                left,
                right,
                label=str(cxn.shape),
                labelfloat=True,
                fontsize=10,
                arrowhead='dot',
                arrowsize=0.25,
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


class PrettyGraphDrawer(GraphDrawer):
    INFRA_BLOQ_TYPES = (Split, Join, Partition, Unpartition)

    def get_binst_table_attributes(self) -> str:
        return 'BORDER="0" CELLBORDER="1" CELLSPACING="0"'

    def get_binst_header_text(self, binst: BloqInstance):
        if isinstance(binst.bloq, self.INFRA_BLOQ_TYPES):
            return ''
        return f'<font point-size="10">{binst.bloq.short_name()}</font>'

    def soq_label(self, soq: Soquet):
        if isinstance(soq.binst, BloqInstance) and isinstance(
            soq.binst.bloq, self.INFRA_BLOQ_TYPES
        ):
            return ''
        return soq.pretty()

    def get_default_text(self, reg: FancyRegister) -> str:
        if reg.name == 'control':
            return '\u2b24'

        return reg.name
