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


from cirq_qubitization.quantum_graph.graphviz import (
    _get_wires_from_conns,
    _get_binsts_from_cxns,
    _get_idx_label,
)


class GraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq
        self._cbloq = cbloq
        self._binsts = _get_binsts_from_cxns(cbloq.cxns)
        self._wires = _get_wires_from_conns(cbloq.cxns)

    def port_id(self, reg_name: str, idx: Tuple[int, ...] = tuple(), *, right=False) -> str:
        idx = "_".join(str(i) for i in idx)
        if idx:
            idx = '_' + idx
        if right:
            right = '_r'
        else:
            right = ''
        return f'{reg_name}{idx}{right}'

    def get_id_parts(
        self, soq: Wire, lr: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        # node, port (if any), east/west
        if isinstance(soq.binst, DanglingT):
            if soq.binst is LeftDangle:
                if lr is not None:
                    assert lr == 'r'
                dir = 'e'
            elif soq.binst is RightDangle:
                if lr is not None:
                    assert lr == 'l'
                dir = 'w'
            else:
                raise ValueError()
            return f'DanglingT_{dir}_{soq.reg_name}', None, dir

        binst = f'{soq.binst.bloq.__class__.__name__}_{soq.binst.i}'

        if lr is None:
            ew = None
        elif lr == 'l':
            ew = 'w'
        elif lr == 'r':
            ew = 'e'
        else:
            raise ValueError()

        if lr == 'r':
            # TODO: only set for special things
            right = True
        else:
            right = False

        return binst, self.port_id(soq.reg_name, soq.idx, right=right), ew

    def node_id(self, thing: Union[Wire, BloqInstance], lr: Optional[str] = None) -> str:
        if isinstance(thing, Wire):
            soq = thing
        elif isinstance(thing, BloqInstance):
            soq = Wire(thing, '')
        else:
            raise ValueError()

        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        return nid

    def node_port_id(self, soq: Wire, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(portname, str)
        return f'{nid}:{portname}'

    def node_port_compass_id(self, soq: Wire, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(portname, str)
        assert isinstance(ew, str)
        return f'{nid}:{portname}:{ew}'

    def node_compass_id(self, soq: Wire, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(ew, str)
        return f'{nid}:{ew}'

    def get_dangle_node(self, wire: Wire) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(
            self.ids[wire], label=_get_idx_label(wire.soq.name, wire.idx), shape='plaintext'
        )

    def add_dangles(self, graph: pydot.Graph, soquets: Soquets, dangle: DanglingT) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        dang = pydot.Subgraph(rank='same')

        if dangle is LeftDangle:
            regs = soquets.lefts()
        elif dangle is RightDangle:
            regs = soquets.rights()
        else:
            raise ValueError()

        for reg in regs:
            for si in itertools.product(*reg.wireshape):
                dang.add_node(self.get_dangle_node(Wire(binst=dangle, soq=reg, idx=si)))
        graph.add_subgraph(dang)
        return graph

    def get_split_register(self, reg) -> str:
        """Return <TR>s for a SplitRegister."""
        label = ''
        *rwireshape, bitsize = reg.right_shape
        first = True
        for idx in itertools.product(*[range(sh) for sh in rwireshape]):
            in_td = (
                f'<TD rowspan="{reg.n}" port="{self.port_id(reg.name)}">{_get_idx_label(reg.name)}</TD>'
                if first
                else ''
            )
            first = False
            label += f'<TR>{in_td}<TD port="{self.port_id(reg.name, idx, right=True)}">{_get_idx_label(reg.name, idx)}</TD></TR>'

        return label

    def get_join_register(self, reg) -> str:
        """Return <TR>s for a JoinRegister."""
        label = ''

        first = True
        *lwireshape, bitsize = reg.left_shape
        for idx in itertools.product(*[range(sh) for sh in lwireshape]):
            out_td = (
                f'<TD rowspan="{reg.n}" port="{self.port_id(reg.name, right=True)}">{_get_idx_label(reg.name)}</TD>'
                if first
                else ''
            )
            first = False
            label += f'<TR><TD port="{self.port_id(reg.name, idx)}">{_get_idx_label(reg.name, idx)}</TD>{out_td}</TR>'

        return label

    def get_apply_f_text(self, reg) -> Tuple[str, str]:
        return reg.name, reg.out_name

    def get_apply_f_register(self, reg) -> str:
        t1, t2 = self.get_apply_f_text(reg)
        return f'<TR><TD port="{self.port_id(reg.name)}">{t1}</TD><TD port="{self.port_id(reg.name, right=True)}">{t2}</TD></TR>'

    def get_default_text(self, reg: FancyRegister) -> str:
        return reg.name

    def get_default_register(self, reg: FancyRegister, colspan: str = '') -> str:
        """Return a <TR> for a normal Register."""
        return f'<TR><TD {colspan} port="{self.port_id(reg.name)}">{self.get_default_text(reg)}</TD></TR>'

    def get_binst_table_attributes(self) -> str:
        """Return the desired table attributes for the bloq."""
        return 'BORDER="1" CELLBORDER="1" CELLSPACING="3"'

    def get_binst_header_text(self, binst: BloqInstance) -> str:
        """Get the text used for the 'header' cell of a bloq."""
        return self.node_id(binst)

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        """Add a BloqInstance to the graph."""
        label = '<'  # graphviz: start an HTML section
        label += f'<TABLE {self.get_binst_table_attributes()}>'

        have_complex_regs = any((not (soq.side is Side.THRU)) for soq in binst.bloq.soquets)
        have_complex_regs = True
        header_colspan = 'colspan="2"' if have_complex_regs else ''
        label += f'<tr><td {header_colspan}>{self.get_binst_header_text(binst)}</td></tr>'

        for soq in binst.bloq.soquets:
            for idx in itertools.product(*(range(sh) for sh in soq.wireshape)):
                wire = Wire(binst, soq, idx)
                label += self.get_wire_cell(wire)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        graph.add_node(pydot.Node(self.node_id(binst), label=label, shape='plain'))
        return graph

    def wire_label(self, wire: Connection) -> str:
        return str(wire.shape)

    def add_edge(self, graph: pydot.Graph, tail: str, head: str, label: str):
        graph.add_edge(
            pydot.Edge(tail, head, arrowhead='normal', label=label, labelfloat=True, fontsize=10)
        )

    def add_wire(self, graph: pydot.Graph, wire: Connection) -> pydot.Graph:
        if wire.left.binst is LeftDangle:
            self.add_edge(
                graph,
                self.node_compass_id(wire.left, lr='r'),
                self.node_port_compass_id(wire.right, lr='l'),
                label=self.wire_label(wire),
            )
        elif wire.right.binst is RightDangle:
            self.add_edge(
                graph,
                self.node_port_compass_id(wire.left, lr='r'),
                self.node_compass_id(wire.right, lr='l'),
                label=self.wire_label(wire),
            )
        else:
            self.add_edge(
                graph,
                self.node_port_compass_id(wire.left, lr='r'),
                self.node_port_compass_id(wire.right, lr='l'),
                label=self.wire_label(wire),
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


class PrettyGraphDrawer(GraphDrawer):
    def get_binst_table_attributes(self) -> str:
        return 'BORDER="0" CELLBORDER="1" CELLSPACING="0"'

    def get_binst_header_text(self, binst: BloqInstance):
        if isinstance(binst.bloq, (Split, Join)):
            return ''
        return f'<font point-size="10">{binst.bloq.short_name()}</font>'

    def get_split_text(self, i: Union[None, int]):
        return ''

    def get_join_text(self, i: Union[None, int]):
        return ''

    def get_apply_f_text(self, reg) -> Tuple[str, str]:
        return reg.in_text, reg.out_text

    def get_default_text(self, reg: FancyRegister) -> str:
        if reg.name == 'control':
            return '\u2b24'

        return reg.name

    def add_edge(self, graph: pydot.Graph, tail: str, head: str, label: str):
        graph.add_edge(pydot.Edge(tail, head, arrowhead='none'))
