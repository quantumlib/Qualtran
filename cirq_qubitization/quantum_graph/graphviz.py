import itertools
from typing import Sequence, Union, Tuple, Set, Optional

import pydot

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import (
    SplitRegister,
    JoinRegister,
    ApplyFRegister,
)
from cirq_qubitization.quantum_graph.quantum_graph import (
    Soquet,
    LeftDangle,
    BloqInstance,
    Wire,
    DanglingT,
    RightDangle,
)
from cirq_qubitization.quantum_graph.util_bloqs import Split, Join


def _get_idx_label(label: str, idx: Tuple[int, ...] = tuple()):
    if len(idx) > 0:
        return f'{label}[{", ".join(str(i) for i in idx)}]'
    return str(label)


class SimplestGraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq
        self._cbloq = cbloq

    def node_id(self, thing: Union[BloqInstance, DanglingT], reg_name: Optional[str] = None) -> str:
        parts = []
        if isinstance(thing, BloqInstance):
            parts += [thing.bloq.__class__.__name__, str(thing.i)]
        else:
            if thing is LeftDangle:
                parts.append('LeftDangle')
            elif thing is RightDangle:
                parts.append('RightDangle')
            else:
                raise ValueError()

        if reg_name is not None:
            parts.append(reg_name)

        return '_'.join(parts)

    def node_compass_id(self, thing: BloqInstance, reg_name: str, ew: str):
        return f'{self.node_id(thing, reg_name)}:{ew}'

    def get_dangle_node(self, dangle: DanglingT, reg_name: str) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(
            self.node_id(dangle, reg_name), label=self.node_id(dangle, reg_name), shape='plaintext'
        )

    def add_dangles(
        self, graph: pydot.Graph, registers: Registers, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        dang = pydot.Subgraph(rank='same')
        for reg in registers:
            dang.add_node(self.get_dangle_node(dangle, reg.name))
        graph.add_subgraph(dang)
        return graph

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = pydot.Cluster(self.node_id(binst), label=self.node_id(binst))

        for reg in binst.bloq.registers:
            subg.add_node(
                pydot.Node(self.node_id(binst, reg.name), label=self.node_id(binst, reg.name))
            )

        graph.add_subgraph(subg)
        return graph

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
        graph.add_edge(
            pydot.Edge(
                self.node_compass_id(wire.left.binst, wire.left.reg_name, 'e'),
                self.node_compass_id(wire.right.binst, wire.right.reg_name, 'w'),
                arrowhead='normal',
                label=str(wire.shape),
                labelfloat=True,
                fontsize=10,
            )
        )

        return graph

    def graphviz(self) -> pydot.Graph:
        binsts = set(
            wire.left.binst
            for wire in self._cbloq.wires
            if not isinstance(wire.left.binst, DanglingT)
        )
        binsts |= set(
            wire.right.binst
            for wire in self._cbloq.wires
            if not isinstance(wire.right.binst, DanglingT)
        )

        graph = pydot.Dot('qual', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, self._cbloq.registers, LeftDangle)

        for binst in binsts:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, self._cbloq.registers, RightDangle)

        for wire in self._cbloq.wires:
            graph = self.add_wire(graph, wire)

        return graph


class GraphDrawer:
    def __init__(self, bloq: Bloq):
        if not isinstance(bloq, CompositeBloq):
            cbloq = bloq.as_composite_bloq()
        else:
            cbloq = bloq

        nodes = set(
            wire.left.binst for wire in cbloq.wires if not isinstance(wire.left.binst, DanglingT)
        )
        nodes |= set(
            wire.right.binst for wire in cbloq.wires if not isinstance(wire.right.binst, DanglingT)
        )
        self.nodes: Set[BloqInstance] = nodes
        self.wires: Sequence[Wire] = cbloq.wires

        self.registers: Registers = cbloq.registers
        self._cbloq: CompositeBloq = cbloq

    def to_pretty(self):
        """Return a PrettyGraphDrawer version of this that overrides methods to make the
        display more pretty but less explicit."""

        # TODO: constructor method
        return PrettyGraphDrawer(self._cbloq)

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
        self, soq: Soquet, lr: Optional[str] = None
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

    def node_id(self, thing: Union[Soquet, BloqInstance], lr: Optional[str] = None) -> str:
        if isinstance(thing, Soquet):
            soq = thing
        elif isinstance(thing, BloqInstance):
            soq = Soquet(thing, '')
        else:
            raise ValueError()

        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        return nid

    def node_port_id(self, soq: Soquet, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(portname, str)
        return f'{nid}:{portname}'

    def node_port_compass_id(self, soq: Soquet, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(portname, str)
        assert isinstance(ew, str)
        return f'{nid}:{portname}:{ew}'

    def node_compass_id(self, soq: Soquet, lr: Optional[str] = None):
        nid, portname, ew = self.get_id_parts(soq=soq, lr=lr)
        assert isinstance(nid, str)
        assert isinstance(ew, str)
        return f'{nid}:{ew}'

    def get_dangle_node(self, dangle: DanglingT, reg_name: str, idx: Tuple[int, ...]) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(
            self.node_id(Soquet(dangle, reg_name, idx)),
            label=_get_idx_label(reg_name, idx),
            shape='plaintext',
        )

    def add_dangles(self, graph: pydot.Graph, dangle: DanglingT) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        dang = pydot.Subgraph(rank='same')
        for reg in self.registers:
            if dangle is LeftDangle:
                shape = reg.left_shape
            elif dangle is RightDangle:
                shape = reg.right_shape
            else:
                raise ValueError()

            *wireshape, bitsize = shape
            for si in itertools.product(*wireshape):
                dang.add_node(self.get_dangle_node(dangle, reg.name, si))
        graph.add_subgraph(dang)
        return graph

    def get_split_register(self, reg: SplitRegister) -> str:
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

    def get_join_register(self, reg: JoinRegister) -> str:
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

    def get_apply_f_text(self, reg: ApplyFRegister) -> Tuple[str, str]:
        return reg.name, reg.out_name

    def get_apply_f_register(self, reg: ApplyFRegister) -> str:
        t1, t2 = self.get_apply_f_text(reg)
        return f'<TR><TD port="{self.port_id(reg.name)}">{t1}</TD><TD port="{self.port_id(reg.name, right=True)}">{t2}</TD></TR>'

    def get_default_text(self, reg: Register) -> str:
        return reg.name

    def get_default_register(self, reg: Register, colspan: str = '') -> str:
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

        complex_reg_types = (SplitRegister, JoinRegister, ApplyFRegister)
        have_complex_regs = any(isinstance(reg, complex_reg_types) for reg in binst.bloq.registers)
        header_colspan = 'colspan="2"' if have_complex_regs else ''
        label += f'<tr><td {header_colspan}>{self.get_binst_header_text(binst)}</td></tr>'

        for reg in binst.bloq.registers:
            if isinstance(reg, SplitRegister):
                label += self.get_split_register(reg)
            elif isinstance(reg, JoinRegister):
                label += self.get_join_register(reg)
            elif isinstance(reg, ApplyFRegister):
                label += self.get_apply_f_register(reg)
            else:
                label += self.get_default_register(reg, header_colspan)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        graph.add_node(pydot.Node(self.node_id(binst), label=label, shape='plain'))
        return graph

    def wire_label(self, wire: Wire) -> str:
        return str(wire.shape)

    def add_edge(self, graph: pydot.Graph, tail: str, head: str, label: str):
        graph.add_edge(
            pydot.Edge(tail, head, arrowhead='normal', label=label, labelfloat=True, fontsize=10)
        )

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
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
        graph = pydot.Dot('qual', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, LeftDangle)

        for binst in self.nodes:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, RightDangle)

        for wire in self.wires:
            graph = self.add_wire(graph, wire)

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

    def get_apply_f_text(self, reg: ApplyFRegister) -> Tuple[str, str]:
        return reg.in_text, reg.out_text

    def get_default_text(self, reg: Register) -> str:
        if reg.name == 'control':
            return '\u2b24'

        return reg.name

    def add_edge(self, graph: pydot.Graph, tail: str, head: str, label: str):
        graph.add_edge(pydot.Edge(tail, head, arrowhead='none'))


def _binst_id2(x: BloqInstance):
    if x is LeftDangle:
        return 'dang_left'
    if x is RightDangle:
        return 'dang_right'

    return f'{x.bloq.__class__.__name__}_{x.i}'


def _pgid(p: Soquet):
    idx = "_".join(str(i) for i in p.idx)
    return f'{_binst_id2(p.binst)}_{p.reg_name}_{idx}'


class PortGraphDrawer(GraphDrawer):
    def get_dangle_node(self, dangle: DanglingT, soqname: str, idx: Tuple[int, ...]) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(
            _pgid(Soquet(dangle, soqname, idx)),
            label=_get_idx_label(soqname, idx),
            shape='plaintext',
        )

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = pydot.Cluster(_binst_id2(binst), label=binst.bloq.pretty_name())

        for reg in binst.bloq.registers:
            *lwireshape, bitsize = reg.left_shape
            *rwireshape, bitsize = reg.right_shape
            if lwireshape == rwireshape:
                draw_args = dict(shape='rect', style='rounded')
                r_idx_iter = []
            else:
                draw_args = dict(shape='house', orientation=-90)
                r_idx_iter = itertools.product(*[range(sh) for sh in rwireshape])

            for idx in itertools.product(*[range(sh) for sh in lwireshape]):
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.name, idx)),
                        label=_get_idx_label(reg.name, idx),
                        **draw_args,
                    )
                )

            for idx in r_idx_iter:
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.name, idx)),
                        label=_get_idx_label(reg.name, idx),
                        shape='house',
                        orientation=90,
                    )
                )

        graph.add_subgraph(subg)
        return graph

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
        self.add_edge(
            graph, _pgid(wire.left) + ':e', _pgid(wire.right) + ":w", label=self.wire_label(wire)
        )

        return graph
