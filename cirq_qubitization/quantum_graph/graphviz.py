from typing import Sequence, Union, Tuple, Set

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


def _binst_id(x: BloqInstance):
    # should be fine as long as all the `i`s are unique.
    return f'{x.bloq.__class__.__name__}_{x.i}'


def _binst_in_port(port: Soquet):
    return f'{_binst_id(port.binst)}:{port.reg_name}:w'


def _binst_out_port(port: Soquet):
    return f'{_binst_id(port.binst)}:{port.reg_name}:e'


def _dangling_id(port: Soquet):
    # Can we collide with a binst_id? Probably not unless we have a class named
    # DanglingT_l with integer reg_name.
    assert isinstance(port.binst, DanglingT)
    return f'DanglingT_{port.binst.direction}_{port.reg_name}'


class GraphDrawer:
    def __init__(self, bloq: Bloq):
        if isinstance(bloq, CompositeBloq):
            cbloq = bloq
        else:
            # TODO: factor out bloq -> compositebloq wrapping
            bb = CompositeBloqBuilder(bloq.registers)
            port_dict = {reg.name: Soquet(LeftDangle, reg.name) for reg in bloq.registers}
            stuff = bb.add(bloq, **port_dict)
            cbloq = bb.finalize(**{reg.name: s for reg, s in zip(bloq.registers, stuff)})

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

    def get_dangle_node(self, dangle: DanglingT, reg: Register) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(
            _dangling_id(Soquet(dangle, reg.name)), label=f'{reg.name}', shape='plaintext'
        )

    def add_dangles(self, graph: pydot.Graph, dangle: DanglingT) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        dang = pydot.Subgraph(rank='same')
        for reg in self.registers:
            dang.add_node(self.get_dangle_node(dangle, reg))
        graph.add_subgraph(dang)
        return graph

    def get_split_text(self, i: Union[None, int]):
        if i is None:
            return 'in'
        return str(i)

    def get_split_register(self, reg: SplitRegister) -> str:
        """Return <TR>s for a SplitRegister."""
        label = ''
        for i in range(reg.bitsize):
            in_td = (
                f'<TD rowspan="{reg.bitsize}" port="{reg.name}">{self.get_split_text(None)}</TD>'
                if i == 0
                else ''
            )
            label += f'<TR>{in_td}<TD port="{reg.name}{i}">{self.get_split_text(i)}</TD></TR>'

        return label

    def get_join_text(self, i: Union[None, int]):
        if i is None:
            return 'out'
        return str(i)

    def get_join_register(self, reg: JoinRegister) -> str:
        """Return <TR>s for a JoinRegister."""
        label = ''
        for i in range(reg.bitsize):
            out_td = (
                f'<TD rowspan="{reg.bitsize}" port="{reg.name}">{self.get_join_text(None)}</TD>'
                if i == 0
                else ''
            )
            label += f'<TR><TD port="{reg.name}{i}">{self.get_join_text(i)}</TD>{out_td}</TR>'

        return label

    def get_apply_f_text(self, reg: ApplyFRegister) -> Tuple[str, str]:
        return reg.name, reg.out_name

    def get_apply_f_register(self, reg: ApplyFRegister) -> str:
        t1, t2 = self.get_apply_f_text(reg)
        return f'<TR><TD port="{reg.name}">{t1}</TD><TD port="{reg.out_name}">{t2}</TD></TR>'

    def get_default_text(self, reg: Register) -> str:
        return reg.name

    def get_default_register(self, reg: Register, colspan: str = '') -> str:
        """Return a <TR> for a normal Register."""
        return f'<TR><TD {colspan} PORT="{reg.name}">{self.get_default_text(reg)}</TD></TR>'

    def get_binst_table_attributes(self) -> str:
        """Return the desired table attributes for the bloq."""
        return 'BORDER="1" CELLBORDER="1" CELLSPACING="3"'

    def get_binst_header_text(self, binst: BloqInstance) -> str:
        """Get the text used for the 'header' cell of a bloq."""
        return _binst_id(binst)

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

        graph.add_node(pydot.Node(_binst_id(binst), label=label, shape='plain'))
        return graph

    def get_arrowhead(self):
        return 'normal'

    def add_edge(self, graph: pydot.Graph, tail: str, head: str):
        graph.add_edge(pydot.Edge(tail, head, arrowhead=self.get_arrowhead()))

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
        if wire.left.binst is LeftDangle:
            self.add_edge(graph, _dangling_id(wire.left), _binst_in_port(wire.right))
        elif wire.right.binst is RightDangle:
            self.add_edge(graph, _binst_out_port(wire.left), _dangling_id(wire.right))
        else:
            self.add_edge(graph, _binst_out_port(wire.left), _binst_in_port(wire.right))

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

    def get_arrowhead(self):
        return 'none'


def _binst_id2(x: BloqInstance):
    if isinstance(x, DanglingT):
        return f'dang{x.direction}'

    return f'{x.bloq.__class__.__name__}_{x.i}'


def _pgid(p: Soquet):
    return f'{_binst_id2(p.binst)}_{p.reg_name}'


class PortGraphDrawer(GraphDrawer):
    def get_dangle_node(self, dangle: DanglingT, reg: Register) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(_pgid(Soquet(dangle, reg.name)), label=reg.name, shape='plaintext')

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        subg = pydot.Cluster(_binst_id2(binst), label=binst.bloq.pretty_name())

        for reg in binst.bloq.registers:
            # TODO: abstract method that gives all the in/out ports
            if isinstance(reg, SplitRegister):
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.name)),
                        label=reg.name,
                        shape='house',
                        orientation=-90,
                    )
                )
                for i in range(reg.bitsize):
                    sname = f'{reg.name}{i}'
                    subg.add_node(
                        pydot.Node(
                            _pgid(Soquet(binst, sname)), label=sname, shape='house', orientation=90
                        )
                    )
            elif isinstance(reg, JoinRegister):
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.name)),
                        label=reg.name,
                        shape='house',
                        orientation=90,
                    )
                )
                for i in range(reg.bitsize):
                    sname = f'{reg.name}{i}'
                    subg.add_node(
                        pydot.Node(
                            _pgid(Soquet(binst, sname)), label=sname, shape='house', orientation=-90
                        )
                    )
            elif isinstance(reg, ApplyFRegister):
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.name)),
                        label=reg.name,
                        shape='house',
                        orientation=-90,
                    )
                )
                subg.add_node(
                    pydot.Node(
                        _pgid(Soquet(binst, reg.out_name)),
                        label=reg.out_name,
                        shape='house',
                        orientation=90,
                    )
                )
            else:
                subg.add_node(
                    pydot.Node(_pgid(Soquet(binst, reg.name)), label=reg.name, shape='rect')
                )

        graph.add_subgraph(subg)
        return graph

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
        self.add_edge(graph, _pgid(wire.left) + ':e', _pgid(wire.right) + ":w")

        return graph
