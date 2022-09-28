import abc
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List, Dict, overload, Any, Tuple

import cirq

from cirq_qubitization import MultiTargetCSwap
from cirq_qubitization.atoms import Join
from cirq_qubitization.gate_with_registers import Registers, SplitRegister, Register
import networkx as nx
from collections import defaultdict
import cirq_qubitization.testing as cq_testing

import pydot


class Bloq(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def pretty_name(self) -> str:
        return self.__class__.__name__


class CompositeBloq(Bloq):
    def __init__(self, wires: Sequence['Wire'], provenance=None):
        self._wires = wires
        self._provenance = provenance
        self._registers = Registers.build()  # todo

    @property
    def registers(self) -> Registers:
        return self._registers


@dataclass(frozen=True)
class BloqInstance:
    bloq: Bloq
    i: int

    def __repr__(self):
        return f'{self.bloq!r}<{self.i}>'


class RegObj:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return f'{self.name}<{id(self)}>'


@dataclass(frozen=True)
class Split(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([SplitRegister(name='sss', bitsize=self.bitsize)])


class BloqBuilder:
    def __init__(self, parent_reg: Registers):
        self.i = 0
        self._wires = []
        self._prev_inst = {reg.name: LeftDangle for reg in parent_reg}
        self._tracers = tuple(RegObj(reg.name) for reg in parent_reg)
        self._used = set()
        print('bbstart', self._prev_inst, self._tracers, self._used)

    def get_tracers(self):
        return self._tracers

    def split(self, x: RegObj, n: int):
        bloq = Split(n)
        splitname = bloq.registers[0].name
        inst = BloqInstance(Split(n), self.i)
        self.i += 1

        fr_name = x.name
        self._wires.append(Wire(self._prev_inst[fr_name], fr_name, inst, splitname))
        print('wire_add', self._wires[-1])
        del self._prev_inst[fr_name]
        new_reg_objs = tuple(RegObj(f'{splitname}{i}') for i in range(n))
        for i in range(n):
            robj = new_reg_objs[i]
            to_name = robj.name
            self._prev_inst[to_name] = inst

        print("prev_inst", self._prev_inst)
        return new_reg_objs

    def add(self, bloq: Bloq, **reg_map: RegObj):
        inst = BloqInstance(bloq, i=self.i)
        self.i += 1
        for to_name, fr_obj in reg_map.items():
            fr_name = fr_obj.name
            if fr_obj in self._used:
                raise TypeError(f"{fr_obj} re-used!")
            self._used.add(fr_obj)
            self._wires.append(Wire(self._prev_inst[fr_name], fr_name, inst, to_name))
            print('wire_add', self._wires[-1])
            del self._prev_inst[fr_name]
            self._prev_inst[to_name] = inst  # todo: translate if bloq has different in/out reg.

        print('prev_inst', self._prev_inst)
        return tuple(RegObj(name=reg.name) for reg in bloq.registers)

    def finalize(self) -> CompositeBloq:
        return CompositeBloq(wires=self._wires)


@dataclass(frozen=True)
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            control=1,
            x=self.x_bitsize
        )


@dataclass(frozen=True)
class ModMultiply(Bloq):
    exponent_bitsize: int
    x_bitsize: int
    mul_constant: int
    mod_N: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            exponent=self.exponent_bitsize,
            x=self.x_bitsize
        )

    def decompose_from_registers(self):
        bb = BloqBuilder(self.registers)
        exponent, x = bb.get_tracers()
        ctls = list(bb.split(exponent, self.exponent_bitsize))

        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant ** 2 ** j % self.mod_N
            ctl, x = bb.add(SingleControlModMultiply(x_bitsize=self.x_bitsize, mul_constant=c),
                            control=ctls[j], x=x)
            ctls[j] = ctl

        return bb.finalize()


class DanglingT:

    def __init__(self, direction: str):
        self.direction = direction

    def __repr__(self):
        if self.direction == 'l':
            return '<|'
        if self.direction == 'r':
            return '|>'
        raise ValueError()


LeftDangle = DanglingT(direction='l')
RightDangle = DanglingT(direction='r')


@dataclass(frozen=True)
class Wire:
    left_gate: Union[BloqInstance, DanglingT]
    left_name: str
    right_gate: Union[BloqInstance, DanglingT]
    right_name: str

    @property
    def tt(self):
        return ((self.left_gate, self.left_name), (self.right_gate, self.right_name))


def _binst_id(x: BloqInstance):
    # should be fine as long as all the `i`s are unique.
    return f'{x.bloq.__class__.__name__}_{x.i}'


def _binst_in_port(x: BloqInstance, port: str):
    return f'{_binst_id(x)}:{port}:w'


def _binst_out_port(x: BloqInstance, port: str):
    return f'{_binst_id(x)}:{port}_out:e'


def _dangling_id(x: DanglingT, reg_name: str):
    # Can we collide with a binst_id? Probably not unless we have a class named
    # DanglingT_l with integer reg_name.
    return f'DanglingT_{x.direction}_{reg_name}'


class GraphDrawer:
    def __init__(self, nodes: List[BloqInstance], wires: List[Wire], registers: Registers):
        self.nodes = nodes
        self.wires = wires
        self.registers = registers

    def to_pretty(self):
        """Return a PrettyGraphDrawer version of this that overrides methods to make the
        display more pretty but less explicit."""
        return PrettyGraphDrawer(self.nodes, self.wires, self.registers)

    def get_dangle_node(self, dangle: DanglingT, reg: Register) -> pydot.Node:
        """Get a Node representing dangling indices."""
        return pydot.Node(_dangling_id(dangle, reg.name), label=f'{reg.name}', shape='plaintext')

    def add_dangles(self, graph: pydot.Graph, dangle: DanglingT) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        dang = pydot.Subgraph(rank='same')
        for reg in self.registers:
            dang.add_node(self.get_dangle_node(dangle, reg))
        graph.add_subgraph(dang)
        return graph

    def get_split_register(self, reg: SplitRegister) -> str:
        """Return a <TR> for a SplitRegister."""
        label = ''
        for i in range(reg.bitsize):
            if i == 0:
                label += f'<TR><TD rowspan="{reg.bitsize}" port="{reg.name}">in</TD><TD port="{reg.name}{i}_out">{i}</TD></TR>'
            else:
                label += f'<TR><TD port="{reg.name}{i}_out">{i}</TD></TR>'

        return label

    def get_default_register(self, reg: Register) -> str:
        """Return a <TR> for a normal Register."""
        return f'<TR><TD PORT="{reg.name}">{reg.name}</TD><TD PORT="{reg.name}_out">{reg.name}_out</TD></TR>'

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
        label += f'<tr><td colspan="2">{self.get_binst_header_text(binst)}</td></tr>'
        for reg in binst.bloq.registers:
            if isinstance(reg, SplitRegister):
                label += self.get_split_register(reg)
            else:
                label += self.get_default_register(reg)
        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        graph.add_node(pydot.Node(_binst_id(binst), label=label, shape='plain'))
        return graph

    def add_wire(self, graph: pydot.Graph, wire: Wire) -> pydot.Graph:
        (lg, ln), (rg, rn) = wire.tt
        if isinstance(lg, DanglingT):
            graph.add_edge(pydot.Edge(_dangling_id(lg, ln), _binst_in_port(rg, rn)))
        elif isinstance(rg, DanglingT):
            graph.add_edge(pydot.Edge(_binst_out_port(lg, ln), _dangling_id(rg, ln)))
        else:
            graph.add_edge(pydot.Edge(_binst_out_port(lg, ln), _binst_in_port(rg, rn)))

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
        return f'<font point-size="10">{binst.bloq.pretty_name()}</font>'

    def get_default_register(self, reg: Register) -> str:
        if reg.name == 'control':
            celllab = '\u2b24'
        else:
            celllab = reg.name

        label = f'<TR><TD PORT="{reg.name}">{celllab}</TD><TD PORT="{reg.name}_out">..</TD></TR>'
        return label


class _QuantumGraphBuilder:
    def __init__(self):
        self.nodes = []
        self.wires = []
        self.left_gate = {}
        self.splits = {}

    def wire_up(self, from_reg_name: str, gate: Bloq, to_reg_name: str):
        if gate not in self.nodes:
            self.nodes.append(gate)

        ma = re.match(r'(\w+)\[(\d+):(\d+)]', from_reg_name)
        if ma is not None:
            self.split(ma.group(1), int(ma.group(2)), int(ma.group(3)))

        # self.wires.append(Wire(self.left_gate[from_reg_name], from_reg_name, gate, to_reg_name))
        # del self.left_gate[from_reg_name]
        # self.left_gate[to_reg_name] = gate

    def split(self, tracer: 'QubitTracer', start: int, stop: int):
        return
        if tracer not in self.splits:
            split = Split(tracer.bitsize)
            self.splits[tracer] = split
            self.nodes.append(split)
            self.wires.append(Wire(self.left_gate[tracer.name], tracer.name, split, 'x'))
        else:
            split = self.splits[tracer]

        self.left_gate[f'{tracer.name}[{start}:{stop}]'] = split


TRACER_NUM = 0


class QubitTracer(Sequence[cirq.Qid], cirq.Qid):
    def __init__(self, name: str, bitsize: int, graph_builder: _QuantumGraphBuilder):
        global TRACER_NUM
        self.name = name
        self.bitsize = bitsize
        self._gb = graph_builder
        self._tid = TRACER_NUM
        TRACER_NUM += 1

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> cirq.Qid:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[cirq.Qid]:
        ...

    def __getitem__(self, index: int) -> cirq.Qid:
        if not isinstance(index, slice):
            if index >= self.bitsize:
                raise IndexError()

            return self[index:index + 1]

        if not (index.step is None or index.step == 1):
            raise NotImplementedError()

        start = index.start if index.start is not None else 0
        return QubitTracer(name=f'{self.name}[{start}:{index.stop}]', bitsize=index.stop - start,
                           graph_builder=self._gb)

    def _comparison_key(self) -> Any:
        return id(self)

    @property
    def dimension(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.bitsize

    def wire_up(self, gate: Bloq, reg_name: str):
        # I'm being plugged in
        print(self.name, gate, reg_name)
        self._gb.wire_up(self.name, gate, reg_name)

    def __eq__(self, other):
        if not isinstance(other, QubitTracer):
            return False

        return other._tid == self._tid

    def __hash__(self):
        return hash(self._tid)

# from cirq_qubitization.shor.mod_multiply import ModMultiply
# gate = ModMultiply(exponent_bitsize=3, x_bitsize=3, mul_constant=123, mod_N=5)
# g = cq_testing.GateHelper(gate)
#
# gb = _QuantumGraphBuilder()
# gb.left_gate = {reg.name: LeftDangle for reg in gate.registers}
# tracer_qubits = {reg.name: QubitTracer(reg.name, reg.bitsize, graph_builder=gb) for reg in gate.registers}
# op = list(gate.decompose_from_registers(**tracer_qubits))
