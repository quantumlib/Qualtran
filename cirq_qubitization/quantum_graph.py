import abc
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List, Dict, overload, Any

import cirq

from cirq_qubitization import MultiTargetCSwap
from cirq_qubitization.atoms import Join
from cirq_qubitization.gate_with_registers import Registers, SplitRegister
import networkx as nx
from collections import defaultdict
import cirq_qubitization.testing as cq_testing

class LeftRightT:
    pass

Left = LeftRightT()
Right = LeftRightT()


@dataclass(frozen=True)
class Port:
    name: str
    lr: LeftRightT

class Bloq(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def ports(self):
        for reg in self.registers:
            yield Port(reg.name, Left)
            yield Port(reg.name, Right)

    def pretty_name(self) -> str:
        return self.__class__.__name__


class CompositeBloq(Bloq):
    def __init__(self, wires: Sequence['Wire'], provenance=None):
        self._wires = wires
        self._provenance = provenance
        self._registers = Registers.build() # todo

    @property
    def registers(self) -> Registers:
        return self._registers


@dataclass(frozen=True)
class BloqInstance:
    bloq: Bloq
    i: int

    def __repr__(self):
        return f'{self.bloq!r}[{self.i}]'


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
        return Registers([SplitRegister(name='xy', bitsize=self.bitsize)])


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
        inst = BloqInstance(Split(n), self.i)
        self.i += 1

        fr_name = x.name
        self._wires.append(Wire(self._prev_inst[fr_name], fr_name, inst, 'x'))
        print('wire_add', self._wires[-1])
        del self._prev_inst[fr_name]
        new_reg_objs = tuple(RegObj(f'y{i}') for i in range(n))
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

    def __init__(self, direction:str):
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


class QuantumGraph:
    def __init__(self, nodes: List[BloqInstance], wires: List[Wire], registers:Registers):
        self.nodes = nodes
        self.wires = wires
        self.registers = registers

    def graphviz(self):
        import pydot

        def gid(x: BloqInstance):
            return f'{x.bloq.pretty_name()}_{x.i}'

        def did(x: DanglingT, reg_name:str):
            return f'DanglingT_{x.direction}_{reg_name}'

        def splittable(n):
            label = ''
            for i in range(n):
                if i == 0:
                    label += f'<TR><TD rowspan="3" port="x">in</TD><TD port="y{i}_out">{i}</TD></TR>'
                else:
                   label += f'<TR><TD port="y{i}_out">{i}</TD></TR>'

            # label = '<TR><TD>in</TD><TD><TABLE><TR><TD>1</TD></TR><TR><TD>2</TD></TR></TABLE></TD></TR>'
            return label

        graph = pydot.Dot('qual', graph_type='digraph', rankdir='LR')

        dang = pydot.Subgraph(rank='same')
        for yi, r in enumerate(self.registers):
            dang.add_node(pydot.Node(did(LeftDangle, r.name), label=f'{r.name}', shape='plaintext'))
        graph.add_subgraph(dang)

        for xi, binst in enumerate(self.nodes):
            # if isinstance(binst.bloq, Split):
            #     graph.add_node(
            #         pydot.Node(gid(binst), shape='triangle', label='', orientation=90))
            #     continue
            # if isinstance(binst.bloq, Join):
            #     graph.add_node(
            #         pydot.Node(gid(binst), shape='triangle', label='', orientation=-90))
            #     continue

            label = '<<TABLE BORDER="1" CELLBORDER="1" CeLLSPACING="3">'
            label += f'<tr><td colspan="2"><font point-size="10">{binst.bloq.pretty_name()}</font></td></tr>'
            for r in binst.bloq.registers:
                if r.name == 'control':
                    celllab = '\u2b24'
                else:
                    celllab = r.name

                if isinstance(r, SplitRegister):
                    label += splittable(r.bitsize)
                else:
                    label += f'<TR><TD PORT="{r.name}">{celllab}</TD><TD PORT="{r.name}_out">..</TD></TR>'
            label += '</TABLE>>'

            graph.add_node(pydot.Node(gid(binst), label=label, shape='plain'))

        dang = pydot.Subgraph(rank='same')
        for yi, r in enumerate(self.registers):
            dang.add_node(
                pydot.Node(did(RightDangle, r.name), label=f'{r.name}', shape='plaintext'))
        graph.add_subgraph(dang)

        for wire in self.wires:
            (lg, ln), (rg, rn) = wire.tt
            if isinstance(lg, DanglingT):
                graph.add_edge(pydot.Edge(did(lg, ln), gid(rg) + ':' + rn))
            elif isinstance(rg, DanglingT):
                graph.add_edge(pydot.Edge(gid(lg) + ':' + ln, did(rg, ln)))
            else:
                graph.add_edge(
                    pydot.Edge(gid(lg) + ':' + ln+'_out' + ':e', gid(rg) + ':' + rn +':w'))

        return graph


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
