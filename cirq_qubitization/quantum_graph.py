import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List, Dict, overload, Any

import cirq

from cirq_qubitization import MultiTargetCSwap
from cirq_qubitization.atoms import Split, Join
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers
import networkx as nx
from collections import defaultdict
import cirq_qubitization.testing as cq_testing


class DanglingT:
    def __repr__(self):
        return '..'


LeftDangle = DanglingT()
RightDangle = DanglingT()


@dataclass(frozen=True)
class Wire:
    left_gate: Union[GateWithRegisters, DanglingT]
    left_name: str
    right_gate: Union[GateWithRegisters, DanglingT]
    right_name: str

    @property
    def tt(self):
        ln = 'x' if isinstance(self.left_gate, Split) else self.left_name
        rn = 'x' if isinstance(self.right_gate, Join) else self.right_name
        return ((self.left_gate, ln), (self.right_gate, rn))


class QuantumGraph:
    def __init__(self, nodes: List[GateWithRegisters], wires: List[Wire]):
        self.nodes = nodes
        self.wires = wires

    def graphviz(self):
        import pydot
        i_by_prefix = defaultdict(lambda: 0)
        saved = {}

        def gid(gate, regname):
            pot = saved.get((gate, regname), None)
            if pot is not None:
                return pot

            gname = gate.__class__.__name__
            i = i_by_prefix[gname, regname]
            i_by_prefix[gname, regname] += 1
            name = f'{gname}_{i}_{regname}'
            saved[gate, regname] = name
            return name

        graph = pydot.Dot('qual', graph_type='digraph', rankdir='LR')

        dang = pydot.Subgraph(rank='same')
        for yi, r in enumerate(g.r):
            dang.add_node(pydot.Node(gid(LeftDangle, r.name), label=f'{r.name}', shape='plaintext'))
        graph.add_subgraph(dang)

        for xi, gate in enumerate(self.nodes):
            if isinstance(gate, Split):
                graph.add_node(
                    pydot.Node(gid(gate, ''), shape='triangle', label='', orientation=90))
                continue
            if isinstance(gate, Join):
                graph.add_node(
                    pydot.Node(gid(gate, ''), shape='triangle', label='', orientation=-90))
                continue

            ports = [f'<{r.name}>{r.name}' for r in gate.registers]
            label = '<<TABLE BORDER="1" CELLBORDER="0" CeLLSPACING="0">'
            label += f'<tr><td><font point-size="10">{gate.pretty_name()}</font></td></tr>'
            for r in gate.registers:
                if r.name == 'control':
                    celllab = '\u2b24'
                else:
                    celllab = r.name

                label += f'<TR><TD PORT="{r.name}">{celllab}</TD></TR>'
            label += '</TABLE>>'

            graph.add_node(pydot.Node(gid(gate, ''), label=label, shape='plain'))

        dang = pydot.Subgraph(rank='same')
        for yi, r in enumerate(g.r):
            dang.add_node(
                pydot.Node(gid(RightDangle, r.name), label=f'{r.name}', shape='plaintext'))
        graph.add_subgraph(dang)

        for wire in self.wires:
            (lg, ln), (rg, rn) = wire.tt
            if isinstance(lg, DanglingT):
                graph.add_edge(pydot.Edge(gid(lg, ln), gid(rg, '') + ':' + rn))
            elif isinstance(rg, DanglingT):
                graph.add_edge(pydot.Edge(gid(lg, '') + ':' + ln, gid(rg, rn)))
            else:
                graph.add_edge(
                    pydot.Edge(gid(lg, '') + ':' + ln, gid(rg, '') + ':' + rn, arrowhead='none'))

        return graph


class _QuantumGraphBuilder:
    def __init__(self):
        self.nodes = []
        self.wires = []
        self.left_gate = {}
        self.splits = {}

    def wire_up(self, from_reg_name: str, gate: GateWithRegisters, to_reg_name: str):
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

    def wire_up(self, gate: GateWithRegisters, reg_name: str):
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
