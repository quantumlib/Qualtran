import itertools

from attrs import frozen, define, Factory
from typing import *
import networkx as nx


@define
class Circuit0:
    ops: List['Op_T']


Op_T = Tuple[str, Tuple[int, ...]]


@frozen
class Operation1:
    gate: str
    invars: Tuple['QVar', ...]
    outvars: Tuple['QVar', ...]


@define
class Circuit1:
    ops: List[Operation1] = Factory(list)


class QVar:
    def __repr__(self):
        return f'q{hex(id(self))[2:]}'


def aaa():
    ctrl, trg = QVar(), QVar()
    ctrl2, trg2 = QVar(), QVar()
    ctrl3, trg3 = QVar(), QVar()

    c = Circuit1([
        Operation1('cnot', invars=(ctrl, trg), outvars=(ctrl2, trg2)),
        Operation1('cnot', invars=(ctrl2, trg2), outvars=(ctrl3, trg3)),
    ])


def bbb():
    a, b = QVar(), QVar()
    a2 = QVar()
    a3, b3 = QVar(), QVar()

    c = Circuit1([
        Operation1('H', invars=(a,), outvars=(a2,)),
        Operation1('CNOT', invars=(a2, b), outvars=(a3, b3)),
    ])


@frozen
class Operation:
    gate: 'Gate'
    invars: Tuple['QVar', ...]
    outvars: Tuple['QVar', ...]

    def __str__(self):
        s = f'{self.gate}\n'
        for invar, outvar in itertools.zip_longest(self.invars, self.outvars):
            s += f'  {invar} -> {outvar}\n'
        return s


@frozen
class Gate:
    name: str
    n_in: int
    n_out: int

    def __str__(self):
        return self.name


CNOT = Gate('cnot', 2, 2)


@define
class Circuit2:
    ops: List[Operation] = Factory(list)

    def add(self, gate: Gate, *ins: QVar):
        outs = tuple(QVar() for _ in range(gate.n_out))
        op = Operation(gate, ins, outs)
        self.ops.append(op)
        return outs

    def __str__(self):
        return '\n'.join(str(op) for op in self.ops)


def ex0_0():
    c = Circuit2()
    qvar = QVar()
    c.add(CNOT, qvar)


def validate0(c: Circuit2):
    for op in c.ops:
        assert len(op.invars) == op.gate.n_in
        assert len(op.outvars) == op.gate.n_out


def ex0_1():
    c = Circuit2()
    qvar = QVar()
    c.add(CNOT, qvar)
    validate0(c)  # throws


def ex0_2():
    c = Circuit2()
    ctrl, trg = QVar(), QVar()
    ctrl, trg = c.add(CNOT, ctrl, trg)
    validate0(c)
    print(c)


@define
class Circuit:
    ops: List[Operation] = Factory(list)
    n_in: int = 0
    n_out: int = 0

    def add(self, gate: Gate, *ins: QVar):
        outs = tuple(QVar() for _ in range(gate.n_out))
        op = Operation(gate, ins, outs)
        self.ops.append(op)
        return outs

    def add_in_reg(self):
        q = None
        self.n_in += 1
        return q

    def add_out_reg(self, q: QVar):
        self.n_out += 1

    def print(self):
        for i, op in enumerate(self.ops):
            print(i, op.gate)
            for invar, outvar in itertools.zip_longest(op.invars, op.outvars, fillvalue=None):
                print('  ', invar, '->', outvar)


def validate(c: Circuit):
    frontier: Set[QVar] = set(c.invars)

    for op in c.ops:
        assert len(op.invars) == op.gate.n_in
        assert len(op.outvars) == op.gate.n_out

        for ivar in op.invars:
            assert ivar in frontier
            frontier.remove(ivar)

        for ovar in op.outvars:
            frontier.add(ovar)

    for ovar in c.outvars:
        frontier.remove(ovar)

    assert len(frontier) == 0


def cnot_id():
    c = Circuit()
    a = c.add_in_reg()
    b = c.add_in_reg()
    a, b = c.add(CNOT, a, b)
    a, b = c.add(CNOT, a, b)
    c.add_out_reg(a)
    c.add_out_reg(b)
    c.print()
    return c


def draw(program2):
    g = nx.DiGraph()
    pos = {}

    for i, op in enumerate(program2):
        for j, invar in enumerate(op.invars):
            g.add_node((i, 'in', j))
            pos[(i, 'in', j)] = (i, j)

            for i2, op2 in enumerate(program2):
                for j2, prev_out in enumerate(op2.outvars):
                    if invar == prev_out:
                        g.add_edge((i2, 'out', j2), (i, 'in', j))

        for j, outvar in enumerate(op.outvars):
            g.add_node((i, 'out', j))
            pos[(i, 'out', j)] = (i + 0.5, j)

        for j, (_, _) in enumerate(zip(op.invars, op.outvars)):
            g.add_edge((i, 'in', j), (i, 'out', j), kind='intraop')

    nx.draw_networkx(g, pos=pos)
