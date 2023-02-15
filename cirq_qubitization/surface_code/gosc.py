import itertools
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import networkx as nx
import numpy as np
import quimb
import quimb.tensor as qtn
from attrs import frozen


@frozen
class Bloq:
    name: str
    registers: Tuple[str, ...]


class Binst:
    def __init__(self, bloq):
        self.bloq = bloq

    def __repr__(self):
        return f'{self.bloq.name}_{id(self)}'


class Soquet:
    def __init__(self, binst, name):
        self.binst = binst
        self.name = name
        assert name in binst.bloq.registers

    def __repr__(self):
        return f'{self.binst.bloq.name}_{self.name}_{id(self)}'


ZERO = Bloq('0', registers=('q',))
H = Bloq('H', registers=('q',))
CNOT = Bloq('CNOT', registers=('c', 't'))
DANGLE = Bloq('Dangle', registers=('q',))


class Builder:
    def __init__(self):
        self.g = nx.DiGraph()

    def add(self, bloq: Bloq, **args):
        binst = Binst(bloq)
        outs = {reg_name: Soquet(binst, reg_name) for reg_name in bloq.registers}

        for reg_name, fr_soq in args.items():
            self.g.add_edge(fr_soq, outs[reg_name])

        outs = [outs[reg_name] for reg_name in bloq.registers]
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


def bell_state_soqgraph():
    g = Builder()
    q0, q1 = [Soquet(Binst(ZERO), 'q') for _ in range(2)]

    q0 = g.add(H, q=q0)
    q0, q1 = g.add(CNOT, c=q0, t=q1)

    g.add(DANGLE, q=q0)
    g.add(DANGLE, q=q1)

    return g.g


def bell_state_soqgraph_old():
    g = nx.DiGraph()
    q0, q1 = [Soquet(Binst(ZERO), 'q') for _ in range(2)]

    h_binst = Binst(H)
    tmp = Soquet(h_binst, 'q')
    g.add_edge(q0, tmp)
    q0 = tmp

    cnot = Binst(CNOT)
    c_tmp = Soquet(cnot, 'c')
    t_tmp = Soquet(cnot, 't')

    g.add_edge(q0, c_tmp)
    g.add_edge(q1, t_tmp)
    q0 = c_tmp
    q1 = t_tmp

    g.add_edge(q0, Soquet(Binst(DANGLE), 'q'))
    g.add_edge(q1, Soquet(Binst(DANGLE), 'q'))
    return g


def soqgraph_to_binstgraph(g: nx.DiGraph):
    bg = nx.DiGraph()
    for edge in g.edges:
        bedge = edge[0].binst, edge[1].binst
        if bedge in bg.edges:
            bg.edges[bedge]['cxns'].append(edge)
        else:
            bg.add_edge(*bedge, cxns=[edge])

    return bg


TN_DATA = {'0': [1, 0], 'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2)}

CNOT_RAW_DATA = (
    np.kron([[1, 0], [0, 0]], np.eye(2)) + np.kron([[0, 0], [0, 1]], [[0, 1], [1, 0]])
).reshape((2,) * 4)

COPY = [1.0, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2))


def test_cnot_factor():
    tn = qtn.TensorNetwork(
        [
            qtn.Tensor(data=COPY, inds=('cin', 'int', 'cout')),
            qtn.Tensor(data=XOR, inds=('tin', 'int', 'tout')),
        ]
    )
    cnot = tn.to_dense(('cout', 'tout'), ('cin', 'tin'))


def binstgraph_to_quimb(bg: nx.DiGraph, pos=None):
    tn = qtn.TensorNetwork([])
    fix = {}

    for gen in nx.topological_generations(bg):
        print('-' * 20)
        for binst in gen:
            print(binst)
            incoming = []
            for pred in bg.pred[binst]:
                cxns = bg.edges[pred, binst]['cxns']
                print(' ', cxns)
                incoming.extend(cxns)

            print(' ', '-' * 3)
            outgoing = []
            for suc in bg.succ[binst]:
                cxns = bg.edges[binst, suc]['cxns']
                print(' ', cxns)
                outgoing.extend(cxns)

            if binst.bloq.name == 'Dangle':
                continue

            if binst.bloq.name in TN_DATA:
                fr_inds = tuple(fr for fr, me in incoming)
                to_inds = tuple(me for me, to in outgoing)
                tn.add(
                    qtn.Tensor(
                        data=TN_DATA[binst.bloq.name],
                        inds=to_inds + fr_inds,
                        tags=[binst.bloq.name, binst],
                    )
                )
                if pos is not None:
                    fix[tuple([binst])] = pos[binst]
                continue

            if binst.bloq.name == 'CNOT':
                fr_map = {}
                to_map = {}
                for fr, me in incoming:
                    # map frmo register name to soquet.
                    fr_map[me.name] = fr
                    to_map[me.name] = me

                internal = qtn.rand_uuid()
                tn.add(
                    qtn.Tensor(
                        data=COPY, inds=(fr_map['c'], to_map['c'], internal), tags=['COPY', binst]
                    )
                )
                tn.add(
                    qtn.Tensor(data=XOR, inds=(fr_map['t'], to_map['t'], internal), tags=['XOR'])
                )
                if pos is not None:
                    fix[tuple([binst])] = pos[binst]
                continue

            raise ValueError()

    return tn, fix
