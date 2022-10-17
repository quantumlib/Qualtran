from dataclasses import dataclass
from functools import cached_property
from typing import Union, Sequence, TYPE_CHECKING, Dict

from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.fancy_registers import Soquets, CustomRegister, Side

from attrs import frozen

import itertools

import numpy as np

from cirq_qubitization.quantum_graph.quantum_graph import Connection, Wire, LeftDangle


def test_split_logic():
    n = 5
    binst = None
    wire_map = {
        'sss': Wire(
            LeftDangle, CustomRegister(name='my_in', bitsize=n, wireshape=tuple(), side=Side.RIGHT)
        )
    }
    soquets = Soquets(
        [
            CustomRegister(name='sss', bitsize=n, wireshape=tuple(), side=Side.LEFT),
            CustomRegister(name='sss', bitsize=1, wireshape=(n,), side=Side.RIGHT),
        ]
    )

    cxns = []
    for soq in soquets.lefts():
        # if we want fancy indexing (which we do), we need numpy
        # this also supports length-zero indexing natively, which is good too.
        in_wires = np.asarray(wire_map[soq.name])
        for li in soq.wire_idxs():
            cxn = Connection(in_wires[li], Wire(binst, soq, idx=li))
            cxns.append(cxn)

    out_wires = []
    for soq in soquets.rights():
        out = np.empty(soq.wireshape, dtype=object)
        for ri in soq.wire_idxs():
            out[ri] = Wire(binst, soq, idx=ri)
        out_wires.append(out)

    ret = tuple(out_wires)
    print(ret)
