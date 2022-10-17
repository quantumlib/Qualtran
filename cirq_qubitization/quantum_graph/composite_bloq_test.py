import cirq
import networkx as nx

from cirq_qubitization.gate_with_registers import Registers

from cirq_qubitization.quantum_graph.bloq_test import TestBloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, _create_binst_graph
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    Wire,
    LeftDangle,
    RightDangle,
)


def _manually_make_test_cbloq_wires():
    tb = TestBloq()
    binst1 = BloqInstance(tb, 1)
    binst2 = BloqInstance(tb, 2)
    assert binst1 != binst2
    return [
        Connection(Wire(LeftDangle, 'q1'), Wire(binst1, 'control')),
        Connection(Wire(LeftDangle, 'q2'), Wire(binst1, 'target')),
        Connection(Wire(binst1, 'control'), Wire(binst2, 'target')),
        Connection(Wire(binst1, 'target'), Wire(binst2, 'control')),
        Connection(Wire(binst2, 'control'), Wire(RightDangle, 'q1')),
        Connection(Wire(binst2, 'target'), Wire(RightDangle, 'q2')),
    ]


def test_create_binst_graph():
    wires = _manually_make_test_cbloq_wires()
    binst1 = wires[2].left.binst
    binst2 = wires[2].right.binst
    binst_graph = _create_binst_graph(wires)

    binst_generations = list(nx.topological_generations(binst_graph))
    assert binst_generations == [[LeftDangle], [binst1], [binst2], [RightDangle]]


def test_composite_bloq():
    wires = _manually_make_test_cbloq_wires()
    cbloq = CompositeBloq(wires=wires, registers=Registers.build(q1=1, q2=1))
    print()
    circuit = cbloq.to_cirq_circuit(q1=[cirq.LineQubit(1)], q2=[cirq.LineQubit(2)])
    cirq.testing.assert_has_diagram(
        circuit,
        desired="""\
1: ───@───X───
      │   │
2: ───X───@─── \
    """,
    )
    print()
