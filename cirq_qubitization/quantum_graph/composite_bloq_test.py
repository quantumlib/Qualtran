from functools import cached_property
from typing import Dict

import cirq
import networkx as nx
import pytest
from attrs import frozen

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_test import TestBloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    _create_binst_graph,
    CompositeBloqBuilder,
    BloqBuilderError,
)
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Wire,
    Soquet,
    LeftDangle,
    RightDangle,
)


def _manually_make_test_cbloq_wires():
    tb = TestBloq()
    binst1 = BloqInstance(tb, 1)
    binst2 = BloqInstance(tb, 2)
    assert binst1 != binst2
    return [
        Wire(Soquet(LeftDangle, 'q1'), Soquet(binst1, 'control')),
        Wire(Soquet(LeftDangle, 'q2'), Soquet(binst1, 'target')),
        Wire(Soquet(binst1, 'control'), Soquet(binst2, 'target')),
        Wire(Soquet(binst1, 'target'), Soquet(binst2, 'control')),
        Wire(Soquet(binst2, 'control'), Soquet(RightDangle, 'q1')),
        Wire(Soquet(binst2, 'target'), Soquet(RightDangle, 'q2')),
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


@frozen
class TestRepBloq(Bloq):
    n_reps: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(x1=1, x2=1)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', x1: 'Soquet', x2: 'Soquet'
    ) -> Dict[str, 'Soquet']:

        for _ in range(self.n_reps):
            x1, x2 = bb.add(TestBloq(), control=x1, target=x2)


def test_bloq_builder():
    registers = Registers.build(x=1, y=1)
    bb = CompositeBloqBuilder(registers)
    initial_soqs = bb.initial_soquets()
    assert initial_soqs == {'x': Soquet(LeftDangle, 'x'), 'y': Soquet(LeftDangle, 'y')}

    x = initial_soqs['x']
    y = initial_soqs['y']
    x, y = bb.add(TestBloq(), control=x, target=y)

    # the next assertion is sortof an implementation detail... these returned
    # soquets should be pretty opaque to the user.
    assert x == Soquet(BloqInstance(TestBloq(), i=0), 'control')

    x, y = bb.add(TestBloq(), control=x, target=y)
    assert x == Soquet(BloqInstance(TestBloq(), i=1), 'control')

    cbloq = bb.finalize(x=x, y=y)

    inds = {binst.i for binst in cbloq.bloq_instances}
    assert len(inds) == 2
    assert len(cbloq.bloq_instances) == 2


def _get_bb():
    registers = Registers.build(x=1, y=1)
    bb = CompositeBloqBuilder(registers)
    initial_soqs = bb.initial_soquets()
    x = initial_soqs['x']
    y = initial_soqs['y']
    return bb, x, y


def test_wrong_soquet():
    bb, x, y = _get_bb()

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available input Soquet for .*target.*'
    ):
        bb.add(TestBloq(), control=x, target=Soquet(BloqInstance(TestBloq(), i=12), 'target'))


def test_double_use_1():
    bb, x, y = _get_bb()

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available input Soquet for .*target.*'
    ):
        bb.add(TestBloq(), control=x, target=x)


def test_double_use_2():
    bb, x, y = _get_bb()

    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available input Soquet for .*control.*'
    ):
        x3, y3 = bb.add(TestBloq(), control=x, target=y)


def test_missing_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*requires an input Soquet named `control`.'):
        bb.add(TestBloq(), target=y)


def test_too_many_args():
    bb, x, y = _get_bb()

    with pytest.raises(
        BloqBuilderError, match=r'.*does not accept input Soquets.*another_control.*'
    ):
        bb.add(TestBloq(), control=x, target=y, another_control=x)


def test_finalize_wrong_soquet():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)
    assert x != x2
    assert y != y2

    with pytest.raises(BloqBuilderError, match=r'.*is not an available final Soquet for .*y.*'):
        bb.finalize(x=x2, y=Soquet(BloqInstance(TestBloq(), i=12), 'target'))


def test_finalize_double_use_1():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available final Soquet for .*y.*'):
        bb.finalize(x=x2, y=x2)


def test_finalize_double_use_2():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available final Soquet for .*x.*'):
        bb.finalize(x=x, y=y2)


def test_finalize_missing_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*requires a final Soquet named `x`.'):
        bb.finalize(y=y2)


def test_finalize_too_many_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*does not accept final Soquet.*z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, 'asdf'))
