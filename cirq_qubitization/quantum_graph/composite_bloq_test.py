from functools import cached_property
from typing import Dict

import attrs
import cirq
import networkx as nx
import numpy as np
import pytest
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_test import TestCNOT
from cirq_qubitization.quantum_graph.composite_bloq import (
    _create_binst_graph,
    BloqBuilderError,
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


def _manually_make_test_cbloq_cxns():
    regs = FancyRegisters.build(q1=1, q2=1)
    q1, q2 = regs
    tcn = TestCNOT()
    control, target = tcn.registers
    binst1 = BloqInstance(tcn, 1)
    binst2 = BloqInstance(tcn, 2)
    assert binst1 != binst2
    return [
        Connection(Soquet(LeftDangle, q1), Soquet(binst1, control)),
        Connection(Soquet(LeftDangle, q2), Soquet(binst1, target)),
        Connection(Soquet(binst1, control), Soquet(binst2, target)),
        Connection(Soquet(binst1, target), Soquet(binst2, control)),
        Connection(Soquet(binst2, control), Soquet(RightDangle, q1)),
        Connection(Soquet(binst2, target), Soquet(RightDangle, q2)),
    ], regs


class TestTwoCNOT(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(q1=1, q2=2)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', q1: 'Soquet', q2: 'Soquet'
    ) -> Dict[str, SoquetT]:
        q1, q2 = bb.add(TestCNOT(), control=q1, target=q2)
        q1, q2 = bb.add(TestCNOT(), control=q2, target=q1)
        return {'q1': q1, 'q2': q2}


def test_create_binst_graph():
    cxns, regs = _manually_make_test_cbloq_cxns()
    binst1 = cxns[2].left.binst
    binst2 = cxns[2].right.binst
    binst_graph = _create_binst_graph(cxns)
    assert nx.is_isomorphic(binst_graph, CompositeBloq(cxns, regs)._binst_graph)

    binst_generations = list(nx.topological_generations(binst_graph))
    assert binst_generations == [[LeftDangle], [binst1], [binst2], [RightDangle]]


def test_composite_bloq():
    cxns, regs = _manually_make_test_cbloq_cxns()
    cbloq = CompositeBloq(cxns=cxns, registers=regs)
    circuit = cbloq.to_cirq_circuit(q1=[cirq.LineQubit(1)], q2=[cirq.LineQubit(2)])
    cirq.testing.assert_has_diagram(
        circuit,
        desired="""\
1: ───@───X───
      │   │
2: ───X───@─── \
    """,
    )

    assert (
        cbloq.debug_text()
        == """\
TestCNOT()<1>
  LeftDangle.q1 -> control
  LeftDangle.q2 -> target
  control -> TestCNOT()<2>.target
  target -> TestCNOT()<2>.control
--------------------
TestCNOT()<2>
  TestCNOT()<1>.control -> target
  TestCNOT()<1>.target -> control
  control -> RightDangle.q1
  target -> RightDangle.q2"""
    )


def test_iter_bloqnections():
    cbloq = TestTwoCNOT().decompose_bloq()
    assert len(list(cbloq.iter_bloqnections())) == len(cbloq.bloq_instances)
    for binst, preds, succs in cbloq.iter_bloqnections():
        assert isinstance(binst, BloqInstance)
        assert len(preds) > 0
        assert len(succs) > 0


def test_iter_bloqsoqs():
    cbloq = TestTwoCNOT().decompose_bloq()
    assert len(list(cbloq.iter_bloqsoqs())) == len(cbloq.bloq_instances)

    for binst, soqs in cbloq.iter_bloqsoqs():
        assert isinstance(binst, BloqInstance)
        assert sorted(soqs.keys()) == ['control', 'target']

    mapping = {binst: attrs.evolve(binst, i=100 + binst.i) for binst in cbloq.bloq_instances}
    for binst, soqs in cbloq.iter_bloqsoqs(binst_map=mapping):
        assert isinstance(binst, BloqInstance)
        for s in soqs.values():
            if not isinstance(s.binst, DanglingT):
                assert s.binst.i >= 100


def test_bb_composite_bloq():
    cbloq_auto = TestTwoCNOT().decompose_bloq()
    circuit = cbloq_auto.to_cirq_circuit(q1=[cirq.LineQubit(1)], q2=[cirq.LineQubit(2)])
    cirq.testing.assert_has_diagram(
        circuit,
        desired="""\
1: ───@───X───
      │   │
2: ───X───@─── \
    """,
    )


def test_bloq_builder():
    registers = FancyRegisters.build(x=1, y=1)
    x, y = registers
    bb, initial_soqs = CompositeBloqBuilder.from_registers(registers)
    assert initial_soqs == {'x': Soquet(LeftDangle, x), 'y': Soquet(LeftDangle, y)}

    x = initial_soqs['x']
    y = initial_soqs['y']
    x, y = bb.add(TestCNOT(), control=x, target=y)

    x, y = bb.add(TestCNOT(), control=x, target=y)

    cbloq = bb.finalize(x=x, y=y)

    inds = {binst.i for binst in cbloq.bloq_instances}
    assert len(inds) == 2
    assert len(cbloq.bloq_instances) == 2


def test_bloq_builder_add_2():
    bb = CompositeBloqBuilder()
    x = bb.add_register('x', 1)
    y = bb.add_register('y', 1)

    binst1, (x, y) = bb.add_2(TestCNOT(), control=x, target=y)
    binst2, (x, y) = bb.add_2(TestCNOT(), control=x, target=y)
    cbloq = bb.finalize(x=x, y=y)

    assert sorted(cbloq.bloq_instances, key=lambda x: x.i) == [binst1, binst2]


def _get_bb():
    bb = CompositeBloqBuilder()
    x = bb.add_register('x', 1)
    y = bb.add_register('y', 1)
    return bb, x, y


def test_wrong_soquet():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*target.*'):
        bad_target_arg = Soquet(BloqInstance(TestCNOT(), i=12), FancyRegister('target', 2))
        bb.add(TestCNOT(), control=x, target=bad_target_arg)


def test_double_use_1():
    bb, x, y = _get_bb()

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `TestCNOT.*target`.*'
    ):
        bb.add(TestCNOT(), control=x, target=x)


def test_double_use_2():
    bb, x, y = _get_bb()

    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `TestCNOT\(\)\.control`\.'
    ):
        x3, y3 = bb.add(TestCNOT(), control=x, target=y)


def test_missing_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*requires a Soquet named `control`.'):
        bb.add(TestCNOT(), target=y)


def test_too_many_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*does not accept Soquets.*another_control.*'):
        bb.add(TestCNOT(), control=x, target=y, another_control=x)


def test_finalize_wrong_soquet():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)
    assert x != x2
    assert y != y2

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=Soquet(BloqInstance(TestCNOT(), i=12), FancyRegister('target', 2)))


def test_finalize_double_use_1():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=x2)


def test_finalize_double_use_2():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `RightDangle\.x`\.'
    ):
        bb.finalize(x=x, y=y2)


def test_finalize_missing_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'Finalizing requires a Soquet named `x`.'):
        bb.finalize(y=y2)


def test_finalize_strict_too_many_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    bb.add_register_allowed = False
    with pytest.raises(BloqBuilderError, match=r'Finalizing does not accept Soquets.*z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, FancyRegister('asdf', 1)))


def test_finalize_bad_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet.*RightDangle\.z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, FancyRegister('asdf', 1)))


def test_finalize_alloc():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)
    z = bb.allocate(1)

    cbloq = bb.finalize(x=x2, y=y2, z=z)
    assert len(list(cbloq.registers.rights())) == 3


class TestMultiCNOT(Bloq):
    # A minimal test-bloq with a complicated `target` register.
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [FancyRegister('control', 1), FancyRegister('target', 1, wireshape=(2, 3))]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', control: 'Soquet', target: NDArray['Soquet']
    ) -> Dict[str, SoquetT]:
        for i in range(2):
            for j in range(3):
                control, target[i, j] = bb.add(TestCNOT(), control=control, target=target[i, j])

        return {'control': control, 'target': target}


def test_complicated_target_register():
    bloq = TestMultiCNOT()
    cbloq = bloq.decompose_bloq()
    assert len(cbloq.bloq_instances) == 2 * 3

    binst_graph = _create_binst_graph(cbloq.connections)
    # note: this includes the two `Dangling` generations.
    assert len(list(nx.topological_generations(binst_graph))) == 2 * 3 + 2

    circuit = cbloq.to_cirq_circuit(**bloq.registers.get_named_qubits())
    cirq.testing.assert_has_diagram(
        circuit,
        """\
control: ───────────@───@───@───@───@───@───
                    │   │   │   │   │   │
target[0, 0, 0]: ───X───┼───┼───┼───┼───┼───
                        │   │   │   │   │
target[0, 1, 0]: ───────X───┼───┼───┼───┼───
                            │   │   │   │
target[0, 2, 0]: ───────────X───┼───┼───┼───
                                │   │   │
target[1, 0, 0]: ───────────────X───┼───┼───
                                    │   │
target[1, 1, 0]: ───────────────────X───┼───
                                        │
target[1, 2, 0]: ───────────────────────X───""",
    )


def test_util_convenience_methods():
    bb = CompositeBloqBuilder()

    qs = bb.allocate(10)
    qs = bb.split(qs)
    qs = bb.join(qs)
    bb.free(qs)
    cbloq = bb.finalize()
    assert len(cbloq.connections) == 1 + 10 + 1


def test_util_convenience_methods_errors():
    bb = CompositeBloqBuilder()

    qs = np.asarray([bb.allocate(5), bb.allocate(5)])
    with pytest.raises(ValueError, match='.*expects a single Soquet'):
        qs = bb.split(qs)

    qs = bb.allocate(5)
    with pytest.raises(ValueError, match='.*expects a 1-d array'):
        qs = bb.join(qs)

    # but this works:
    qs = np.asarray([bb.allocate(), bb.allocate()])
    qs = bb.join(qs)

    qs = np.asarray([bb.allocate(5), bb.allocate(5)])
    with pytest.raises(ValueError, match='.*expects a single Soquet'):
        bb.free(qs)


@frozen
class Atom(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=1)


class TestSerialBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=1)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', stuff: 'SoquetT'
    ) -> Dict[str, 'Soquet']:

        for i in range(3):
            (stuff,) = bb.add(Atom(), stuff=stuff)
        return {'stuff': stuff}


@frozen
class TestParallelBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=3)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', stuff: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        stuff = bb.split(stuff)
        for i in range(len(stuff)):
            stuff[i] = bb.add(Atom(), stuff=stuff[i])[0]

        return {'stuff': bb.join(stuff)}


@pytest.mark.parametrize('cls', [TestSerialBloq, TestParallelBloq])
def test_copy(cls):
    cbloq = cls().decompose_bloq()
    cbloq2 = cbloq.copy()
    assert cbloq is not cbloq2
    assert cbloq != cbloq2
    assert cbloq.debug_text() == cbloq2.debug_text()


def test_add_from():
    bb = CompositeBloqBuilder()
    stuff = bb.add_register('stuff', 3)
    (stuff,) = bb.add(TestParallelBloq(), stuff=stuff)
    (stuff,) = bb.add_from(TestParallelBloq().decompose_bloq(), stuff=stuff)
    bloq = bb.finalize(stuff=stuff)
    assert (
        bloq.debug_text()
        == """\
TestParallelBloq()<0>
  LeftDangle.stuff -> stuff
  stuff -> Split(n=3)<1>.split
--------------------
Split(n=3)<1>
  TestParallelBloq()<0>.stuff -> split
  split[0] -> Atom()<2>.stuff
  split[1] -> Atom()<3>.stuff
  split[2] -> Atom()<4>.stuff
--------------------
Atom()<2>
  Split(n=3)<1>.split[0] -> stuff
  stuff -> Join(n=3)<5>.join[0]
Atom()<3>
  Split(n=3)<1>.split[1] -> stuff
  stuff -> Join(n=3)<5>.join[1]
Atom()<4>
  Split(n=3)<1>.split[2] -> stuff
  stuff -> Join(n=3)<5>.join[2]
--------------------
Join(n=3)<5>
  Atom()<2>.stuff -> join[0]
  Atom()<3>.stuff -> join[1]
  Atom()<4>.stuff -> join[2]
  join -> RightDangle.stuff"""
    )


def test_add_duplicate_register():
    bb = CompositeBloqBuilder()
    _ = bb.add_register('control', 1)
    y = bb.add_register('control', 2)
    with pytest.raises(ValueError):
        bb.finalize(control=y)
