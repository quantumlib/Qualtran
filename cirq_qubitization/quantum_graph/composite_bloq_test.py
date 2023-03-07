from functools import cached_property
from typing import Dict

import cirq
import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_test import TestBloq
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
    LeftDangle,
    RightDangle,
    Soquet,
)


def _manually_make_test_cbloq_cxns():
    regs = FancyRegisters.build(q1=1, q2=1)
    q1, q2 = regs
    tb = TestBloq()
    control, target = tb.registers
    binst1 = BloqInstance(tb, 1)
    binst2 = BloqInstance(tb, 2)
    assert binst1 != binst2
    return [
        Connection(Soquet(LeftDangle, q1), Soquet(binst1, control)),
        Connection(Soquet(LeftDangle, q2), Soquet(binst1, target)),
        Connection(Soquet(binst1, control), Soquet(binst2, target)),
        Connection(Soquet(binst1, target), Soquet(binst2, control)),
        Connection(Soquet(binst2, control), Soquet(RightDangle, q1)),
        Connection(Soquet(binst2, target), Soquet(RightDangle, q2)),
    ], regs


class TestComposite(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(q1=1, q2=2)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', q1: 'Soquet', q2: 'Soquet'
    ) -> Dict[str, SoquetT]:
        q1, q2 = bb.add(TestBloq(), control=q1, target=q2)
        q1, q2 = bb.add(TestBloq(), control=q2, target=q1)
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
BloqInstance(bloq=TestBloq(), i=1)
  LeftDangle.q1 -> control
  LeftDangle.q2 -> target
  control -> BloqInstance(bloq=TestBloq(), i=2).target
  target -> BloqInstance(bloq=TestBloq(), i=2).control
--------------------
BloqInstance(bloq=TestBloq(), i=2)
  BloqInstance(bloq=TestBloq(), i=1).control -> target
  BloqInstance(bloq=TestBloq(), i=1).target -> control
  control -> RightDangle.q1
  target -> RightDangle.q2"""
    )

    assert len(list(cbloq.iter_bloqnections())) == len(cbloq.bloq_instances)
    for binst, preds, succs in cbloq.iter_bloqnections():
        assert isinstance(binst, BloqInstance)
        assert len(preds) > 0
        assert len(succs) > 0


def test_bb_composite_bloq():
    cbloq_auto = TestComposite().decompose_bloq()
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
    x, y = bb.add(TestBloq(), control=x, target=y)

    x, y = bb.add(TestBloq(), control=x, target=y)

    cbloq = bb.finalize(x=x, y=y)

    inds = {binst.i for binst in cbloq.bloq_instances}
    assert len(inds) == 2
    assert len(cbloq.bloq_instances) == 2


def _get_bb():
    bb = CompositeBloqBuilder()
    x = bb.add_register('x', 1)
    y = bb.add_register('y', 1)
    return bb, x, y


def test_wrong_soquet():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*target.*'):
        bad_target_arg = Soquet(BloqInstance(TestBloq(), i=12), FancyRegister('target', 2))
        bb.add(TestBloq(), control=x, target=bad_target_arg)


def test_double_use_1():
    bb, x, y = _get_bb()

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `TestBloq.*target`.*'
    ):
        bb.add(TestBloq(), control=x, target=x)


def test_double_use_2():
    bb, x, y = _get_bb()

    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `TestBloq\(\)\.control`\.'
    ):
        x3, y3 = bb.add(TestBloq(), control=x, target=y)


def test_missing_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*requires a Soquet named `control`.'):
        bb.add(TestBloq(), target=y)


def test_too_many_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqBuilderError, match=r'.*does not accept Soquets.*another_control.*'):
        bb.add(TestBloq(), control=x, target=y, another_control=x)


def test_finalize_wrong_soquet():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)
    assert x != x2
    assert y != y2

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=Soquet(BloqInstance(TestBloq(), i=12), FancyRegister('target', 2)))


def test_finalize_double_use_1():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=x2)


def test_finalize_double_use_2():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(
        BloqBuilderError, match=r'.*is not an available Soquet for `RightDangle\.x`\.'
    ):
        bb.finalize(x=x, y=y2)


def test_finalize_missing_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'Finalizing requires a Soquet named `x`.'):
        bb.finalize(y=y2)


def test_finalize_strict_too_many_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    bb.add_register_allowed = False
    with pytest.raises(BloqBuilderError, match=r'Finalizing does not accept Soquets.*z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, FancyRegister('asdf', 1)))


def test_finalize_bad_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)

    with pytest.raises(BloqBuilderError, match=r'.*is not an available Soquet.*RightDangle\.z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, FancyRegister('asdf', 1)))


def test_finalize_alloc():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestBloq(), control=x, target=y)
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
                control, target[i, j] = bb.add(TestBloq(), control=control, target=target[i, j])

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
