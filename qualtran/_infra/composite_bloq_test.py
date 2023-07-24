#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import Dict, List, Tuple

import attrs
import cirq
import networkx as nx
import numpy as np
import pytest
from attrs import frozen
from cirq_ft import TComplexity
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    BloqError,
    BloqInstance,
    CompositeBloq,
    Connection,
    LeftDangle,
    Register,
    RightDangle,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.bloq_test import TestCNOT
from qualtran._infra.composite_bloq import _create_binst_graph, _get_dangling_soquets
from qualtran.bloqs.util_bloqs import Join
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _manually_make_test_cbloq_cxns():
    signature = Signature.build(q1=1, q2=1)
    q1, q2 = signature
    tcn = TestCNOT()
    control, target = tcn.signature
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
    ], signature


@attrs.frozen
class TestTwoCNOT(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q1=1, q2=1)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q1: 'Soquet', q2: 'Soquet'
    ) -> Dict[str, SoquetT]:
        q1, q2 = bb.add(TestCNOT(), control=q1, target=q2)
        q1, q2 = bb.add(TestCNOT(), control=q2, target=q1)
        return {'q1': q1, 'q2': q2}


def test_create_binst_graph():
    cxns, signature = _manually_make_test_cbloq_cxns()
    binst1 = cxns[2].left.binst
    binst2 = cxns[2].right.binst
    binst_graph = _create_binst_graph(cxns)
    assert nx.is_isomorphic(binst_graph, CompositeBloq(cxns, signature)._binst_graph)

    binst_generations = list(nx.topological_generations(binst_graph))
    assert binst_generations == [[LeftDangle], [binst1], [binst2], [RightDangle]]


def test_composite_bloq():
    cxns, signature = _manually_make_test_cbloq_cxns()
    cbloq = CompositeBloq(connections=cxns, signature=signature)
    circuit, _ = cbloq.to_cirq_circuit(q1=[cirq.LineQubit(1)], q2=[cirq.LineQubit(2)])
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

    for binst, isoqs, osoqs in cbloq.iter_bloqsoqs():
        assert isinstance(binst, BloqInstance)
        assert sorted(isoqs.keys()) == ['control', 'target']
        assert len(osoqs) == 2


def test_map_soqs():
    cbloq = TestTwoCNOT().decompose_bloq()
    bb, _ = BloqBuilder.from_signature(cbloq.signature)
    bb._i = 100

    soq_map: List[Tuple[SoquetT, SoquetT]] = []
    for binst, in_soqs, old_out_soqs in cbloq.iter_bloqsoqs():
        if binst.i == 0:
            assert in_soqs == bb.map_soqs(in_soqs, soq_map)
        elif binst.i == 1:
            for k, val in bb.map_soqs(in_soqs, soq_map).items():
                assert val.binst.i >= 100
        else:
            raise AssertionError()

        in_soqs = bb.map_soqs(in_soqs, soq_map)
        new_out_soqs = bb.add_t(binst.bloq, **in_soqs)
        soq_map.extend(zip(old_out_soqs, new_out_soqs))

    fsoqs = bb.map_soqs(cbloq.final_soqs(), soq_map)
    for k, val in fsoqs.items():
        assert val.binst.i >= 100
    cbloq = bb.finalize(**fsoqs)
    assert isinstance(cbloq, CompositeBloq)


def test_bb_composite_bloq():
    cbloq_auto = TestTwoCNOT().decompose_bloq()
    circuit, _ = cbloq_auto.to_cirq_circuit(q1=[cirq.LineQubit(1)], q2=[cirq.LineQubit(2)])
    cirq.testing.assert_has_diagram(
        circuit,
        desired="""\
1: ───@───X───
      │   │
2: ───X───@─── \
    """,
    )


def test_bloq_builder():
    signature = Signature.build(x=1, y=1)
    x_reg, y_reg = signature
    bb, initial_soqs = BloqBuilder.from_signature(signature)
    assert initial_soqs == {'x': Soquet(LeftDangle, x_reg), 'y': Soquet(LeftDangle, y_reg)}

    x = initial_soqs['x']
    y = initial_soqs['y']
    x, y = bb.add(TestCNOT(), control=x, target=y)

    x, y = bb.add(TestCNOT(), control=x, target=y)

    cbloq = bb.finalize(x=x, y=y)

    inds = {binst.i for binst in cbloq.bloq_instances}
    assert len(inds) == 2
    assert len(cbloq.bloq_instances) == 2


def _get_bb():
    bb = BloqBuilder()
    x = bb.add_register('x', 1)
    y = bb.add_register('y', 1)
    return bb, x, y


def test_wrong_soquet():
    bb, x, y = _get_bb()

    with pytest.raises(BloqError, match=r'.*is not an available Soquet for .*target.*'):
        bad_target_arg = Soquet(BloqInstance(TestCNOT(), i=12), Register('target', 2))
        bb.add(TestCNOT(), control=x, target=bad_target_arg)


def test_double_use_1():
    bb, x, y = _get_bb()

    with pytest.raises(BloqError, match=r'.*is not an available Soquet for `TestCNOT.*target`.*'):
        bb.add(TestCNOT(), control=x, target=x)


def test_double_use_2():
    bb, x, y = _get_bb()

    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(
        BloqError, match=r'.*is not an available Soquet for `TestCNOT\(\)\.control`\.'
    ):
        x3, y3 = bb.add(TestCNOT(), control=x, target=y)


def test_missing_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqError, match=r'.*requires a Soquet named `control`.'):
        bb.add(TestCNOT(), target=y)


def test_too_many_args():
    bb, x, y = _get_bb()

    with pytest.raises(BloqError, match=r'.*does not accept Soquets.*another_control.*'):
        bb.add(TestCNOT(), control=x, target=y, another_control=x)


def test_finalize_wrong_soquet():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)
    assert x != x2
    assert y != y2

    with pytest.raises(BloqError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=Soquet(BloqInstance(TestCNOT(), i=12), Register('target', 2)))


def test_finalize_double_use_1():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqError, match=r'.*is not an available Soquet for .*y.*'):
        bb.finalize(x=x2, y=x2)


def test_finalize_double_use_2():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqError, match=r'.*is not an available Soquet for `RightDangle\.x`\.'):
        bb.finalize(x=x, y=y2)


def test_finalize_missing_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqError, match=r'Finalizing requires a Soquet named `x`.'):
        bb.finalize(y=y2)


def test_finalize_strict_too_many_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    bb.add_register_allowed = False
    with pytest.raises(BloqError, match=r'Finalizing does not accept Soquets.*z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, Register('asdf', 1)))


def test_finalize_bad_args():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)

    with pytest.raises(BloqError, match=r'.*is not an available Soquet.*RightDangle\.z.*'):
        bb.finalize(x=x2, y=y2, z=Soquet(RightDangle, Register('asdf', 1)))


def test_finalize_alloc():
    bb, x, y = _get_bb()
    x2, y2 = bb.add(TestCNOT(), control=x, target=y)
    z = bb.allocate(1)

    cbloq = bb.finalize(x=x2, y=y2, z=z)
    assert len(list(cbloq.signature.rights())) == 3


def test_get_soquets():
    soqs = _get_dangling_soquets(Join(10).signature, right=True)
    assert list(soqs.keys()) == ['join']
    soq = soqs['join']
    assert soq.binst == RightDangle
    assert soq.reg.bitsize == 10

    soqs = _get_dangling_soquets(Join(10).signature, right=False)
    assert list(soqs.keys()) == ['join']
    soq = soqs['join']
    assert soq.shape == (10,)
    assert soq[0].reg.bitsize == 1


class TestMultiCNOT(Bloq):
    # A minimal test-bloq with a complicated `target` register.
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('control', 1), Register('target', 1, shape=(2, 3))])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', control: 'Soquet', target: NDArray['Soquet']
    ) -> Dict[str, SoquetT]:
        for i in range(2):
            for j in range(3):
                control, target[i, j] = bb.add(TestCNOT(), control=control, target=target[i, j])

        return {'control': control, 'target': target}


def test_complicated_target_register():
    bloq = TestMultiCNOT()
    cbloq = assert_valid_bloq_decomposition(bloq)
    assert len(cbloq.bloq_instances) == 2 * 3

    binst_graph = _create_binst_graph(cbloq.connections)
    # note: this includes the two `Dangling` generations.
    assert len(list(nx.topological_generations(binst_graph))) == 2 * 3 + 2

    circuit, _ = cbloq.to_cirq_circuit(**bloq.signature.get_cirq_quregs())
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
    bb = BloqBuilder()

    qs = bb.allocate(10)
    qs = bb.split(qs)
    qs = bb.join(qs)
    bb.free(qs)
    cbloq = bb.finalize()
    assert len(cbloq.connections) == 1 + 10 + 1


def test_util_convenience_methods_errors():
    bb = BloqBuilder()

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
    def signature(self) -> Signature:
        return Signature.build(stuff=1)

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=100)


class TestSerialBloq(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(stuff=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', stuff: 'SoquetT') -> Dict[str, 'Soquet']:

        for i in range(3):
            stuff = bb.add(Atom(), stuff=stuff)
        return {'stuff': stuff}


@frozen
class TestParallelBloq(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(stuff=3)

    def build_composite_bloq(self, bb: 'BloqBuilder', stuff: 'SoquetT') -> Dict[str, 'Soquet']:
        stuff = bb.split(stuff)
        for i in range(len(stuff)):
            stuff[i] = bb.add(Atom(), stuff=stuff[i])

        return {'stuff': bb.join(stuff)}


def test_test_serial_bloq_decomp():
    sbloq = TestSerialBloq()
    assert_valid_bloq_decomposition(sbloq)


def test_test_parallel_bloq_decomp():
    pbloq = TestParallelBloq()
    assert_valid_bloq_decomposition(pbloq)


@pytest.mark.parametrize('cls', [TestSerialBloq, TestParallelBloq])
def test_copy(cls):
    assert cls().supports_decompose_bloq()
    cbloq = cls().decompose_bloq()
    cbloq2 = cbloq.copy()
    assert cbloq is not cbloq2
    assert cbloq == cbloq2
    assert cbloq.debug_text() == cbloq2.debug_text()


@pytest.mark.parametrize('call_decompose', [False, True])
def test_add_from(call_decompose):
    bb = BloqBuilder()
    stuff = bb.add_register('stuff', 3)
    stuff = bb.add(TestParallelBloq(), stuff=stuff)
    if call_decompose:
        (stuff,) = bb.add_from(TestParallelBloq().decompose_bloq(), stuff=stuff)
    else:
        (stuff,) = bb.add_from(TestParallelBloq(), stuff=stuff)
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
    bb = BloqBuilder()
    _ = bb.add_register('control', 1)
    y = bb.add_register('control', 2)
    with pytest.raises(ValueError):
        bb.finalize(control=y)


def test_flatten():
    bb = BloqBuilder()
    stuff = bb.add_register('stuff', 3)
    stuff = bb.add(TestParallelBloq(), stuff=stuff)
    stuff = bb.add(TestParallelBloq(), stuff=stuff)
    cbloq = bb.finalize(stuff=stuff)
    assert len(cbloq.bloq_instances) == 2

    cbloq2 = cbloq.flatten_once(lambda binst: True)
    assert len(cbloq2.bloq_instances) == 5 * 2

    with pytest.raises(NotImplementedError):
        # Will keep trying to flatten non-decomposable things
        cbloq.flatten(lambda x: True)

    cbloq3 = cbloq.flatten(lambda binst: binst.bloq.supports_decompose_bloq())
    assert len(cbloq3.bloq_instances) == 5 * 2


def test_t_complexity():
    assert Atom().t_complexity().t == 100
    assert TestSerialBloq().decompose_bloq().t_complexity().t == 3 * 100
    assert TestParallelBloq().decompose_bloq().t_complexity().t == 3 * 100

    assert TestSerialBloq().t_complexity().t == 3 * 100
    assert TestParallelBloq().t_complexity().t == 3 * 100


def test_notebook():
    execute_notebook('composite_bloq')
