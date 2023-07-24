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

import attrs
import pytest
from attrs import frozen

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
)
from qualtran._infra.bloq_test import TestCNOT
from qualtran.testing import (
    assert_connections_compatible,
    assert_registers_match_dangling,
    assert_registers_match_parent,
    assert_soquets_belong_to_registers,
    assert_soquets_used_exactly_once,
)


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


def test_assert_registers_match_parent():
    @frozen
    class BadRegBloq(Bloq):
        @cached_property
        def signature(self) -> 'Signature':
            return Signature.build(x=2, y=3)

        def decompose_bloq(self) -> 'CompositeBloq':
            # !! order of registers swapped.
            bb, soqs = BloqBuilder.from_signature(Signature.build(y=3, x=2))
            x, y = bb.add(BadRegBloq(), x=soqs['x'], y=soqs['y'])
            return bb.finalize(x=x, y=y)

    with pytest.raises(BloqError, match=r'Parent registers do not match.*'):
        assert_registers_match_parent(BadRegBloq())


def test_assert_registers_match_dangling():
    cxns, _ = _manually_make_test_cbloq_cxns()
    cbloq = CompositeBloq(cxns, signature=Signature.build(ctrl=1, target=1))
    with pytest.raises(BloqError, match=r'.*.*does not match the registers of the bloq.*'):
        assert_registers_match_dangling(cbloq)


def test_assert_connections_compatible():
    from qualtran.bloqs.basic_gates import CSwap, TwoBitCSwap

    bb = BloqBuilder()
    ctrl = bb.add_register('c', 1)
    x = bb.add_register('x', 10)
    y = bb.add_register('y', 10)
    ctrl, x, y = bb.add(CSwap(10), ctrl=ctrl, x=x, y=y)
    ctrl, x, y = bb.add(TwoBitCSwap(), ctrl=ctrl, x=x, y=y)
    cbloq = bb.finalize(c=ctrl, x=x, y=y)
    assert_registers_match_dangling(cbloq)
    with pytest.raises(BloqError, match=r'.*bitsizes are incompatible.*'):
        assert_connections_compatible(cbloq)


def test_assert_soquets_belong_to_registers():
    cxns, signature = _manually_make_test_cbloq_cxns()
    cxns[3] = attrs.evolve(cxns[3], left=attrs.evolve(cxns[3].left, reg=Register('q3', 1)))
    cbloq = CompositeBloq(cxns, signature)
    assert_registers_match_dangling(cbloq)
    assert_connections_compatible(cbloq)
    with pytest.raises(BloqError, match=r".*register doesn't exist on its bloq.*"):
        assert_soquets_belong_to_registers(cbloq)


def test_assert_soquets_used_exactly_once():
    cxns, signature = _manually_make_test_cbloq_cxns()
    binst1 = BloqInstance(TestCNOT(), 1)
    binst2 = BloqInstance(TestCNOT(), 2)
    control, target = TestCNOT().signature

    cxns.append(Connection(Soquet(binst1, target), Soquet(binst2, control)))
    cbloq = CompositeBloq(cxns, signature)
    assert_registers_match_dangling(cbloq)
    assert_connections_compatible(cbloq)
    assert_soquets_belong_to_registers(cbloq)
    with pytest.raises(BloqError, match=r".*had already been produced by a different bloq.*"):
        assert_soquets_used_exactly_once(cbloq)
