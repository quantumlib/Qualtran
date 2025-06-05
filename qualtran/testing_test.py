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
import re
from functools import cached_property

import attrs
import pytest
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqError,
    BloqInstance,
    CompositeBloq,
    Connection,
    LeftDangle,
    QBit,
    QDType,
    QFxp,
    QInt,
    QUInt,
    Register,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.for_testing import TestAtom, TestParallelCombo, TestTwoBitOp
from qualtran.testing import (
    assert_bloq_example_decompose,
    assert_bloq_example_make,
    assert_connections_compatible,
    assert_consistent_classical_action,
    assert_registers_match_dangling,
    assert_registers_match_parent,
    assert_soquets_belong_to_registers,
    assert_soquets_used_exactly_once,
    BloqCheckException,
    BloqCheckResult,
    check_bloq_example_decompose,
    check_bloq_example_make,
)


def _manually_make_test_cbloq_cxns():
    signature = Signature.build(q1=1, q2=1)
    q1, q2 = signature
    tcn = TestTwoBitOp()
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


def _manually_make_test_cbloq_typed_cxns(dtype_a: QDType, dtype_b: QDType):
    signature = Signature.build_from_dtypes(q1=dtype_a, q2=dtype_b)
    q1, q2 = signature
    add = Add(QInt(4))
    a, b = add.signature
    binst1 = BloqInstance(add, 1)
    binst2 = BloqInstance(add, 2)
    assert binst1 != binst2
    return [
        Connection(Soquet(LeftDangle, q1), Soquet(binst1, a)),
        Connection(Soquet(LeftDangle, q2), Soquet(binst1, b)),
        Connection(Soquet(binst1, a), Soquet(binst2, b)),
        Connection(Soquet(binst1, b), Soquet(binst2, a)),
        Connection(Soquet(binst2, a), Soquet(RightDangle, q1)),
        Connection(Soquet(binst2, b), Soquet(RightDangle, q2)),
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


def test_assert_soquets_belong_to_registers():
    cxns, signature = _manually_make_test_cbloq_cxns()
    cxns[3] = attrs.evolve(cxns[3], left=attrs.evolve(cxns[3].left, reg=Register('q3', QBit())))
    cbloq = CompositeBloq(cxns, signature)
    assert_registers_match_dangling(cbloq)
    assert_connections_compatible(cbloq)
    with pytest.raises(BloqError, match=r".*register doesn't exist on its bloq.*"):
        assert_soquets_belong_to_registers(cbloq)


def test_assert_soquets_used_exactly_once():
    cxns, signature = _manually_make_test_cbloq_cxns()
    binst1 = BloqInstance(TestTwoBitOp(), 1)
    binst2 = BloqInstance(TestTwoBitOp(), 2)
    control, target = TestTwoBitOp().signature

    cxns.append(Connection(Soquet(binst1, target), Soquet(binst2, control)))
    cbloq = CompositeBloq(cxns, signature)
    assert_registers_match_dangling(cbloq)
    assert_connections_compatible(cbloq)
    assert_soquets_belong_to_registers(cbloq)
    with pytest.raises(BloqError, match=r".*had already been produced by a different bloq.*"):
        assert_soquets_used_exactly_once(cbloq)


def test_check_bloq_example_make():
    @bloq_example
    def _my_cnot() -> Bloq:
        return 'CNOT 0 1'  # type: ignore[return-value]

    res, msg = check_bloq_example_make(_my_cnot)
    assert res is BloqCheckResult.FAIL, msg
    assert re.match(r'.*is not an instance of Bloq', msg)

    with pytest.raises(BloqCheckException) as raises_ctx:
        assert_bloq_example_make(_my_cnot)
        assert raises_ctx.value.check_result is BloqCheckResult.FAIL

    @bloq_example
    def _my_cnot_2() -> CNOT:
        return CNOT()

    res, msg = check_bloq_example_make(_my_cnot_2)
    assert res is BloqCheckResult.PASS, msg
    assert msg == ''

    assert_bloq_example_make(_my_cnot_2)


def test_check_bloq_decompose_pass():
    @bloq_example
    def _my_bloq() -> TestParallelCombo:
        return TestParallelCombo()

    res, msg = check_bloq_example_decompose(_my_bloq)
    assert res is BloqCheckResult.PASS, msg
    assert msg == ''

    assert_bloq_example_decompose(_my_bloq)


def test_check_bloq_decompose_na():
    @bloq_example
    def _my_bloq() -> TestAtom:
        return TestAtom()

    res, msg = check_bloq_example_decompose(_my_bloq)
    assert res is BloqCheckResult.NA, msg
    assert re.match(r'.*is atomic', msg)

    with pytest.raises(BloqCheckException) as raises_ctx:
        assert_bloq_example_decompose(_my_bloq)
        assert raises_ctx.value.check_result is BloqCheckResult.NA


@frozen
class TestMissingDecomp(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature([])


def test_check_bloq_decompose_missing():
    @bloq_example
    def _my_bloq() -> TestMissingDecomp:
        return TestMissingDecomp()

    res, msg = check_bloq_example_decompose(_my_bloq)
    assert res is BloqCheckResult.MISSING, msg
    assert re.match(r'.*declare a decomposition', msg)

    with pytest.raises(BloqCheckException) as raises_ctx:
        assert_bloq_example_decompose(_my_bloq)
        assert raises_ctx.value.check_result is BloqCheckResult.MISSING


@pytest.mark.parametrize(
    'dtype_a, dtype_b, expect_raise',
    (
        (QInt(4), QUInt(4), False),
        (QInt(4), QInt(5), True),
        (QUInt(4), QFxp(4, 4), False),
        (QUInt(4), QFxp(4, 2), True),
    ),
)
def test_assert_connections_compatible(dtype_a, dtype_b, expect_raise):
    cxns, signature = _manually_make_test_cbloq_typed_cxns(dtype_a, dtype_b)
    cbloq = CompositeBloq(cxns, signature=signature)
    if expect_raise:
        with pytest.raises(BloqError, match=r'.*QDTypes are incompatible.*'):
            assert_connections_compatible(cbloq)


def test_assert_valid_classical_action_valid_bloq():
    bitsize = 3
    valid_range = range(-(2**2), 2**2)
    assert_consistent_classical_action(Add(QInt(bitsize)), a=valid_range, b=valid_range)


def test_assert_valid_classical_action_valid_invalid_bloq():
    class BloqWithInvalidClassicaAction(Add):
        def on_classical_vals(self, a, b):
            return {'a': a, 'b': b}

    bitsize = 3
    valid_range = range(-(2**2), 2**2)
    b = BloqWithInvalidClassicaAction(QInt(bitsize))
    with pytest.raises(AssertionError):
        assert_consistent_classical_action(b, a=valid_range, b=valid_range)
