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

import dataclasses
from typing import Union

import attrs
import cirq
import sympy

from qualtran import Bloq, ControlledBloq, Signature
from qualtran._infra.composite_bloq_test import TestTwoCNOT
from qualtran.bloqs.factoring.mod_exp import ModExp
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.cirq_interop._cirq_to_bloq_test import TestCNOT as TestCNOTCirq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.protos import registers_pb2
from qualtran.serialization import bloq as bloq_serialization


def test_bloq_to_proto_cnot():
    bloq_serialization.RESOLVER_DICT.update({'TestCNOT': TestCNOTCirq})

    cnot = TestCNOTCirq()
    proto_lib = bloq_serialization.bloqs_to_proto(cnot)
    assert len(proto_lib.table) == 2
    assert len(proto_lib.table[0].decomposition) == 4
    assert len(proto_lib.table[1].decomposition) == 0

    proto = proto_lib.table[0].bloq
    assert proto.name == "TestCNOT"
    assert len(proto.registers.registers) == 2
    assert proto.registers.registers[0].name == 'control'
    assert proto.registers.registers[0].bitsize.int_val == 1
    assert proto.registers.registers[0].side == registers_pb2.Register.Side.THRU

    deserialized_bloqs = bloq_serialization.bloqs_from_proto(proto_lib)
    assert cnot in deserialized_bloqs
    assert CirqGateAsBloq(cirq.CNOT) in deserialized_bloqs


def test_cbloq_to_proto_two_cnot():
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCNOT': TestTwoCNOT})

    cbloq = TestTwoCNOT().decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2  # TestTwoCNOT and TestCNOT
    # First one is always the CompositeBloq.
    assert len(proto_lib.table[0].decomposition) == 6
    # Test round trip.
    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


@attrs.frozen
class TestCSwap(Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def t_complexity(self) -> TComplexity:
        return TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)


@dataclasses.dataclass(frozen=True)
class TestTwoCSwap(Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def build_composite_bloq(self, bb, ctrl, x, y):
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=x, y=y)
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=y, y=x)
        return {'ctrl': ctrl, 'x': x, 'y': y}


def test_cbloq_to_proto_test_two_cswap():
    bloq_serialization.RESOLVER_DICT.update({'TestCSwap': TestCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCSwap': TestTwoCSwap})

    bitsize = sympy.Symbol("a") * sympy.Symbol("b")
    cswap_proto_lib = bloq_serialization.bloqs_to_proto(TestCSwap(bitsize))
    assert len(cswap_proto_lib.table) == 1
    assert len(cswap_proto_lib.table[0].decomposition) == 0
    cswap_proto = cswap_proto_lib.table[0].bloq
    assert cswap_proto.name == "TestCSwap"
    assert len(cswap_proto.args) == 1
    assert cswap_proto.args[0].name == "bitsize"
    assert sympy.parse_expr(cswap_proto.args[0].sympy_expr) == bitsize
    assert len(cswap_proto.registers.registers) == 3

    assert TestCSwap(bitsize) in bloq_serialization.bloqs_from_proto(cswap_proto_lib)

    cswap_proto = bloq_serialization.bloqs_to_proto(TestCSwap(100)).table[0].bloq
    cbloq = TestTwoCSwap(100).decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2
    assert proto_lib.table[1].bloq == cswap_proto
    assert proto_lib.table[0].bloq.t_complexity.t == 7 * 100 * 2
    assert proto_lib.table[0].bloq.t_complexity.clifford == 10 * 100 * 2
    assert len(proto_lib.table[0].decomposition) == 9

    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


def test_cbloq_to_proto_test_mod_exp():
    mod_exp = ModExp.make_for_shor(17 * 19, g=8)
    cbloq = mod_exp.decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq, max_depth=1)
    num_binst = len(set(binst.bloq for binst in cbloq.bloq_instances))
    assert len(proto_lib.table) == 1 + num_binst

    cbloq = ControlledBloq(mod_exp)
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq, max_depth=1)
    # 2x that of ModExp.make_for_shor(17 * 19).decompose_bloq() because each bloq in the
    # decomposition is now controlled and each Controlled(subbloq) requires 2 entries in the
    # table - one for ControlledBloq and second for subbloq.
    assert len(proto_lib.table) == 2 * (1 + num_binst)

    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


@attrs.frozen
class TestMetaBloq(Bloq):
    sub_bloq_one: Bloq
    sub_bloq_two: Bloq

    def __post_attrs_init__(self):
        assert self.sub_bloq_one.signature == self.sub_bloq_two.signature

    @property
    def signature(self) -> 'Signature':
        return self.sub_bloq_one.signature

    def build_composite_bloq(self, bb, **soqs):
        soqs |= bb.add_d(self.sub_bloq_one, **soqs)
        soqs |= bb.add_d(self.sub_bloq_two, **soqs)
        return soqs


def test_meta_bloq_to_proto():
    bloq_serialization.RESOLVER_DICT.update({'TestCSwap': TestCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCSwap': TestTwoCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestMetaBloq': TestMetaBloq})

    sub_bloq_one = TestTwoCSwap(20)
    sub_bloq_two = TestTwoCSwap(20).decompose_bloq()
    bloq = TestMetaBloq(sub_bloq_one, sub_bloq_two)
    proto_lib = bloq_serialization.bloqs_to_proto(bloq, name="Meta Bloq Test")
    assert proto_lib.name == "Meta Bloq Test"
    assert len(proto_lib.table) == 3  # TestMetaBloq, TestTwoCSwap, CompositeBloq

    proto_lib = bloq_serialization.bloqs_to_proto(bloq, max_depth=2)
    assert len(proto_lib.table) == 4  # TestMetaBloq, TestTwoCSwap, CompositeBloq, TestCSwap

    assert proto_lib.table[0].bloq.name == 'TestMetaBloq'
    assert len(proto_lib.table[0].decomposition) == 9

    assert proto_lib.table[1].bloq.name == 'TestTwoCSwap'
    assert len(proto_lib.table[1].decomposition) == 9

    assert proto_lib == bloq_serialization.bloqs_to_proto(bloq, bloq, TestTwoCSwap(20), max_depth=2)
    assert bloq in bloq_serialization.bloqs_from_proto(proto_lib)
