import dataclasses
from typing import Union

import attrs
import cirq_ft
import pytest
import sympy

import qualtran
from qualtran.api import registers_pb2
from qualtran.bloq_algos.factoring.mod_exp import ModExp
from qualtran.quantum_graph.bloq_test import TestCNOT
from qualtran.quantum_graph.composite_bloq_test import TestTwoCNOT
from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.serialization import bloq_to_proto


def test_bloq_to_proto_cnot():
    cnot = TestCNOT()
    proto = bloq_to_proto.bloqs_to_proto(cnot)
    assert len(proto.table) == 1
    proto = proto.table[0]
    assert len(proto.decomposition) == 0
    proto = proto.bloq
    assert proto.name == "TestCNOT"
    assert len(proto.registers.registers) == 2
    assert proto.registers.registers[0].name == 'control'
    assert proto.registers.registers[0].bitsize.int_val == 1
    assert proto.registers.registers[0].side == registers_pb2.Register.Side.THRU


def test_cbloq_to_proto_two_cnot():
    cbloq = TestTwoCNOT().decompose_bloq()
    proto_lib = bloq_to_proto.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2  # TestTwoCNOT and TestCNOT
    # First one is always the CompositeBloq.
    assert len(proto_lib.table[0].decomposition) == 6
    assert proto_lib.table[0].bloq.t_complexity.clifford == 2


@attrs.frozen
class TestCSwap(qualtran.Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def t_complexity(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)


@dataclasses.dataclass
class TestTwoCSwap(qualtran.Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def build_composite_bloq(self, bb, ctrl, x, y):
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=x, y=y)
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=y, y=x)
        return {'ctrl': ctrl, 'x': x, 'y': y}


def test_cbloq_to_proto_test_two_cswap():
    bitsize = sympy.Symbol("a") * sympy.Symbol("b")
    cswap_proto = bloq_to_proto.bloqs_to_proto(TestCSwap(bitsize))
    assert len(cswap_proto.table) == 1
    assert len(cswap_proto.table[0].decomposition) == 0
    cswap_proto = cswap_proto.table[0].bloq
    assert cswap_proto.name == "TestCSwap"
    assert len(cswap_proto.args) == 1
    assert cswap_proto.args[0].name == "bitsize"
    assert sympy.parse_expr(cswap_proto.args[0].sympy_expr) == bitsize
    assert len(cswap_proto.registers.registers) == 3

    cswap_proto = bloq_to_proto.bloqs_to_proto(TestCSwap(100)).table[0].bloq
    cbloq = TestTwoCSwap(100).decompose_bloq()
    proto_lib = bloq_to_proto.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2
    assert proto_lib.table[1].bloq == cswap_proto
    assert proto_lib.table[0].bloq.t_complexity.t == 7 * 100 * 2
    assert proto_lib.table[0].bloq.t_complexity.clifford == 10 * 100 * 2
    assert len(proto_lib.table[0].decomposition) == 9


def test_cbloq_to_proto_test_mod_exp():
    cbloq = ModExp.make_for_shor(17 * 19).decompose_bloq()
    proto_lib = bloq_to_proto.bloqs_to_proto(cbloq, max_depth=1)
    assert len(proto_lib.table) == 1 + len(set(binst.bloq for binst in cbloq.bloq_instances))


@pytest.mark.parametrize(
    'bitsize, shape, side',
    [
        (1, (2, 3), Side.RIGHT),
        (10, (), Side.LEFT),
        (1000, (10, 1, 20), Side.THRU),
        (
            sympy.Symbol("a") * sympy.Symbol("b"),
            (sympy.Symbol("c") + sympy.Symbol("d"),),
            Side.THRU,
        ),
    ],
)
def test_registers_to_proto(bitsize, shape, side):
    reg = FancyRegister('my_reg', bitsize=bitsize, shape=shape, side=side)
    reg_proto = bloq_to_proto.register_to_proto(reg)
    assert reg_proto.name == 'my_reg'
    if shape and isinstance(shape[0], sympy.Expr):
        assert tuple(sympy.parse_expr(s.sympy_expr) for s in reg_proto.shape) == shape
    else:
        assert tuple(s.int_val for s in reg_proto.shape) == shape
    if isinstance(bitsize, int):
        assert reg_proto.bitsize.int_val == bitsize
    else:
        assert sympy.parse_expr(reg_proto.bitsize.sympy_expr) == bitsize
    assert reg_proto.side == side.value

    reg2 = attrs.evolve(reg, name='my_reg2')
    reg3 = attrs.evolve(reg, name='my_reg3')
    registers = FancyRegisters([reg, reg2, reg3])
    registers_proto = bloq_to_proto.registers_to_proto(registers)
    assert list(registers_proto.registers) == [
        bloq_to_proto.register_to_proto(r) for r in registers
    ]


def test_t_complexity_to_proto():
    t_complexity = cirq_ft.TComplexity(t=10, clifford=100, rotations=1000)
    proto = bloq_to_proto.t_complexity_to_proto(t_complexity)
    assert (proto.t, proto.clifford, proto.rotations) == (10, 100, 1000)
