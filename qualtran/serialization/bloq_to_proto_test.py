import dataclasses
from typing import Union

import attrs
import cirq_ft
import pytest
import sympy

import qualtran
from qualtran.api import registers_pb2
from qualtran.quantum_graph.bloq_test import TestCNOT
from qualtran.quantum_graph.composite_bloq_test import TestTwoCNOT
from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.serialization import bloq_to_proto


def test_bloq_to_proto_cnot():
    cnot = TestCNOT()
    proto = bloq_to_proto.bloq_to_proto(cnot)
    assert proto.name == "TestCNOT"
    assert len(proto.registers.registers) == 2
    assert proto.registers.registers[0].name == 'control'
    assert proto.registers.registers[0].bitsize == 1
    assert proto.registers.registers[0].side == registers_pb2.Register.Side.THRU


def test_cbloq_to_proto_two_cnot():
    cbloq = TestTwoCNOT().decompose_bloq()
    proto = bloq_to_proto.bloq_to_proto(cbloq)
    assert len(proto.table.bloqs) == 1
    assert len(proto.cbloq.connections) == 6
    assert proto.t_complexity.clifford == 2


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
    cswap_proto = bloq_to_proto.bloq_to_proto(TestCSwap(100))
    assert cswap_proto.name == "TestCSwap"
    assert len(cswap_proto.args) == 1
    assert cswap_proto.args[0].name == "bitsize"
    assert cswap_proto.args[0].int_val == 100
    assert len(cswap_proto.registers.registers) == 3

    cbloq = TestTwoCSwap(100).decompose_bloq()
    proto = bloq_to_proto.bloq_to_proto(cbloq)
    assert proto.table.bloqs[0].bloq == cswap_proto
    assert proto.t_complexity.t == 7 * 100 * 2
    assert proto.t_complexity.clifford == 10 * 100 * 2
    assert len(proto.cbloq.connections) == 9


@pytest.mark.parametrize(
    'bitsize, shape, side',
    [(1, (2, 3), Side.RIGHT), (10, (), Side.LEFT), (1000, (10, 1, 20), Side.THRU)],
)
def test_registers_to_proto(bitsize, shape, side):
    reg = FancyRegister('my_reg', bitsize=bitsize, shape=shape, side=side)
    reg_proto = bloq_to_proto.register_to_proto(reg)
    assert reg_proto.name == 'my_reg'
    assert tuple(reg_proto.shape) == shape
    assert reg_proto.bitsize == bitsize
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
