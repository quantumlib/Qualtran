import attrs
import pytest
import sympy

from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.serialization.registers import (
    register_from_proto,
    register_to_proto,
    registers_from_proto,
    registers_to_proto,
)


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
    reg_proto = register_to_proto(reg)
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

    assert register_from_proto(reg_proto) == reg

    reg2 = attrs.evolve(reg, name='my_reg2')
    reg3 = attrs.evolve(reg, name='my_reg3')
    registers = FancyRegisters([reg, reg2, reg3])
    registers_proto = registers_to_proto(registers)
    assert list(registers_proto.registers) == [register_to_proto(r) for r in registers]

    assert registers_from_proto(registers_proto) == registers
