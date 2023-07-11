from typing import Iterable, List

from qualtran.components.registers import Register, Side
from qualtran.protos import registers_pb2
from qualtran.serialization import args


def registers_to_proto(registers: Iterable[Register]) -> registers_pb2.Registers:
    return registers_pb2.Registers(registers=[register_to_proto(reg) for reg in registers])


def registers_from_proto(registers: registers_pb2.Registers) -> List[Register]:
    return [register_from_proto(reg) for reg in registers.registers]


def register_to_proto(register: Register) -> registers_pb2.Register:
    return registers_pb2.Register(
        name=register.name,
        bitsize=args.int_or_sympy_to_proto(register.bitsize),
        shape=(args.int_or_sympy_to_proto(s) for s in register.shape),
        side=_side_to_proto(register.side),
    )


def register_from_proto(register: registers_pb2.Register) -> Register:
    return Register(
        name=register.name,
        bitsize=args.int_or_sympy_from_proto(register.bitsize),
        shape=tuple(args.int_or_sympy_from_proto(s) for s in register.shape),
        side=_side_from_proto(register.side),
    )


def _side_to_proto(side: Side) -> registers_pb2.Register.Side:
    if side == Side.LEFT:
        return registers_pb2.Register.Side.LEFT
    if side == Side.RIGHT:
        return registers_pb2.Register.Side.RIGHT
    if side == Side.THRU:
        return registers_pb2.Register.Side.THRU
    return registers_pb2.Register.Side.UNKNOWN


def _side_from_proto(side: registers_pb2.Register.Side) -> Side:
    if side == registers_pb2.Register.Side.LEFT:
        return Side.LEFT
    if side == registers_pb2.Register.Side.RIGHT:
        return Side.RIGHT
    if side == registers_pb2.Register.Side.THRU:
        return Side.THRU
    return Side.UNKNOWN
