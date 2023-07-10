from typing import Iterable

from qualtran.components.registers import Register, Side
from qualtran.protos import registers_pb2
from qualtran.serialization import args_to_proto


def registers_to_proto(registers: Iterable[Register]) -> registers_pb2.Registers:
    return registers_pb2.Registers(registers=[register_to_proto(reg) for reg in registers])


def register_to_proto(register: Register) -> registers_pb2.Register:
    return registers_pb2.Register(
        name=register.name,
        bitsize=args_to_proto.int_or_sympy_to_proto(register.bitsize),
        shape=(args_to_proto.int_or_sympy_to_proto(s) for s in register.shape),
        side=_side_to_proto(register.side),
    )


def _side_to_proto(side: Side) -> registers_pb2.Register.Side:
    if side == Side.LEFT:
        return registers_pb2.Register.Side.LEFT
    if side == Side.RIGHT:
        return registers_pb2.Register.Side.RIGHT
    if side == Side.THRU:
        return registers_pb2.Register.Side.THRU
    return registers_pb2.Register.Side.UNKNOWN
