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

from typing import Iterable, Tuple

from qualtran import Register, Side
from qualtran.protos import registers_pb2
from qualtran.serialization import args
from qualtran.serialization.data_types import data_type_from_proto, data_type_to_proto


def registers_to_proto(registers: Iterable[Register]) -> registers_pb2.Registers:
    return registers_pb2.Registers(registers=[register_to_proto(reg) for reg in registers])


def registers_from_proto(registers: registers_pb2.Registers) -> Tuple[Register, ...]:
    return tuple(register_from_proto(reg) for reg in registers.registers)


def register_to_proto(register: Register) -> registers_pb2.Register:
    return registers_pb2.Register(
        name=register.name,
        dtype=data_type_to_proto(register.dtype),
        shape=(args.int_or_sympy_to_proto(s) for s in register.shape_symbolic),
        side=_side_to_proto(register.side),
    )


def register_from_proto(register: registers_pb2.Register) -> Register:
    return Register(
        name=register.name,
        dtype=data_type_from_proto(register.dtype),
        shape=tuple(args.int_or_sympy_from_proto(s) for s in register.shape),
        side=_side_from_proto(register.side),
    )


def _side_to_proto(side: Side) -> registers_pb2.Register.Side.ValueType:
    if side == Side.LEFT:
        return registers_pb2.Register.Side.LEFT
    if side == Side.RIGHT:
        return registers_pb2.Register.Side.RIGHT
    if side == Side.THRU:
        return registers_pb2.Register.Side.THRU
    return registers_pb2.Register.Side.UNKNOWN


def _side_from_proto(side: registers_pb2.Register.Side.ValueType) -> Side:
    if side == registers_pb2.Register.Side.LEFT:
        return Side.LEFT
    if side == registers_pb2.Register.Side.RIGHT:
        return Side.RIGHT
    if side == registers_pb2.Register.Side.THRU:
        return Side.THRU
    raise ValueError(f'Unknown Side type {side}')
