import dataclasses
from typing import cast, Dict, List, overload, Type

import attrs
import cirq_ft

from qualtran.api import annotations_pb2, args_pb2, bloq_pb2, registers_pb2
from qualtran.quantum_graph.bloq import Bloq
from qualtran.quantum_graph.composite_bloq import CompositeBloq
from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.quantum_graph.quantum_graph import BloqInstance, Connection, DanglingT, Soquet
from qualtran.serialization import bloq_args_to_proto


@overload
def bloq_to_proto(bloq: CompositeBloq) -> bloq_pb2.CompositeBloq:
    ...


@overload
def bloq_to_proto(bloq: Bloq) -> bloq_pb2.Bloq:
    ...


def bloq_to_proto(bloq):
    if isinstance(bloq, CompositeBloq):
        return _composite_bloq_to_proto(bloq)
    else:
        return _bloq_to_proto(bloq)


def _composite_bloq_to_proto(bloq: CompositeBloq) -> bloq_pb2.CompositeBloq:
    bloq_to_idx: Dict[Bloq, int] = {}
    _populate_bloq_to_idx(bloq, bloq_to_idx)
    assert all(i == v for i, (k, v) in enumerate(bloq_to_idx.items()))
    flat_bloqs = [*bloq_to_idx.keys()]
    assert flat_bloqs[-1] == bloq

    table = bloq_pb2.BloqTable()
    for flat_bloq in flat_bloqs[:-1]:
        if isinstance(flat_bloq, CompositeBloq):
            table.bloqs.add(cbloq=_composite_bloq_to_proto_lite(flat_bloq, bloq_to_idx))
        else:
            table.bloqs.add(bloq=_bloq_to_proto(flat_bloq))

    t_complexity_proto = (
        t_complexity_to_proto(bloq.t_complexity()) if bloq.declares_t_complexity() else None
    )
    # TODO: Add support for bloq-counts.
    bloq_counts = None

    return bloq_pb2.CompositeBloq(
        table=table,
        cbloq=_composite_bloq_to_proto_lite(bloq, bloq_to_idx),
        t_complexity=t_complexity_proto,
        bloq_counts=bloq_counts,
    )


def _composite_bloq_to_proto_lite(
    bloq: CompositeBloq, bloq_to_idx: Dict[Bloq, int]
) -> bloq_pb2.CompositeBloqLite:
    return bloq_pb2.CompositeBloqLite(
        name=bloq.pretty_name(),
        registers=registers_to_proto(bloq.registers),
        connections=(_connection_to_proto(cxn, bloq_to_idx) for cxn in bloq.connections),
    )


def _connection_to_proto(cxn: Connection, bloq_to_idx: Dict[Bloq, int]):
    return bloq_pb2.Connection(
        left=_soquet_to_proto(cxn.left, bloq_to_idx), right=_soquet_to_proto(cxn.right, bloq_to_idx)
    )


def _soquet_to_proto(soq: Soquet, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Soquet:
    if isinstance(soq.binst, DanglingT):
        return bloq_pb2.Soquet(
            dangling_t=repr(soq.binst), register=register_to_proto(soq.reg), index=soq.idx
        )
    else:
        return bloq_pb2.Soquet(
            bloq_instance=_bloq_instance_to_proto(soq.binst, bloq_to_idx),
            register=register_to_proto(soq.reg),
            index=soq.idx,
        )


def _bloq_instance_to_proto(
    binst: BloqInstance, bloq_to_idx: Dict[Bloq, int]
) -> bloq_pb2.BloqInstance:
    return bloq_pb2.BloqInstance(id=binst.i, bloq_id=bloq_to_idx[binst.bloq])


def _populate_bloq_to_idx(bloq: Bloq, bloq_to_idx: Dict[Bloq, int]):
    if isinstance(bloq, CompositeBloq):
        for binst in bloq.bloq_instances:
            _populate_bloq_to_idx(binst.bloq, bloq_to_idx)
    if bloq not in bloq_to_idx:
        next_idx = len(bloq_to_idx)
        bloq_to_idx[bloq] = next_idx


def _bloq_to_proto(bloq: Bloq) -> bloq_pb2.Bloq:
    return bloq_pb2.Bloq(
        name=bloq.pretty_name(),
        registers=registers_to_proto(bloq.registers),
        t_complexity=t_complexity_to_proto(bloq.t_complexity()),
        args=_bloq_args_to_proto(bloq),
    )


def _bloq_args_to_proto(bloq: Bloq) -> List[args_pb2.BloqArg]:
    ret: List[args_pb2.BloqArg] = []
    if dataclasses.is_dataclass(type(bloq)):
        for field in dataclasses.fields(bloq):
            ret.append(
                bloq_args_to_proto.arg_to_proto(name=field.name, val=getattr(bloq, field.name))
            )
    elif attrs.has(type(bloq)):
        for field in attrs.fields(cast(Type[attrs.AttrsInstance], type(bloq))):
            ret.append(
                bloq_args_to_proto.arg_to_proto(name=field.name, val=getattr(bloq, field.name))
            )
    return ret


def registers_to_proto(registers: FancyRegisters) -> registers_pb2.Registers:
    return registers_pb2.Registers(registers=[register_to_proto(reg) for reg in registers])


def register_to_proto(register: FancyRegister) -> registers_pb2.Register:
    return registers_pb2.Register(
        name=register.name,
        bitsize=register.bitsize,
        shape=register.shape,
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


def t_complexity_to_proto(t_complexity: cirq_ft.TComplexity) -> annotations_pb2.TComplexity:
    return annotations_pb2.TComplexity(
        clifford=t_complexity.clifford, rotations=t_complexity.rotations, t=t_complexity.t
    )
