import dataclasses
from typing import Any, Callable, Dict, List

import attrs
import cirq_ft

from qualtran.api import annotations_pb2, args_pb2, bloq_pb2, registers_pb2
from qualtran.quantum_graph.bloq import Bloq
from qualtran.quantum_graph.bloq_counts import SympySymbolAllocator
from qualtran.quantum_graph.composite_bloq import CompositeBloq
from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.quantum_graph.quantum_graph import BloqInstance, Connection, DanglingT, Soquet
from qualtran.serialization import bloq_args_to_proto


def _iter_fields(bloq: Bloq):
    if dataclasses.is_dataclass(type(bloq)):
        for field in dataclasses.fields(bloq):
            yield field
    elif attrs.has(type(bloq)):
        for field in attrs.fields(type(bloq)):
            yield field


def bloqs_to_proto(
    *bloqs: Bloq,
    name: str = '',
    pred: Callable[[BloqInstance], bool] = lambda _: True,
    max_depth: int = 1,
) -> bloq_pb2.BloqLibrary:
    bloq_to_idx = {b: i for i, b in enumerate(bloqs)}
    for bloq in bloqs:
        _populate_bloq_to_idx(bloq, bloq_to_idx, pred, max_depth)

    # `bloq_to_idx` would now contain a list of all bloqs that should be serialized.
    library = bloq_pb2.BloqLibrary(name=name)
    for bloq, bloq_id in bloq_to_idx.items():
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            decomposition = [_connection_to_proto(cxn, bloq_to_idx) for cxn in cbloq.connections]
        except (NotImplementedError, KeyError):
            decomposition = None

        try:
            bloq_counts = {
                bloq_to_idx[b]: bloq_args_to_proto.int_or_sympy_to_proto(c)
                for c, b in bloq.bloq_counts(SympySymbolAllocator())
            }
        except (NotImplementedError, KeyError):
            bloq_counts = None

        library.table.add(
            bloq_id=bloq_id,
            decomposition=decomposition,
            bloq_counts=bloq_counts,
            bloq=_bloq_to_proto(bloq, bloq_to_idx=bloq_to_idx),
        )
    return library


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
    return bloq_pb2.BloqInstance(instance_id=binst.i, bloq_id=bloq_to_idx[binst.bloq])


def _add_bloq_to_dict(bloq: Bloq, bloq_to_idx: Dict[Bloq, int]):
    if bloq not in bloq_to_idx:
        next_idx = len(bloq_to_idx)
        bloq_to_idx[bloq] = next_idx


def _populate_bloq_to_idx(
    bloq: Bloq, bloq_to_idx: Dict[Bloq, int], pred: Callable[[BloqInstance], bool], max_depth: int
):
    """Recursively track all primitive Bloqs to be serialized, as part of `bloq_to_idx` dictionary."""

    assert bloq in bloq_to_idx
    if max_depth > 0:
        # Decompose the current Bloq and track it's decomposed Bloqs.
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            for binst in cbloq.bloq_instances:
                _add_bloq_to_dict(bloq, bloq_to_idx)
                if pred(binst):
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, max_depth - 1)
                else:
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, 0)
        except Exception as e:
            print(e)

        # Approximately decompose the current Bloq and it's decomposed Bloqs.
        try:
            for _, subbloq in bloq.bloq_counts(SympySymbolAllocator()):
                _add_bloq_to_dict(subbloq, bloq_to_idx)
                _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)

        except Exception as e:
            print(e)

    # If the current Bloq contains other Bloqs as sub-bloqs, add them to the `bloq_to_idx` dict.
    # This is only supported for Bloqs implemented as dataclasses / attrs.
    for field in _iter_fields(bloq):
        subbloq = getattr(bloq, field.name)
        if isinstance(subbloq, Bloq):
            _add_bloq_to_dict(field.val, bloq_to_idx)
            _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)


def _bloq_to_proto(bloq: Bloq, *, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Bloq:
    try:
        t_complexity = t_complexity_to_proto(bloq.t_complexity())
    except:
        t_complexity = None

    return bloq_pb2.Bloq(
        name=bloq.pretty_name(),
        registers=registers_to_proto(bloq.registers),
        t_complexity=t_complexity,
        args=_bloq_args_to_proto(bloq, bloq_to_idx=bloq_to_idx),
    )


def _bloq_args_to_proto(bloq: Bloq, *, bloq_to_idx: Dict[Bloq, int]) -> List[args_pb2.BloqArg]:
    ret = [
        _bloq_arg_to_proto(name=field.name, val=getattr(bloq, field.name), bloq_to_idx=bloq_to_idx)
        for field in _iter_fields(bloq)
    ]
    return ret if ret else None


def _bloq_arg_to_proto(name: str, val: Any, bloq_to_idx: Dict[Bloq, int]) -> args_pb2.BloqArg:
    if isinstance(val, Bloq):
        return args_pb2.BloqArg(name=name, subbloq=bloq_to_idx[val])
    return bloq_args_to_proto.arg_to_proto(name=name, val=val)


def registers_to_proto(registers: FancyRegisters) -> registers_pb2.Registers:
    return registers_pb2.Registers(registers=[register_to_proto(reg) for reg in registers])


def register_to_proto(register: FancyRegister) -> registers_pb2.Register:
    return registers_pb2.Register(
        name=register.name,
        bitsize=bloq_args_to_proto.int_or_sympy_to_proto(register.bitsize),
        shape=(bloq_args_to_proto.int_or_sympy_to_proto(s) for s in register.shape),
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
