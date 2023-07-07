import dataclasses
from typing import Callable, cast, Dict, List, overload, Sequence, Type

import attrs
import cirq_ft

from qualtran.api import annotations_pb2, args_pb2, bloq_pb2, registers_pb2
from qualtran.quantum_graph.bloq import Bloq
from qualtran.quantum_graph.bloq_counts import SympySymbolAllocator
from qualtran.quantum_graph.composite_bloq import CompositeBloq
from qualtran.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from qualtran.quantum_graph.meta_bloq import ControlledBloq
from qualtran.quantum_graph.quantum_graph import BloqInstance, Connection, DanglingT, Soquet
from qualtran.serialization import bloq_args_to_proto


@overload
def bloq_to_proto(bloq: CompositeBloq) -> bloq_pb2.BloqLibrary:
    ...


@overload
def bloq_to_proto(bloq: ControlledBloq) -> bloq_pb2.BloqLibrary:
    ...


@overload
def bloq_to_proto(bloq: Bloq) -> bloq_pb2.Bloq:
    ...


def bloq_to_proto(
    bloq: Bloq, pred: Callable[[BloqInstance], bool] = lambda _: True, max_depth: int = 1000
):
    if type(bloq) == CompositeBloq:
        return _composite_bloq_to_proto(bloq, pred, max_depth)
    elif type(bloq) == ControlledBloq:
        return bloq_library_to_proto([bloq])
    else:
        return _bloq_to_proto(bloq, bloqs_to_idx={})


def bloq_library_to_proto(bloqs: Sequence[Bloq], *, name: str = ''):
    bloqs_to_idx = {b: i for i, b in enumerate(bloqs)}
    print("DEBUG:", bloqs)
    curr_idx = len(bloqs)
    for bloq in bloqs:
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            for binst in cbloq.bloq_instances:
                if binst.bloq not in bloqs_to_idx:
                    bloqs_to_idx[binst.bloq] = curr_idx
                    curr_idx = curr_idx + 1
        except:
            pass
        try:
            for _, subbloq in bloq.bloq_counts():
                if subbloq not in bloqs_to_idx:
                    bloqs_to_idx[subbloq] = curr_idx
                    curr_idx = curr_idx + 1
        except:
            pass

        if isinstance(bloq, ControlledBloq):
            if bloq.subbloq not in bloqs_to_idx:
                bloqs_to_idx[bloq.subbloq] = curr_idx
                curr_idx = curr_idx + 1

    library = bloq_pb2.BloqLibrary(name=name)
    for bloq, bloq_id in bloqs_to_idx.items():
        bloq_counts, decomposition = None, None
        print("DEBUG:", bloq, bloq_id, len(bloqs))
        print(bloqs_to_idx)
        if bloq_id < len(bloqs):
            try:
                cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
                decomposition = [
                    _connection_to_proto(cxn, bloqs_to_idx) for cxn in cbloq.connections
                ]
            except Exception as e:
                print(e)

            try:
                bloq_counts = {
                    bloqs_to_idx[b]: bloq_args_to_proto.int_or_sympy_to_proto(c)
                    for c, b in bloq.bloq_counts(SympySymbolAllocator())
                }
            except:
                pass
        library.table.add(
            bloq_id=bloq_id,
            decomposition=decomposition,
            bloq_counts=bloq_counts,
            bloq=_bloq_to_proto(bloq, bloqs_to_idx=bloqs_to_idx),
        )
    return library


def _composite_bloq_to_proto(
    cbloq: CompositeBloq, pred: Callable[[BloqInstance], bool], max_depth: int
) -> bloq_pb2.BloqLibrary:
    bloq_to_idx: Dict[Bloq, int] = {cbloq: 0}
    _populate_bloq_to_idx(cbloq, bloq_to_idx, pred, max_depth - 1)
    return bloq_library_to_proto(list(bloq_to_idx.keys()), name=cbloq.pretty_name())


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


def _populate_bloq_to_idx(
    cbloq: CompositeBloq,
    bloq_to_idx: Dict[Bloq, int],
    pred: Callable[[BloqInstance], bool],
    max_depth: int,
):
    if max_depth <= 0:
        return
    for binst in cbloq.bloq_instances:
        if binst.bloq in bloq_to_idx:
            continue
        next_idx = len(bloq_to_idx)
        bloq_to_idx[binst.bloq] = next_idx
        if pred(binst):
            try:
                _populate_bloq_to_idx(binst.bloq.decompose_bloq(), bloq_to_idx, pred, max_depth - 1)
            except:
                pass


def _bloq_to_proto(bloq: Bloq, *, bloqs_to_idx: Dict[Bloq, int]) -> bloq_pb2.Bloq:
    try:
        t_complexity = t_complexity_to_proto(bloq.t_complexity())
    except:
        t_complexity = None

    if isinstance(bloq, ControlledBloq):
        args = [args_pb2.BloqArg(name='subbloq', subbloq=bloqs_to_idx[bloq.subbloq])]
    else:
        args = _bloq_args_to_proto(bloq)

    return bloq_pb2.Bloq(
        name=bloq.pretty_name(),
        registers=registers_to_proto(bloq.registers),
        t_complexity=t_complexity,
        args=args,
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
    return ret if ret else None


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
