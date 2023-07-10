import dataclasses
from typing import Any, Callable, Dict, List

import attrs

from qualtran.components.bloq import Bloq
from qualtran.components.composite_bloq import CompositeBloq
from qualtran.components.quantum_graph import BloqInstance, Connection, DanglingT, Soquet
from qualtran.protos import args_pb2, bloq_pb2
from qualtran.resource_counting.bloq_counts import SympySymbolAllocator
from qualtran.serialization import annotations_to_proto, args_to_proto, registers_to_proto


def bloqs_to_proto(
    *bloqs: Bloq,
    name: str = '',
    pred: Callable[[BloqInstance], bool] = lambda _: True,
    max_depth: int = 1,
) -> bloq_pb2.BloqLibrary:
    """Serializes one or more Bloqs as a `BloqLibrary`."""

    bloq_to_idx: Dict[Bloq, int] = {}
    for bloq in bloqs:
        _add_bloq_to_dict(bloq, bloq_to_idx)
        _populate_bloq_to_idx(bloq, bloq_to_idx, pred, max_depth)

    # `bloq_to_idx` would now contain a list of all bloqs that should be serialized.
    library = bloq_pb2.BloqLibrary(name=name)
    for bloq, bloq_id in bloq_to_idx.items():
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            decomposition = [_connection_to_proto(cxn, bloq_to_idx) for cxn in cbloq.connections]
        except (NotImplementedError, KeyError):
            # NotImplementedError is raised if `bloq` does not have a decomposition.
            # KeyError is raises if `bloq` has a decomposition but we do not wish to serialize it
            # because of conditions checked by `pred` and `max_depth`.
            decomposition = None

        try:
            bloq_counts = {
                bloq_to_idx[b]: args_to_proto.int_or_sympy_to_proto(c)
                for c, b in bloq.bloq_counts(SympySymbolAllocator())
            }
        except (NotImplementedError, KeyError):
            # NotImplementedError is raised if `bloq` does not implement bloq_counts.
            # KeyError is raises if `bloq` has `bloq_counts` but we do not wish to serialize it
            # because of conditions checked by `pred` and `max_depth`.
            bloq_counts = None

        library.table.add(
            bloq_id=bloq_id,
            decomposition=decomposition,
            bloq_counts=bloq_counts,
            bloq=_bloq_to_proto(bloq, bloq_to_idx=bloq_to_idx),
        )
    return library


def _iter_fields(bloq: Bloq):
    """Yields fields of `bloq` iff `type(bloq)` is implemented using `dataclasses` or `attr`."""
    if dataclasses.is_dataclass(type(bloq)):
        for field in dataclasses.fields(bloq):
            yield field
    elif attrs.has(type(bloq)):
        for field in attrs.fields(type(bloq)):
            yield field


def _connection_to_proto(cxn: Connection, bloq_to_idx: Dict[Bloq, int]):
    return bloq_pb2.Connection(
        left=_soquet_to_proto(cxn.left, bloq_to_idx), right=_soquet_to_proto(cxn.right, bloq_to_idx)
    )


def _soquet_to_proto(soq: Soquet, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Soquet:
    if isinstance(soq.binst, DanglingT):
        return bloq_pb2.Soquet(
            dangling_t=repr(soq.binst),
            register=registers_to_proto.register_to_proto(soq.reg),
            index=soq.idx,
        )
    else:
        return bloq_pb2.Soquet(
            bloq_instance=_bloq_instance_to_proto(soq.binst, bloq_to_idx),
            register=registers_to_proto.register_to_proto(soq.reg),
            index=soq.idx,
        )


def _bloq_instance_to_proto(
    binst: BloqInstance, bloq_to_idx: Dict[Bloq, int]
) -> bloq_pb2.BloqInstance:
    return bloq_pb2.BloqInstance(instance_id=binst.i, bloq_id=bloq_to_idx[binst.bloq])


def _add_bloq_to_dict(bloq: Bloq, bloq_to_idx: Dict[Bloq, int]):
    """Adds `{bloq: len(bloq_to_idx)}` to `bloq_to_idx` dictionary if it doesn't exist already."""
    if bloq not in bloq_to_idx:
        next_idx = len(bloq_to_idx)
        bloq_to_idx[bloq] = next_idx


def _cbloq_dot_bloq_instances(cbloq: CompositeBloq) -> List[BloqInstance]:
    """Equivalent to `cbloq.bloq_instances`, but preserves insertion order among Bloq instances."""
    ret = {}
    for cxn in cbloq._cxns:
        for soq in [cxn.left, cxn.right]:
            if not isinstance(soq.binst, DanglingT):
                ret[soq.binst] = 0
    return list(ret.keys())


def _populate_bloq_to_idx(
    bloq: Bloq, bloq_to_idx: Dict[Bloq, int], pred: Callable[[BloqInstance], bool], max_depth: int
):
    """Recursively track all primitive Bloqs to be serialized, as part of `bloq_to_idx` dictionary."""

    assert bloq in bloq_to_idx
    if max_depth > 0:
        # Decompose the current Bloq and track it's decomposed Bloqs.
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            for binst in _cbloq_dot_bloq_instances(cbloq):
                _add_bloq_to_dict(binst.bloq, bloq_to_idx)
                if pred(binst):
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, max_depth - 1)
                else:
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, 0)
        except NotImplementedError:
            # NotImplementedError is raised if `bloq` does not have a decomposition.
            ...

        # Approximately decompose the current Bloq and it's decomposed Bloqs.
        try:
            for _, subbloq in bloq.bloq_counts(SympySymbolAllocator()):
                _add_bloq_to_dict(subbloq, bloq_to_idx)
                _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)

        except NotImplementedError:
            # NotImplementedError is raised if `bloq` does not implement bloq_counts.
            ...

    # If the current Bloq contains other Bloqs as sub-bloqs, add them to the `bloq_to_idx` dict.
    # This is only supported for Bloqs implemented as dataclasses / attrs.
    for field in _iter_fields(bloq):
        subbloq = getattr(bloq, field.name)
        if isinstance(subbloq, Bloq):
            _add_bloq_to_dict(subbloq, bloq_to_idx)
            _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)


def _bloq_to_proto(bloq: Bloq, *, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Bloq:
    try:
        t_complexity = annotations_to_proto.t_complexity_to_proto(bloq.t_complexity())
    except:
        t_complexity = None

    return bloq_pb2.Bloq(
        name=bloq.pretty_name(),
        registers=registers_to_proto.registers_to_proto(bloq.signature),
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
    return args_to_proto.arg_to_proto(name=name, val=val)
