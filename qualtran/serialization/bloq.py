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

import dataclasses
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import attrs
import cirq
import numpy as np
import sympy

from qualtran import (
    Bloq,
    BloqInstance,
    CompositeBloq,
    Connection,
    CtrlSpec,
    DanglingT,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    LeftDangle,
    QDType,
    Register,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran.bloqs.cryptography.ecc import ECPoint
from qualtran.protos import bloq_pb2
from qualtran.serialization import (
    annotations,
    args,
    ctrl_spec,
    data_types,
    ec_point,
    registers,
    resolver_dict,
    sympy_to_proto,
)


def arg_to_proto(*, name: str, val: Any) -> bloq_pb2.BloqArg:
    if isinstance(val, int):
        return bloq_pb2.BloqArg(name=name, int_val=val)
    if isinstance(val, float):
        return bloq_pb2.BloqArg(name=name, float_val=val)
    if isinstance(val, str):
        return bloq_pb2.BloqArg(name=name, string_val=val)
    if isinstance(val, sympy.Expr):
        return bloq_pb2.BloqArg(name=name, sympy_expr=sympy_to_proto.sympy_expr_to_proto(val))
    if isinstance(val, Register):
        return bloq_pb2.BloqArg(name=name, register=registers.register_to_proto(val))
    if isinstance(val, tuple) and all(isinstance(x, Register) for x in val):
        return bloq_pb2.BloqArg(name=name, registers=registers.registers_to_proto(val))
    if isinstance(val, cirq.Gate):
        gzipped = cirq.to_json_gzip(val)
        if gzipped is None:
            raise ValueError(f"Cannot gzip {val}")
        return bloq_pb2.BloqArg(name=name, cirq_json_gzip=gzipped)
    if isinstance(val, QDType):
        return bloq_pb2.BloqArg(name=name, qdata_type=data_types.data_type_to_proto(val))
    if isinstance(val, CtrlSpec):
        return bloq_pb2.BloqArg(name=name, ctrl_spec=ctrl_spec.ctrl_spec_to_proto(val))
    if isinstance(val, (np.ndarray, tuple, list)):
        return bloq_pb2.BloqArg(name=name, ndarray=args.ndarray_to_proto(np.asarray(val)))
    if np.iscomplexobj(val):
        return bloq_pb2.BloqArg(name=name, complex_val=args.complex_to_proto(val))
    if isinstance(val, ECPoint):
        return bloq_pb2.BloqArg(name=name, ec_point=ec_point.ec_point_to_proto(val))
    raise ValueError(f"Cannot serialize {val} of unknown type {type(val)}")


def arg_from_proto(arg: bloq_pb2.BloqArg) -> Dict[str, Any]:
    if arg.HasField("int_val"):
        return {arg.name: arg.int_val}
    if arg.HasField("float_val"):
        return {arg.name: arg.float_val}
    if arg.HasField("string_val"):
        return {arg.name: arg.string_val}
    if arg.HasField("sympy_expr"):
        return {arg.name: sympy_to_proto.sympy_expr_from_proto(arg.sympy_expr)}
    if arg.HasField("register"):
        return {arg.name: registers.register_from_proto(arg.register)}
    if arg.HasField("registers"):
        return {arg.name: registers.registers_from_proto(arg.registers)}
    if arg.HasField("cirq_json_gzip"):
        return {arg.name: cirq.read_json_gzip(gzip_raw=arg.cirq_json_gzip)}
    if arg.HasField("qdata_type"):
        return {arg.name: data_types.data_type_from_proto(arg.qdata_type)}
    if arg.HasField("ctrl_spec"):
        return {arg.name: ctrl_spec.ctrl_spec_from_proto(arg.ctrl_spec)}
    if arg.HasField("ndarray"):
        return {arg.name: args.ndarray_from_proto(arg.ndarray)}
    if arg.HasField("complex_val"):
        return {arg.name: args.complex_from_proto(arg.complex_val)}
    if arg.HasField("ec_point"):
        return {arg.name: ec_point.ec_point_from_proto(arg.ec_point)}
    raise ValueError(f"Cannot deserialize {arg=}")


class _BloqLibDeserializer:
    def __init__(self, lib: bloq_pb2.BloqLibrary):
        self.id_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition] = {
            b.bloq_id: b for b in lib.table
        }
        self.id_to_bloq: Dict[int, Bloq] = {}
        self.dangling_to_singleton = {"LeftDangle": LeftDangle, "RightDangle": RightDangle}

    def bloq_id_to_bloq(self, bloq_id: int):
        """Constructs a Bloq corresponding to a `bloq_id` given an `id_to_proto` mapping.

        The `id_to_proto` mappping is constructed using a `BloqLibrary`.
        The `id_to_bloq` mapping acts as a cache to avoid redundant deserialization of Bloqs.
        """
        if bloq_id in self.id_to_bloq:
            return self.id_to_bloq[bloq_id]
        bloq_proto: bloq_pb2.BloqLibrary.BloqWithDecomposition = self.id_to_proto[bloq_id]
        if bloq_proto.bloq.name.endswith('.CompositeBloq'):
            self.id_to_bloq[bloq_id] = CompositeBloq(
                connections=tuple(
                    self._connection_from_proto(cxn) for cxn in bloq_proto.decomposition
                ),
                signature=Signature(registers.registers_from_proto(bloq_proto.bloq.registers)),
            )
        elif bloq_proto.bloq.name in resolver_dict.RESOLVER_DICT:
            kwargs = {}
            for arg in bloq_proto.bloq.args:
                if arg.HasField('subbloq'):
                    kwargs[arg.name] = self.bloq_id_to_bloq(arg.subbloq)
                else:
                    kwargs.update(arg_from_proto(arg))
            self.id_to_bloq[bloq_id] = self._construct_bloq(bloq_proto.bloq.name, **kwargs)
        else:
            raise ValueError(f"Unable to find a Bloq corresponding to {bloq_proto.bloq.name=}")
        return self.id_to_bloq[bloq_id]

    def _construct_bloq(self, name: str, **kwargs):
        """Construct a Bloq using serialized name and BloqArgs."""
        return resolver_dict.RESOLVER_DICT[name](**kwargs)  # type: ignore[operator]

    def _connection_from_proto(self, cxn: bloq_pb2.Connection) -> Connection:
        return Connection(
            left=self._soquet_from_proto(cxn.left), right=self._soquet_from_proto(cxn.right)
        )

    def _soquet_from_proto(self, soq: bloq_pb2.Soquet) -> Soquet:
        binst: Union[BloqInstance, DanglingT] = (
            self.dangling_to_singleton[soq.dangling_t]
            if soq.HasField('dangling_t')
            else BloqInstance(
                i=soq.bloq_instance.instance_id,
                bloq=self.bloq_id_to_bloq(soq.bloq_instance.bloq_id),
            )
        )
        return Soquet(
            binst=binst, reg=registers.register_from_proto(soq.register), idx=tuple(soq.index)
        )


def bloqs_from_proto(lib: bloq_pb2.BloqLibrary) -> List[Bloq]:
    """Deserializes a BloqLibrary as a list of Bloqs."""
    deserializer = _BloqLibDeserializer(lib)
    return [deserializer.bloq_id_to_bloq(bloq.bloq_id) for bloq in lib.table]


def bloqs_to_proto(
    *bloqs: Bloq,
    name: str = '',
    pred: Callable[[BloqInstance], bool] = lambda _: True,
    max_depth: int = 1,
) -> bloq_pb2.BloqLibrary:
    """Serializes one or more Bloqs as a `BloqLibrary`.

    A `BloqLibrary` contains multiple bloqs and their hierarchical decompositions. Since
    decompositions can use bloq objects that are not explicitly listed in the `bloqs` argument to
    this function, this routine will recursively (up to `max_depth`) add any bloq objects
    encountered in decompositions to the bloq library.

    For bloqs within `max_depth` decompositions of the bloqs passed explicitly to this function,
    we perform a full serialization: each bloq is serialized with its decomposition and resource
    costs. For bloqs encountered only through references from full decompositions, or through
    bloqs included as compiled-time classical parameters to bloqs; we perform a "shallow"
    serialization where only the bloq, its signature, and its attributes are included in the
    BloqLibrary.
    """

    # The bloq library uses a unique integer index as a simple address for each bloq object.
    # Set up this mapping and populate it by recursively searching for subbloqs.
    # Each value is an (id: bool, shallow: bool) tuple, where the second entry can be set to
    # `True` for bloqs that need to be referred to but do not need a full serialization.
    bloq_to_id_ext: Dict[Bloq, Tuple[int, bool]] = {}
    for bloq in bloqs:
        _assign_bloq_an_id(bloq, bloq_to_id_ext, shallow=True)
        _search_for_subbloqs(bloq, bloq_to_id_ext, pred, max_depth)

    bloq_to_id = {bloq: bloq_id for bloq, (bloq_id, shallow) in bloq_to_id_ext.items()}

    # Decompose[..]Error is raised if `bloq` does not have a decomposition.
    # KeyError is raised if `bloq` has a decomposition, but we do not wish to serialize it
    # because of conditions checked by `pred` and `max_depth`.
    stop_recursing_exceptions = (DecomposeNotImplementedError, DecomposeTypeError, KeyError)

    # `bloq_to_id` contains a list of all bloqs that should be serialized.
    library = bloq_pb2.BloqLibrary(name=name)
    for bloq, (bloq_id, shallow) in bloq_to_id_ext.items():
        if shallow:
            decomposition = None
        else:
            try:
                cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
                decomposition = [_connection_to_proto(cxn, bloq_to_id) for cxn in cbloq.connections]
            except stop_recursing_exceptions:
                decomposition = None

        if shallow:
            bloq_counts = None
        else:
            try:
                bloq_counts = {
                    bloq_to_id[b]: args.int_or_sympy_to_proto(c)
                    for b, c in sorted(
                        bloq.bloq_counts().items(), key=lambda x: type(x[0]).__name__
                    )
                }
            except stop_recursing_exceptions:
                bloq_counts = None

        library.table.add(
            bloq_id=bloq_to_id[bloq],
            decomposition=decomposition,
            bloq_counts=bloq_counts,
            bloq=_bloq_to_proto(bloq, bloq_to_id=bloq_to_id, shallow=shallow),
        )
    return library


def _iter_fields(bloq: Bloq):
    """Yields fields of `bloq` iff `type(bloq)` is implemented using `dataclasses` or `attr`.

    The method only yields Fields corresponding to attributes that are part of the __init__ method
    of the bloq. This ensures that for attrs / dataclasses based Bloqs that have a custom init
    method, we yield only the fields that are accepted by the constructor (eg: `IntState` Bloq).
    Note that this is a hacky solution and a more generalized long-term approach would be to have
    a protocol to query init params for each class and use them as Bloq args during
    serialization / deserialization.
    """

    if dataclasses.is_dataclass(bloq):
        for field in dataclasses.fields(bloq):
            if field.name in inspect.signature(type(bloq).__init__).parameters:
                yield field
    elif attrs.has(type(bloq)):
        for field in attrs.fields(type(bloq)):  # type: ignore[arg-type]
            if field.name in inspect.signature(type(bloq).__init__).parameters:
                yield field


def _connection_to_proto(cxn: Connection, bloq_to_id: Dict[Bloq, int]):
    return bloq_pb2.Connection(
        left=_soquet_to_proto(cxn.left, bloq_to_id), right=_soquet_to_proto(cxn.right, bloq_to_id)
    )


def _soquet_to_proto(soq: Soquet, bloq_to_id: Dict[Bloq, int]) -> bloq_pb2.Soquet:
    if isinstance(soq.binst, DanglingT):
        return bloq_pb2.Soquet(
            dangling_t=repr(soq.binst), register=registers.register_to_proto(soq.reg), index=soq.idx
        )
    else:
        return bloq_pb2.Soquet(
            bloq_instance=_bloq_instance_to_proto(soq.binst, bloq_to_id),
            register=registers.register_to_proto(soq.reg),
            index=soq.idx,
        )


def _bloq_instance_to_proto(
    binst: BloqInstance, bloq_to_id: Dict[Bloq, int]
) -> bloq_pb2.BloqInstance:
    return bloq_pb2.BloqInstance(instance_id=binst.i, bloq_id=bloq_to_id[binst.bloq])


def _assign_bloq_an_id(bloq: Bloq, bloq_to_id: Dict[Bloq, Tuple[int, bool]], shallow: bool = False):
    """Assigns a new index for `bloq` and records it into the `bloq_to_id` mapping."""
    if bloq in bloq_to_id:
        # Keep the same id, but if anyone requests a non-shallow serialization; do it.
        bloq_id, existing_shallow = bloq_to_id[bloq]
        bloq_to_id[bloq] = (bloq_id, (existing_shallow and shallow))
    else:
        next_idx = len(bloq_to_id)
        bloq_to_id[bloq] = next_idx, shallow


def _cbloq_ordered_bloq_instances(cbloq: CompositeBloq) -> List[BloqInstance]:
    """Equivalent to `cbloq.bloq_instances`, but preserves insertion order among bloq instances."""
    ret = {}
    for cxn in cbloq.connections:
        for soq in [cxn.left, cxn.right]:
            if not isinstance(soq.binst, DanglingT):
                ret[soq.binst] = 0
    return list(ret.keys())


def _search_for_subbloqs(
    bloq: Bloq,
    bloq_to_id: Dict[Bloq, Tuple[int, bool]],
    pred: Callable[[BloqInstance], bool],
    max_depth: int,
) -> None:
    """Recursively finds all bloqs.

    This function inspects `bloq`'s 1) decomposition, 2) call graph, and 3) attributes list for
    any bloq objects. For each bloq object that we discover, we will recurse on it.

    All bloqs are stored in `bloq_to_id` as we find them.

    `max_depth` will be decremented for each level of recursion. If `max_depth` reaches zero,
    only the bloqs attributes will be searched.

    `pred` is evaluated on each bloq instance in the bloq's decomposition. If it evaluates to
    `False`, recursion will stop *after*  processing the sub-bloq and its attributes.

    `pred` is not used when querying the call graph  nor when inspecting the bloq's attributes.
    """

    if max_depth > 0:
        # Ensure full serialization of this bloq
        _assign_bloq_an_id(bloq, bloq_to_id, shallow=False)

        # Search the bloq's decomposition
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            for binst in _cbloq_ordered_bloq_instances(cbloq):
                subbloq = binst.bloq
                _assign_bloq_an_id(subbloq, bloq_to_id, shallow=True)
                if pred(binst):
                    _search_for_subbloqs(subbloq, bloq_to_id, pred, max_depth - 1)
                else:
                    _search_for_subbloqs(subbloq, bloq_to_id, pred, 0)
        except (DecomposeTypeError, DecomposeNotImplementedError) as e:
            # No decomposition, nothing to recurse on.
            pass

        # Search the bloq's call graph
        try:
            for subbloq, _ in bloq.bloq_counts().items():
                _assign_bloq_an_id(subbloq, bloq_to_id, shallow=True)
                _search_for_subbloqs(subbloq, bloq_to_id, pred, 0)
        except NotImplementedError:
            # No call graph, nothing to recurse on.
            pass

    # Search the bloq's attributes.
    # This is only supported for Bloqs implemented as dataclasses / attrs.
    for field in _iter_fields(bloq):
        subbloq = getattr(bloq, field.name)
        if isinstance(subbloq, Bloq):
            _assign_bloq_an_id(subbloq, bloq_to_id, shallow=True)
            _search_for_subbloqs(subbloq, bloq_to_id, pred, 0)


def _bloq_to_proto(
    bloq: Bloq, *, bloq_to_id: Dict[Bloq, int], shallow: bool = False
) -> bloq_pb2.Bloq:
    if shallow:
        t_complexity = None
    else:
        try:
            t_complexity = annotations.t_complexity_to_proto(bloq.t_complexity())
        except (DecomposeTypeError, DecomposeNotImplementedError, TypeError):
            t_complexity = None

    name = bloq.__module__ + "." + bloq.__class__.__qualname__
    return bloq_pb2.Bloq(
        name=name,
        registers=registers.registers_to_proto(bloq.signature),
        t_complexity=t_complexity,
        args=_bloq_args_to_proto(bloq, bloq_to_id=bloq_to_id),
    )


def _bloq_args_to_proto(
    bloq: Bloq, *, bloq_to_id: Dict[Bloq, int]
) -> Optional[List[bloq_pb2.BloqArg]]:
    if isinstance(bloq, CompositeBloq):
        return None

    ret = [
        _bloq_arg_to_proto(name=field.name, val=getattr(bloq, field.name), bloq_to_id=bloq_to_id)
        for field in _iter_fields(bloq)
        if getattr(bloq, field.name) is not None
    ]
    return ret if ret else None


def _bloq_arg_to_proto(name: str, val: Any, bloq_to_id: Dict[Bloq, int]) -> bloq_pb2.BloqArg:
    if isinstance(val, Bloq):
        return bloq_pb2.BloqArg(name=name, subbloq=bloq_to_id[val])
    return arg_to_proto(name=name, val=val)
