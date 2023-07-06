from typing import Any, Dict

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr

from qualtran.api import args_pb2


def arg_to_proto(*, name: str, val: Any) -> args_pb2.BloqArg:
    if isinstance(val, int):
        return args_pb2.BloqArg(name=name, int_val=val)
    if isinstance(val, float):
        return args_pb2.BloqArg(name=name, float_val=val)
    if isinstance(val, str):
        return args_pb2.BloqArg(name=name, string_val=val)
    if isinstance(val, sympy.Expr):
        return args_pb2.BloqArg(name=name, sympy_expr=str(val))
    if isinstance(val, np.ndarray):
        return args_pb2.BloqArg(name=name, ndarray=_ndarray_to_proto(val))
    raise ValueError(f"Cannot serialize {val} of unknown type {type(val)}")


def proto_to_arg(arg: args_pb2.BloqArg) -> Dict[str, Any]:
    if arg.HasField("int_val"):
        return {arg.name: arg.int_val}
    if arg.HasField("float_val"):
        return {arg.name: arg.float_val}
    if arg.HasField("string_val"):
        return {arg.name: arg.string_val}
    if arg.HasField("sympy_expr"):
        return {arg.name: parse_expr(arg.sympy_expr)}
    if arg.HasField("ndarray"):
        return {arg.name: _proto_to_ndarray(arg.ndarray)}


def _ndarray_to_proto(arr: np.ndarray) -> args_pb2.NDArray:
    return args_pb2.NDArray(shape=arr.shape, dtype=f'np.{arr.dtype!r}', data=arr.tobytes())


def _proto_to_ndarray(arr: args_pb2.NDArray) -> np.ndarray:
    return np.ndarray(arr.shape, eval(arr.dtype), arr.data)
