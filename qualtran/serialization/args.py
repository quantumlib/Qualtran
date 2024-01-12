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
from io import BytesIO
from typing import Any, Dict, Union

import cirq
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr

from qualtran.protos import args_pb2


def int_or_sympy_to_proto(val: Union[int, sympy.Expr]) -> args_pb2.IntOrSympy:
    return (
        args_pb2.IntOrSympy(sympy_expr=str(val))
        if isinstance(val, sympy.Expr)
        else args_pb2.IntOrSympy(int_val=val)
    )


def int_or_sympy_from_proto(val: args_pb2.IntOrSympy) -> Union[int, sympy.Expr]:
    return val.int_val if val.HasField('int_val') else parse_expr(val.sympy_expr)


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
        return args_pb2.BloqArg(name=name, ndarray=ndarray_to_proto(val))
    if isinstance(val, cirq.Gate):
        return args_pb2.BloqArg(name=name, cirq_json_gzip=cirq.to_json_gzip(val))
    raise ValueError(f"Cannot serialize {val} of unknown type {type(val)}")


def arg_from_proto(arg: args_pb2.BloqArg) -> Dict[str, Any]:
    if arg.HasField("int_val"):
        return {arg.name: arg.int_val}
    if arg.HasField("float_val"):
        return {arg.name: arg.float_val}
    if arg.HasField("string_val"):
        return {arg.name: arg.string_val}
    if arg.HasField("sympy_expr"):
        return {arg.name: parse_expr(arg.sympy_expr)}
    if arg.HasField("ndarray"):
        return {arg.name: ndarray_from_proto(arg.ndarray)}
    if arg.HasField("cirq_json_gzip"):
        return {arg.name: cirq.read_json_gzip(gzip_raw=arg.cirq_json_gzip)}
    raise ValueError(f"Cannot deserialize {arg=}")


def ndarray_to_proto(arr: np.ndarray) -> args_pb2.NDArray:
    arr_bytes = BytesIO()
    np.save(arr_bytes, arr, allow_pickle=False)
    return args_pb2.NDArray(ndarray=arr_bytes.getvalue())


def ndarray_from_proto(arr: args_pb2.NDArray) -> np.ndarray:
    arr_bytes = BytesIO(arr.ndarray)
    return np.load(arr_bytes, allow_pickle=False)
