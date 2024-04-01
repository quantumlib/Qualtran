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
from typing import Union

import numpy as np
import sympy
from sympy.parsing import parse_expr

from qualtran.protos import args_pb2


def int_or_sympy_to_proto(val: Union[int, sympy.Expr]) -> args_pb2.IntOrSympy:
    return (
        args_pb2.IntOrSympy(sympy_expr=str(val))
        if isinstance(val, sympy.Expr)
        else args_pb2.IntOrSympy(int_val=val)
    )


def int_or_sympy_from_proto(val: args_pb2.IntOrSympy) -> Union[int, sympy.Expr]:
    return val.int_val if val.HasField('int_val') else parse_expr(val.sympy_expr)


def ndarray_to_proto(arr: np.ndarray) -> args_pb2.NDArray:
    arr_bytes = BytesIO()
    np.save(arr_bytes, arr, allow_pickle=False)
    return args_pb2.NDArray(ndarray=arr_bytes.getvalue())


def ndarray_from_proto(arr: args_pb2.NDArray) -> np.ndarray:
    arr_bytes = BytesIO(arr.ndarray)
    return np.load(arr_bytes, allow_pickle=False)


def complex_to_proto(val: complex) -> args_pb2.Complex:
    return args_pb2.Complex(real=np.real(val), imag=np.imag(val))


def complex_from_proto(val: args_pb2.IntOrSympy) -> complex:
    return val.real + 1j * val.imag
