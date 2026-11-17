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

import numpy as np
import pytest
import sympy
from google.protobuf.message import DecodeError

from qualtran.protos import args_pb2, sympy_pb2
from qualtran.serialization import args


@pytest.mark.parametrize('arg', [1, sympy.Symbol('a') * sympy.Symbol('b') + sympy.Symbol('c') / 10])
def test_int_or_sympy_to_proto(arg):
    proto = args.int_or_sympy_to_proto(arg)
    if isinstance(arg, sympy.Expr):
        assert isinstance(proto.sympy_expr, sympy_pb2.Term)
    arg_from_proto = args.int_or_sympy_from_proto(proto)
    assert arg_from_proto == arg


def test_int_or_sympy_rejects_raw_string_wire_payload():
    payload = b"__import__('os').system('echo pwn')"
    # Wire tag 0x12 = field 2 (sympy_expr), wire type 2 (length-delimited).
    wire_bytes = b"\x12" + bytes([len(payload)]) + payload
    with pytest.raises(DecodeError):
        args_pb2.IntOrSympy.FromString(wire_bytes)


def test_ndarray_to_proto():
    x = np.random.random(100)
    proto = args.ndarray_to_proto(x)
    x_from_proto = args.ndarray_from_proto(proto)
    assert np.allclose(x, x_from_proto)


@pytest.mark.parametrize('arg', [1j, np.complex64(1j)])
def test_complex_to_proto(arg):
    proto = args.complex_to_proto(arg)
    arg_from_proto = args.complex_from_proto(proto)
    assert arg_from_proto == arg
