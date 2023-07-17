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

from qualtran.serialization import args


@pytest.mark.parametrize(
    'arg',
    [
        1,
        2.0,
        'hello world',
        sympy.Symbol('a') * sympy.Symbol('b') + sympy.Symbol('c') / 10,
        np.array([*range(100)], dtype=np.complex128).reshape((10, 10)),
    ],
)
def test_arg_to_proto_round_trip(arg):
    proto = args.arg_to_proto(name='custom_name', val=arg)
    arg_dict = args.arg_from_proto(proto)
    if isinstance(arg, np.ndarray):
        arr = arg_dict['custom_name']
        assert arr.shape == arg.shape
        assert arr.dtype == arg.dtype
        assert np.allclose(arr, arg)
    else:
        assert arg_dict['custom_name'] == arg
