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


@pytest.mark.parametrize('arg', [1, sympy.Symbol('a') * sympy.Symbol('b') + sympy.Symbol('c') / 10])
def test_int_or_sympy_to_proto(arg):
    proto = args.int_or_sympy_to_proto(arg)
    arg_from_proto = args.int_or_sympy_from_proto(proto)
    assert arg_from_proto == arg


def test_ndarray_to_proto():
    x = np.random.random(100)
    proto = args.ndarray_to_proto(x)
    x_from_proto = args.ndarray_from_proto(proto)
    assert np.allclose(x, x_from_proto)
