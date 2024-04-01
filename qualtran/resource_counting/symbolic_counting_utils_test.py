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
import sympy
from sympy.codegen.cfunctions import log2 as sympy_log2

from qualtran.resource_counting.symbolic_counting_utils import ceil, log2, smax


def test_log2():
    assert log2(sympy.Symbol('x')) == sympy_log2(sympy.Symbol('x'))
    assert log2(sympy.Number(10)) == sympy_log2(sympy.Number(10))
    assert log2(10) == np.log2(10)


def test_ceil():
    assert ceil(sympy.Symbol('x')) == sympy.ceiling(sympy.Symbol('x'))
    assert ceil(sympy.Number(10.123)) == sympy.ceiling(sympy.Number(11))
    assert isinstance(ceil(sympy.Number(10.123)), sympy.Basic)
    assert ceil(10.123) == 11


def test_smax():
    assert smax(1, 2) == 2
    assert smax(1.1, 2.2) == 2.2
    assert smax(1, sympy.Symbol('x')) == sympy.Max(1, sympy.Symbol('x'))
