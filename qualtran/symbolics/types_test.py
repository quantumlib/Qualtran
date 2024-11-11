#  Copyright 2024 Google LLC
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

import pytest
import sympy

from qualtran.symbolics import is_symbolic, Shaped, slen


@pytest.mark.parametrize(
    "shape",
    [(4,), (1, 2), (1, 2, 3), (sympy.Symbol('n'),), (sympy.Symbol('n'), sympy.Symbol('m'), 100)],
)
def test_shaped(shape: tuple[int, ...]):
    shaped = Shaped(shape=shape)
    assert is_symbolic(shaped)
    assert slen(shaped) == shape[0]
