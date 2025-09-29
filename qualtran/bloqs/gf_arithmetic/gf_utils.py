#  Copyright 2025 Google LLC
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
from typing import Sequence, Union

from galois import Poly

from qualtran import QGF
from qualtran.symbolics import is_symbolic, SymbolicInt


def qgf_converter(x: Union[QGF, int, Poly, SymbolicInt, Sequence[int]]) -> QGF:
    if isinstance(x, QGF):
        return x
    if isinstance(x, int):
        return QGF(2, x)
    if is_symbolic(x):
        return QGF(2, x)
    if isinstance(x, Poly):
        return QGF(2, x.degree, x)
    p = Poly.Degrees(x)
    return QGF(2, p.degree, p)
