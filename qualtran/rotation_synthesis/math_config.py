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

"""A module that defines various mathematical config"""

import math
from typing import Callable, cast

import attrs
import mpmath
import numpy as np

from qualtran.rotation_synthesis._typing import Real


@attrs.frozen(hash=False)
class MathConfig:
    name: str = "vanilla"
    zero: Real = 0.0
    one: Real = 1.0
    pi: Real = math.pi
    sqrt2: Real = math.sqrt(2)
    _log: Callable[[Real], Real] = math.log
    _sqrt: Callable[[Real], Real] = math.sqrt
    _isclose: Callable[[Real, Real], bool] = cast(Callable[[Real, Real], bool], np.isclose)
    _cos: Callable[[Real], Real] = math.cos
    _sin: Callable[[Real], Real] = math.sin
    _floor: Callable[[Real], int] = math.floor
    _ceil: Callable[[Real], int] = math.ceil
    _arctan2: Callable[[Real, Real], Real] = math.atan2
    _arcsin: Callable[[Real], Real] = math.asin
    _arccos: Callable[[Real], Real] = math.acos
    _number: Callable[[Real], Real] = float

    def number(self, x) -> Real:
        return self._number(x)

    def log(self, x: Real) -> Real:
        return self._log(x)

    def sqrt(self, x: Real) -> Real:
        return self._sqrt(x)

    def isclose(self, x: Real, y: Real) -> bool:
        return self._isclose(x, y)

    def cos(self, x: Real) -> Real:
        return self._cos(x)

    def sin(self, x: Real) -> Real:
        return self._sin(x)

    def floor(self, x: Real) -> int:
        return self._floor(x)

    def arctan2(self, y: Real, x: Real) -> Real:
        return self._arctan2(y, x)

    def ceil(self, x: Real) -> int:
        return self._ceil(x)

    def __hash__(self) -> int:
        return hash(self.name)

    def arcsin(self, x: Real) -> Real:
        return self._arcsin(x)

    def arccos(self, x: Real) -> Real:
        return self._arccos(x)


NumpyConfig = MathConfig(
    "numpy",
    np.float64(0),
    np.float64(1),
    np.pi,
    np.sqrt(2, dtype=np.float64),
    np.log,
    np.sqrt,
    cast(Callable[[Real, Real], bool], np.isclose),
    np.cos,
    np.sin,
    lambda x: int(np.floor(x)),
    lambda x: int(np.ceil(x)),
    np.arctan2,
    np.arcsin,
    np.arccos,
    np.float128,
)


def isclose(x, y, eps=1e-6):  # mpmath.power(10, -mpmath.mp.dps//2+2)):
    return abs(x - y) < eps or abs(x - y) < eps * abs(x)


def with_dps(dps: int) -> MathConfig:
    """Creates an mpmath based MathConfig with the given digits of precision.

    Args:
        dps: The target digits of precision.

    Returns:
        A MathConfig backed by mpmath.
    """
    mpmath.mp.dps = dps
    return MathConfig(
        f"mpmath_{dps}",
        mpmath.mpf(0),
        mpmath.mpf(1),
        mpmath.pi,
        mpmath.sqrt(2),
        mpmath.log,
        mpmath.sqrt,
        lambda x, y: abs(x - y) <= mpmath.power(10, -mpmath.mp.dps // 2 + 2),
        mpmath.cos,
        mpmath.sin,
        lambda x: int(mpmath.floor(x)),
        lambda x: int(mpmath.ceil(x)),
        mpmath.atan2,
        mpmath.asin,
        mpmath.acos,
        mpmath.mpf,
    )
